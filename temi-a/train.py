import os
import argparse
import torch
import numpy as np

from tqdm.auto import tqdm
from collections import Counter

from heads import MultiHead
from losses import MultiHeadTEMILoss, MultiHeadWPMILoss
from metrics import accuracy_with_reassignment, nmi_geom, standartify_clusters

@torch.no_grad()
def update_teachers(students, teachers, momentum):
    for teacher_param, student_param in zip(teachers.parameters(), students.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.detach().data, alpha=1-momentum)
    
def load_labels(labels_path):
    labels = None
    with open(labels_path, 'rb') as f:
        labels = np.load(f)
    
    ld = {label: i for i, label in enumerate(set(labels))} 
    return np.array([ld[label] for label in labels])

def predict_labels(teachers, teacher_idx, embeds, batch_size):
    n_batches = (embeds.shape[0] + batch_size - 1) // batch_size
    labels = []

    for i in range(n_batches):
        batch = embeds[i * batch_size: (i + 1)*batch_size]
        probs = teachers(batch)[teacher_idx]
        labels.extend(list(torch.argmax(probs, dim=1)))
    
    return np.array(labels)

def eval_head(teachers, teacher_idx, embeds, labels, batch_size):
    pseudo_labels = standartify_clusters(predict_labels(teachers, teacher_idx, embeds, batch_size))
    acc = accuracy_with_reassignment(labels, pseudo_labels)
    nmi = nmi_geom(labels, pseudo_labels)
    print(Counter(pseudo_labels))
    return acc, nmi

def train_epoch(
    optimizer, 
    students, 
    teachers, 
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    embeds, 
    knn_indices, 
    epoch, 
    batch_size, 
    criterion, 
    n_heads
):
    students.train()
    teachers.train()
    n_its_per_epoch = embeds.shape[0] // batch_size
    bar = tqdm(range(n_its_per_epoch), desc=f"Train epoch {epoch}")
    n_embeds = embeds.shape[0]
    n_neighbours = knn_indices.shape[1]
    epoch_losses = [0] * n_heads 

    for it in bar:
        # update weight decay and learning rate according to their schedule
        it = n_its_per_epoch * epoch + it  # global training iteration        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Sample random knn pairs for batch
        samples = torch.randint(size=(batch_size,), high=n_embeds)
        neighbours = torch.randint(size=(batch_size,), high=n_neighbours)
        neighbour_indices = knn_indices[samples, neighbours]
        x1 = embeds[samples, :]
        x2 = embeds[neighbour_indices, :]

        # Compute probabilities
        sprobs = list(zip(students(x1), students(x2)))
        tprobs = list(zip(teachers(x1), teachers(x2)))

        # Calculate loss
        losses = criterion(sprobs, tprobs)
        avg_loss = sum(losses) / len(losses)
        epoch_losses = [l1 + l2.item() / n_its_per_epoch for l1, l2 in zip(epoch_losses, losses)]

        # Backpropagation for students
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        # Update teachers
        update_teachers(teachers=teachers, students=students, momentum=momentum_schedule[it])

    return epoch_losses

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train(args, students, teachers, embeds, knn_indices):
    labels = load_labels(args.labels_path) 
    # Optimizer
    optimizer = torch.optim.AdamW(get_params_groups(students))
    # Loss function
    loss_kwargs = {
        'n_heads': args.n_heads,
        'n_classes': args.n_clusters,
        'momentum': 0.99,
        'beta': args.beta
    }
    criterion = None
    if args.loss_func == "TEMI":
        criterion = MultiHeadTEMILoss(**loss_kwargs)
    elif args.loss_func == "WPMI":
        criterion = MultiHeadWPMILoss(**loss_kwargs)
    else:
        raise ValueError("Unknown loss function")
    # Schedulers
    bs_factor = args.batch_size / 256.
    its_per_epoch = embeds.shape[0] // args.batch_size
    lr_schedule = cosine_scheduler(
        args.lr * bs_factor,  # linear scaling rule
        args.lr * bs_factor,
        args.n_epochs, its_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay,
        args.n_epochs, its_per_epoch 
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher, 
        args.momentum_teacher,
        args.n_epochs, 
        its_per_epoch
    )

    # Train loop
    best_head_idx = 0

    for i in range(args.n_epochs):
        losses = train_epoch(
            optimizer=optimizer,
            students=students,
            teachers=teachers,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            embeds=embeds,
            knn_indices=knn_indices,
            epoch=i,
            batch_size=args.batch_size,
            criterion=criterion,
            n_heads=args.n_heads
        )
        best_head_idx = min(range(len(losses)), key = lambda idx: losses[idx])
        print(f"Epoch {i}, average head loss = {sum(losses) / len(losses)}")
        acc, nmi = eval_head(
            teachers=teachers, 
            teacher_idx=best_head_idx, 
            embeds=embeds, 
            labels=labels, 
            batch_size=args.batch_size
        )
        print(f"Best head {best_head_idx}, acc={acc}, NMI={nmi}")

def load_data(data_dir):
    embeds = torch.load(os.path.join(data_dir, 'embeds.pth'))
    knn_indices = torch.load(os.path.join(data_dir, 'knn.pth'))
    return embeds, knn_indices

def main(args):
    embeds, knn_indices = load_data(args.data_dir) 
    embeds = (embeds - embeds.mean(dim=0)) / embeds.std(dim=0)
    if args.l2_norm:
        embeds /= embeds.norm(dim=-1, keepdim=True)

    n_embed = embeds.shape[1] 
    multihead_kwargs = {
        'n_heads': args.n_heads,
        'n_embed': n_embed,
        'n_hidden': args.n_hidden,
        'n_classes': args.n_clusters
    }
    students = MultiHead(**multihead_kwargs) 
    teachers = MultiHead(**multihead_kwargs) 
    students.load_state_dict(teachers.state_dict())

    for param in teachers.parameters():
        param.requires_grad = False

    train(
        args=args, 
        students=students, 
        teachers=teachers, 
        embeds=embeds, 
        knn_indices=knn_indices
    )

description = \
"""
Trains ansemble of clusterization heads using
pre-generated audio features via TEMI 
method [https://arxiv.org/abs/2303.17896].
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='train',
                    description=description 
    )
    parser.add_argument(
        '--data_dir', default="embeds", type=str,
        help="Path to your embeddings and their KNNs"
    )
    parser.add_argument(
        '--labels_path', default="", type=str,
        help="Path to labels for evaluation"
    )
    parser.add_argument(
        '--n_clusters', required=True, type=int,
        help="Number of clusters in your data"
    )
    parser.add_argument(
        '--n_heads', default=4, type=int,
        help="Number of heads to train"
    )
    parser.add_argument(
        '--n_hidden', default=512, type=int,
        help="Dimensionality of hidden layers in heads"
    )
    parser.add_argument(
        '--n_epochs', default=4, type=int,
        help="Number of epochs to train"
    )
    parser.add_argument(
        '--batch_size', default=256, type=int,
        help="Minibatch size"
    )
    parser.add_argument(
        '--lr', default=1e-4, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--l2_norm', default=False, action="store_true",
        help="Normalize embeddings using Euclidean norm"
    )
    parser.add_argument(
        '--beta', default=0.6, type=float,
        help="Beta used as a power inside PMI"
    )
    parser.add_argument(
        '--momentum_teacher', default=0.996, type=float,
        help="Momentum for updating teachers using EMA"
    )
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        '--warmup_epochs', default=20, type=int,
        help="Number of warmup epochs"
    )
    parser.add_argument(
        '--loss_func', default="TEMI", type=str, choices=["TEMI", "WPMI"],
        help="Loss function name"
    )
    main(parser.parse_args())

"""
python train.py --data_dir="embeds\DCASE2018_TASK5-PaSST" --n_clusters=9 --labels_path="data\labels.npy" --n_epochs=100 --n_heads=1

acc=0.75, nmi=0.59
python train.py --data_dir="embeds\DCASE2018_TASK5-PaSST" --n_clusters=9 --labels_path="data\labels.npy" --n_epochs=300 --n_heads=1 --lr=1e-4 --batch_size=256 --warmup_epochs=0
"""
#  