import os
import argparse
import torch
import numpy as np

from tqdm.auto import tqdm
from collections import Counter

from heads import MultiHead
from losses import MultiHeadTEMILoss
from metrics import accuracy_with_reassignment, nmi_geom, standartify_clusters

@torch.no_grad()
def update_teachers(students, teachers):
    momentum = 0.996
    for teacher_param, student_param in zip(teachers.parameters(), students.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1-momentum)
    
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
    print(f"Accuracy = {acc}, NMI = {nmi}")
    print(Counter(pseudo_labels))

def train_epoch(optimizer, students, teachers, embeds, knn_indices, epoch, batch_size, criterion):
    students.train()
    teachers.train()
    n_its = embeds.shape[0] // batch_size
    bar = tqdm(range(n_its), desc=f"Train epoch {epoch}")
    n_embeds = embeds.shape[0]
    n_neighbours = knn_indices.shape[1]
    epoch_loss = 0

    for _ in bar:
        optimizer.zero_grad()

        # Sample random knn pairs for batch
        samples = torch.randint(size=(batch_size,), high=n_embeds)
        neighbours = torch.randint(size=(batch_size,), high=n_neighbours)
        x1 = embeds[samples, :]
        x2 = embeds[neighbours, :]

        # Compute probabilities
        sprobs = list(zip(students(x1), students(x2)))
        tprobs = list(zip(teachers(x1), teachers(x2)))

        # Calculate loss
        losses = criterion(sprobs, tprobs)
        avg_loss = sum(losses) / len(losses)
        epoch_loss += avg_loss.item() / n_its

        # Backpropagation for students
        avg_loss.backward()
        optimizer.step()

        # Update teachers
        update_teachers(teachers=teachers, students=students)

    return epoch_loss

def train(args, students, teachers, embeds, knn_indices):
    labels = load_labels(args.labels_path) 
    optimizer = torch.optim.AdamW(students.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = MultiHeadTEMILoss(n_heads=args.n_heads, n_classes=args.n_clusters, momentum=0.99)

    for i in range(args.n_epochs):
        loss = train_epoch(
            optimizer=optimizer,
            students=students,
            teachers=teachers,
            embeds=embeds,
            knn_indices=knn_indices,
            epoch=i,
            batch_size=args.batch_size,
            criterion=criterion
        )
        print(f"Epoch {i}, loss = {loss}")
        eval_head(
            teachers=teachers, 
            teacher_idx=0, 
            embeds=embeds, 
            labels=labels, 
            batch_size=args.batch_size
        )

def load_data(data_dir):
    embeds = torch.load(os.path.join(data_dir, 'embeds.pth'))
    knn_indices = torch.load(os.path.join(data_dir, 'knn.pth'))
    return embeds, knn_indices

def main(args):
    embeds, knn_indices = load_data(args.data_dir) 
    embeds = (embeds - embeds.mean(dim=0)) / embeds.std(dim=0)
    n_embed = embeds.shape[1] 
    students = MultiHead(n_heads=args.n_heads, n_embed=n_embed, n_hidden=args.n_hidden, n_classes=args.n_clusters) 
    teachers = MultiHead(n_heads=args.n_heads, n_embed=n_embed, n_hidden=args.n_hidden, n_classes=args.n_clusters) 
    students.load_state_dict(teachers.state_dict()) # TODO: Check if needed

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
        '--batch_size', default=64, type=int,
        help="Minibatch size"
    )
    parser.add_argument(
        '--lr', default=1e-4, type=float,
        help="Learning rate"
    )
    main(parser.parse_args())

#  python train.py --data_dir="embeds\DCASE2018_TASK5-PaSST" --n_clusters=9 --labels_path="data\labels.npy" --n_epochs=100 --n_heads=1