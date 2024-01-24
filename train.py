import byol_a
import dcase5
import metrics
import augmentations
import json
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import MLP
import torch.nn.functional as F

def cos_matrix(a, b):
    return torch.mm(F.normalize(a, dim=1), torch.transpose(F.normalize(b, dim=1), 0, 1))

@torch.no_grad()
def e_step(encoder, head, batch, aug, device):
    head.eval()
    aug_features = encoder(aug(batch))
    logprobs = F.log_softmax(head(aug_features), dim=1)
    B, K = logprobs.shape
    n_conf_samples = B // K
    indices = torch.topk(logprobs, n_conf_samples, dim=0, largest=True).indices
    features = encoder(batch)
    cluster_centers = torch.zeros((K, features.shape[1]), dtype=features.dtype, device=device)

    # TODO: vectorize
    for i in range(K): 
        cluster_centers[i] = torch.mean(features[indices[:, i]], dim=0)
    
    sims = cos_matrix(cluster_centers, features) # (K, B)

    indices = torch.topk(sims, n_conf_samples, dim=1, largest=False).indices.view(-1)
    # features_s = features[indices]
    proto_labels = torch.transpose(torch.arange(0, K).repeat(n_conf_samples, 1), 0, 1).flatten().to(device)

    return batch[indices], proto_labels


def m_step(encoder, head, head_optimizer, batch_s, proto_labels, aug, double_softmax=False):
    head.train()
    head_optimizer.zero_grad()
    logits = head(encoder(aug(batch_s)))
    loss = None
    if double_softmax:
        probs = F.softmax(logits, dim=1) 
        log_softmax_probs = F.log_softmax(probs, dim=1)
        loss = F.nll_loss(log_softmax_probs, proto_labels)
    else:
        loss = F.cross_entropy(logits, proto_labels)

    loss.backward()
    head_optimizer.step()

    return loss.item()

def train_epoch_2nd_stage(
    encoder, 
    heads, 
    head_optimizers, 
    dataloader, 
    e_step_aug, 
    m_step_aug, 
    device, 
    double_softmax=False
):
    epoch_losses = [0] * len(heads)
    n_samples = 0 
    for batch, _ in tqdm(dataloader, desc="Train epoch"):
        batch = batch.to(device)
        log = {}
        for i, (head, head_optimizer) in enumerate(zip(heads, head_optimizers)):
            head.to(device)
            torch.cuda.empty_cache()
            batch_s, proto_labels = e_step(
                encoder=encoder,
                head=head,
                batch=batch,
                aug=e_step_aug,
                device=device
            )
            loss = m_step(
                encoder=encoder,
                head=head,
                head_optimizer=head_optimizer,
                batch_s=batch_s,
                proto_labels=proto_labels,
                aug=m_step_aug,
                double_softmax=double_softmax
            )
            epoch_losses[i] += loss * batch.shape[0]
            n_samples += batch.shape[0]

            log.update({
                'loss_head_' + str(i): loss
            })
            torch.cuda.empty_cache()
            head.cpu()
        wandb.log(log)

    return [epoch_loss / n_samples for epoch_loss in epoch_losses]

def train_2nd_stage(
    n_epochs, 
    encoder, 
    n_heads,
    head_creator,
    head_optimizer_creator,
    dataset,
    dataloader, 
    e_step_aug, 
    m_step_aug, 
    double_softmax,
    device
):
    heads = [head_creator() for i in range(n_heads)]
    head_optimizers = [head_optimizer_creator(head) for head in heads]
    min_epoch_loss = float('inf')
    for epoch in tqdm(range(n_epochs), desc="Second stage training"):
        log = {}
        losses = train_epoch_2nd_stage(
            encoder, 
            heads, 
            head_optimizers, 
            dataloader, 
            e_step_aug, 
            m_step_aug, 
            double_softmax=double_softmax,
            device=device
        )

        for i, (head, head_optimizer) in enumerate(zip(heads, head_optimizers)):
            if min_epoch_loss > losses[i]:
                min_epoch_loss = losses[i]
                torch.save(head.state_dict(), 'best_head.pth')
            nmi, acc = test(encoder, head, dataset, device)
            suf = "_head_" + str(i)
            log.update({
                'epoch_loss' + suf: losses[i],
                'epoch_NMI' + suf: nmi,
                'epoch_accuracy' + suf: acc
            })

        log.update({
            'epoch': epoch
        })
        wandb.log(log, commit=False)

def test(encoder, head, dataset, device):
    head.eval()
    head.to(device)
    pseudo_labels = metrics.standartify_clusters(np.array(predict_pseudo_labels(encoder, head, dataset, device=device)))
    labels = dataset.get_labels()
    nmi = metrics.nmi_geom(labels, pseudo_labels)
    acc = metrics.accuracy_with_reassignment(np.array(labels), np.array(pseudo_labels))
    head.cpu()
    torch.cuda.empty_cache()
    return nmi, acc

@torch.no_grad()
def predict_pseudo_labels(encoder, head, dataset, device):
    head.eval()
    labels = torch.zeros(len(dataset), dtype=torch.long, device=device) 
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)
        logits = head(encoder(batch))
        labels[i * batch_size: (i + 1)*batch_size] = torch.argmax(logits, dim=1)
    return labels.tolist()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    torch.manual_seed(4444)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_config('cfg.json')
    dataset = dcase5.get_dataset(cfg['data_dir'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True) 
    byola_model = byol_a.get_frozen_pretrained_byola(dataset.calc_norm_stats(), device=device)
    head_creator = lambda: MLP(in_channels=3072, hidden_channels=[6144, cfg['n_clusters']])
    head_optimizer_creator = lambda head: torch.optim.Adam(head.parameters())

    wandb.init(
        project="spice-a",
        config=cfg
    )

    train_2nd_stage(
        n_epochs=cfg['n_epochs'], 
        encoder=byola_model,
        n_heads=cfg['n_heads'],
        head_creator=head_creator, 
        head_optimizer_creator=head_optimizer_creator,
        dataloader=dataloader,
        dataset=dataset,
        e_step_aug=augmentations.get_augmentation(cfg['e_step_aug']),
        m_step_aug=augmentations.get_augmentation(cfg['m_step_aug']),
        device=device,
        double_softmax=cfg['double_softmax']
    )
    head_artifact = wandb.Artifact(name='head_weights', type='weights')
    head_artifact.add_file("best_head.pth")
    wandb.log_artifact(head_artifact)

    wandb.finish()

if __name__ == "__main__":
    main()