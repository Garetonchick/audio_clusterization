import byol_a
import dcase5
import json
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import MLP
import torch.nn.functional as F

def cos_matrix(a, b):
    return torch.mm(F.normalize(a, dim=1), torch.transpose(F.normalize(b, dim=1), 0, 1))

@torch.no_grad()
def e_step(encoder, head, batch, aug):
    print(f"\nbatch shape: {batch.shape}\n")
    head.eval()
    aug_features = encoder(aug(batch))
    logprobs = F.log_softmax(head(aug_features), dim=1)
    B, K = logprobs.shape
    n_conf_samples = B // K
    indices = torch.topk(logprobs, n_conf_samples, dim=0, largest=True).indices
    features = encoder(batch)
    cluster_centers = torch.zeros((K, features.shape[1]), dtype=features.dtype)

    # TODO: vectorize
    for i in range(K): 
        cluster_centers[i] = torch.mean(features[indices[:, i]], dim=0)
    
    sims = cos_matrix(cluster_centers, features) # (K, B)

    indices = torch.topk(sims, n_conf_samples, dim=1, largest=False).indices.view(-1)
    # features_s = features[indices]
    proto_labels = torch.transpose(torch.arange(0, K).repeat(n_conf_samples, 1), 0, 1).flatten()

    return batch[indices], proto_labels


def m_step(encoder, head, head_optimizer, batch_s, proto_labels, aug):
    head.train()
    head_optimizer.zero_grad()
    logits = head(encoder(aug(batch_s)))
    # probs = F.softmax(logits, dim=1) 
    # log_softmax_probs = F.log_softmax(probs, dim=1)
    # loss = F.nll_loss(log_softmax_probs, proto_labels)
    loss = F.cross_entropy(logits, proto_labels)

    if torch.isnan(loss).item():
        print("\n\nLoss is nan!!!!\n\n")
        print(f"\n\nlogits={logits}\n\n")
        print(f"\n\nproto_labels={proto_labels}\n\n")

    loss.backward()
    head_optimizer.step()

    return loss.item()

def train_epoch_2nd_stage(encoder, head, head_optimizer, dataloader, e_step_aug, m_step_aug):
    epoch_loss = 0
    n_samples = 0 
    for batch, _ in tqdm(dataloader, desc="Train epoch"):
        batch_s, proto_labels = e_step(
            encoder=encoder,
            head=head,
            batch=batch,
            aug=e_step_aug
        )
        print("E step done")
        loss = m_step(
            encoder=encoder,
            head=head,
            head_optimizer=head_optimizer,
            batch_s=batch_s,
            proto_labels=proto_labels,
            aug=m_step_aug
        )
        print("M step done")
        print(f"Batch loss: {loss}\n")
        epoch_loss += loss * batch.shape[0]
        n_samples += batch.shape[0]

    return epoch_loss / n_samples

def train_2nd_stage(n_epochs, encoder, head, head_optimizer, dataloader, e_step_aug, m_step_aug):
    for epoch in tqdm(range(n_epochs), desc="Second stage training"):
        loss = train_epoch_2nd_stage(encoder, head, head_optimizer, dataloader, e_step_aug, m_step_aug)
        print(f"Epoch: {epoch}, loss: {loss}")

def test(encoder, head, dataset):
    pass

@torch.no_grad()
def predict_pseudo_labels(encoder, head, dataset):
    head.eval()
    labels = torch.Tensor(len(dataset), dtype=torch.long) 
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, (batch, _) in enumerate(dataloader):
        logits = head(encoder(batch))
        labels[i * batch_size: (i + 1)*batch_size] = torch.argmax(logits, dim=1)
    return labels

def load_config(path):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    torch.manual_seed(4444)
    # cfg = load_config('cfg.json')
    print("Start")
    dataset = dcase5.get_dataset('smol_dcase')
    print("Loaded dataset")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True) 
    byola_model = byol_a.get_frozen_pretrained_byola(dataset.calc_norm_stats())
    print("Loaded byola")
    head = MLP(in_channels=3072, hidden_channels=[6144, 10])
    head_optimizer = torch.optim.Adam(head.parameters())

    train_2nd_stage(
        n_epochs=1, 
        encoder=byola_model,
        head=head,
        head_optimizer=head_optimizer,
        dataloader=dataloader,
        e_step_aug=lambda x: x,
        m_step_aug=lambda x: x
    )

    test(
        encoder=byola_model,
        head=head,
        dataset=dataset
    )
    pseudo_labels = predict_pseudo_labels(byola_model, head, dataset)