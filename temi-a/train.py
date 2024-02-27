import argparse
import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, n_embed, n_classes, n_hidden=512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_classes)
        )
    
    def forward(self, x):
        return self.body(x)

class MultiHead(nn.Module):
    def __init__(self, n_heads, n_embed, n_classes, n_hidden=512):
        super().__init__()
        head_kwargs = {
            "n_embed": n_embed, 
            "n_classes": n_classes, 
            "hidden_dim": n_hidden
        }
        self.heads = nn.ModuleList([Head(**head_kwargs) for _ in range(n_heads)])

    def forward(self, x):
        return [head(x) for head in self.heads]

def copy_module(src, dst):
    pass

def train_epoch(students, teachers, embeds_loader):
    for batch, _ in embeds_loader:
        pass

def train(args, embeds_loader, n_embed):
    students = MultiHead(n_heads=args.n_heads, n_embed=n_embed, n_hidden=args.n_hidden) 
    teachers = MultiHead(n_heads=args.n_heads, n_embed=n_embed, n_hidden=args.n_hidden) 
    copy_module(students, teachers)

    optimizer = torch.optim.AdamW(students.parameters(), lr=1e-4, weight_decay=1e-4)

    for i in range(args.n_epochs):
        loss = train_epoch()

def main(args):
    train()

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
        '--n_clusters', required=True, type=int,
        help="Number of clusters in your data"
    )
    main(parser.parse_args())