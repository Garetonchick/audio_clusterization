import torch.nn as nn

class Head(nn.Module):
    def __init__(self, n_embed, n_classes, n_hidden=512, temp=0.1):
        super().__init__()

        self.temp = temp
        self.body = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_classes)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.body(x)
        logits /= self.temp
        return self.softmax(logits)

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