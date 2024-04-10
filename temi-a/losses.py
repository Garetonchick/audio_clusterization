import torch
import torch.nn as nn

from stolen.losses import TEMI as StolenTEMILoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PMILoss(nn.Module):
    def __init__(self, n_classes, momentum, beta, device='cpu'):
        super().__init__()
        self.momentum = momentum
        self.beta = beta
        self.updates = 0
        self.register_buffer("global_tprobs", torch.ones(1, n_classes, device=DEVICE)/n_classes)

    def raw_pmi(self, sprobs, tprobs):
        prob_sum = (torch.pow(sprobs * tprobs, self.beta) / self.global_tprobs).sum(dim=1)
        return torch.log(prob_sum)

    @torch.no_grad()
    def update_global_tprobs(self, tprobs):
        # self.global_tprobs = torch.tensor([0.263588, 0.246937, 0.244109, 0.076029, 0.073516, 0.034244, 0.027647,0.021363, 0.012567], dtype=torch.float)
        self.global_tprobs = self.momentum * self.global_tprobs + (1 - self.momentum) * tprobs.mean(dim=0)
        self.global_tprobs /= self.global_tprobs.norm()
        # print(self.global_tprobs)
    
    def forward(self, sprobs, tprobs):
        """
        Calculate Pointwise Mutual Information (PMI) Loss
        :param sprobs: tuple of two tensors with student probabilities for x and x', shape=(B, C)
        :param tprobs: tuple of two tensors with teacher probabilities for x and x', shape=(B, C)
        :return: tensor of shape (B) - losses for each pair of images in batch
        """
        losses = -(self.raw_pmi(sprobs[0], tprobs[1]) + self.raw_pmi(sprobs[1], tprobs[0])) / 2
        self.update_global_tprobs(tprobs[0])

        return losses

class MultiHeadWPMILoss(nn.Module):
    def __init__(self, n_heads, n_classes, momentum, beta, device='cpu'):
        super().__init__()
        self.n_heads = n_heads
        pmiloss_kwargs = {
            'n_classes': n_classes,
            'momentum': momentum,
            'beta': beta,
            'device': device
        }
        self.pmi_losses = [PMILoss(**pmiloss_kwargs) for _ in range(n_heads)]

    def forward(self, sprobs, tprobs):
        """
        Calculate teacher-Weighted Pointwise Mutual Information (WPMI) Loss
        :param sprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains student probabilities of image x, second contains the same for image x'
        :param tprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains teacher probabilities of image x, second contains the same for image x'
        :return: losses for each head 
        """
        losses = []
        for i in range(self.n_heads):
            tweights = (tprobs[i][0] * tprobs[i][1]).sum(dim=1)
            losses.append((tweights * self.pmi_losses[i](sprobs[i], tprobs[i])).mean())

        return losses
            

class MultiHeadTEMILoss(nn.Module):
    def __init__(self, n_heads, n_classes, momentum, beta, device='cpu'):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        pmiloss_kwargs = {
            'n_classes': n_classes,
            'momentum': momentum,
            'beta': beta,
            'device': device
        }
        self.pmi_losses = [PMILoss(**pmiloss_kwargs) for _ in range(n_heads)]
     
    def forward(self, sprobs, tprobs):
        """
        Calculate teacher-Weighted Pointwise Mutual Information (TEMI) Loss
        :param sprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains student probabilities of image x, second contains the same for image x'
        :param tprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains teacher probabilities of image x, second contains the same for image x'
        :return: losses for each head 
        """
        B = sprobs[0][0].shape[0] 
        tweights = [(tprobs[i][0] * tprobs[i][1]).sum(dim=1) for i in range(self.n_heads)]
        tweights = torch.cat(tweights).mean(dim=0)

        tweights /= self.n_heads
        losses = [] 

        for i in range(self.n_heads):
            losses.append((tweights * self.pmi_losses[i](sprobs[i], tprobs[i])).mean())

        return losses

class StolenTEMILossAdapter(nn.Module):
    def __init__(self, n_epochs, n_heads, n_classes, batch_size, beta):
        super().__init__()
        dino_loss_args = dict(
            out_dim=n_classes,
            batchsize=batch_size,
            warmup_teacher_temp=0.1,
            teacher_temp=0.1,
            warmup_teacher_temp_epochs=20,
            nepochs=n_epochs,
            num_heads=n_heads,
            beta=beta)
        self.loss_func = StolenTEMILoss(**dino_loss_args)

     
    def forward(self, slogits, tlogits, epoch):
        """
        Calculate teacher-Weighted Pointwise Mutual Information (TEMI) Loss
        :param slogits: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains student logits of image x, second contains the same for image x'
        :param tlogits: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains teacher logits of image x, second contains the same for image x'
        :return: losses for each head 
        """
        slogits = [torch.cat(lg) for lg in slogits]
        tlogits = [torch.cat(lg) for lg in tlogits]
        return self.loss_func(slogits, tlogits, epoch)
