import torch
import torch.nn as nn

class PMILoss(nn.Module):
    def __init__(self, n_classes, momentum, device='cpu'):
        super().__init__()
        self.momentum = momentum
        self.global_tprobs = torch.full((n_classes,), fill_value=1/n_classes, device=device)

    def raw_pmi(self, sprobs, tprobs):
        prob_sum = (sprobs * tprobs * torch.reciprocal(self.global_tprobs)).sum(dim=1)
        return torch.log(prob_sum)

    def update_global_tprobs(self, tprobs):
        self.global_tprobs = self.momentum * self.global_tprobs + (1 - self.momentum) * tprobs.mean(dim=0)
    
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
    def __init__(self, n_heads, n_classes, momentum, device='cpu'):
        super().__init__()
        self.n_heads = n_heads
        self.pmi_losses = [PMILoss(n_classes=n_classes, momentum=momentum, device=device) for _ in range(n_heads)]

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
    def __init__(self, n_heads, n_classes, momentum, device='cpu'):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.pmi_losses = [PMILoss(n_classes=n_classes, momentum=momentum, device=device) for _ in range(n_heads)]
     
    def forward(self, sprobs, tprobs):
        """
        Calculate teacher-Weighted Pointwise Mutual Information (WPMI) Loss
        :param sprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains student probabilities of image x, second contains the same for image x'
        :param tprobs: list of pairs of tensors with shape (B, C). list index determines head, first 
        tensor in pair contains teacher probabilities of image x, second contains the same for image x'
        :return: losses for each head 
        """
        B = sprobs[0][0].shape[0] 
        tweights = torch.zeros((B,), device=self.device)

        for i in range(self.n_heads):
            tweights += (tprobs[i][0] * tprobs[i][1]).sum(dim=1)

        tweights /= self.n_heads
        losses = [] 

        for i in range(self.n_heads):
            losses.append((tweights * self.pmi_losses[i](sprobs[i], tprobs[i])).mean())

        return losses