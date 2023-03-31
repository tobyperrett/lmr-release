"""
This file presents a minimal example of LMR on some dummy data on the CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.random.manual_seed(1)

class LMR(torch.nn.Module):
    def __init__(self, class_counts, l, d, omega, epsilon):
        super().__init__()

        # number of samples in each class in the training set
        self.class_counts = class_counts
        self.num_classes = class_counts.shape[0]

        # reconstruction contribution of the class with the lowest count
        self.l = l

        # decay on class count contribution
        self.d = d

        # few-shot threshold for excluding samples from reconstructions
        self.omega = omega

        # used to ensure log produces a positive output
        self.epsilon = epsilon

        # pre-compute c(y)
        self.set_class_weights()

        # set mask which identifies classes which are few-shot
        self.set_fs_classes()


    # pre-compute c(y)
    def set_class_weights(self):

        # Eq. 1 in paper.
        tilde_C = 1.0 / torch.log((self.class_counts * self.d) + self.epsilon)

        # Eq. 2 in paper.
        numerator = tilde_C - torch.min(tilde_C)
        denominator = torch.max(tilde_C) - torch.min(tilde_C)
        self.c = numerator / denominator * self.l


    # identify classes which are few-shot. 1 for few-shot, 0 otherwise    
    def set_fs_classes(self):
        fs_classes = torch.where(self.class_counts <= self.omega, 1, 0)
        self.fs_classes = nn.Parameter(fs_classes, requires_grad=False)


    # compute reconstructions for each sample and combine with original sample
    # based on the class size contribution
    def reconstruct(self, x, y):
        n_batch, dim = x.shape

        # calculate similarities
        x_norm = F.layer_norm(x, [dim]) / torch.sqrt(torch.tensor(dim))
        sim = torch.matmul(x_norm, x_norm.t())

        # mask to remove similarity to self
        self_mask = torch.eye(n_batch)

        # mask to remove similarity to few-shot classes
        fs_mask = torch.index_select(input=self.fs_classes, dim=0, index=y)

        # Eq. 3 in paper.
        # combine masks to create exclusion mask E. 
        # entries are either 1 (ignore) or 0 (don't ignore)
        E = self_mask + fs_mask
        E = torch.where(E >= 1.0, 1.0, 0.0)

        # Eq. 4 in paper.
        # apply mask and softmax to calculate W. 
        sim = sim - 1e5 * E
        sim = F.softmax(sim, dim=-1)

        # get contribution of reconstruction for each sample, based on class count
        contrib = torch.index_select(input=self.c, dim=0, index=y)

        # Eq. 5 in paper.
        # combine reconstructions with original samples, to get R. 
        reconstructions = torch.matmul(sim, x)
        R = (reconstructions.t() * contrib).t() + (x.t() * (1 - contrib)).t()
        return R


    # perform pairwise label mixing
    def pairwise_mix(self, x, y):

        n_batch = x.shape[0]

        # generate one-hot labels ready for mixing
        y_oh = F.one_hot(y, self.num_classes)

        # beta selects other samples to mix with
        beta = torch.randint(low=0, high=n_batch, size=[n_batch])
        beta = F.one_hot(beta, n_batch)

        # alpha are the pairwise mixing weights
        # set half the elements of alpha to 1, otherwise random
        alpha = torch.where(torch.rand(n_batch) > 0.5, torch.rand(n_batch), torch.ones(n_batch))

        # Eq. 6 in paper.
        # Mixing mask M. 
        M = (torch.eye(n_batch) * alpha + beta.t() * (1 - alpha)).t()

        # Eq. 7 in main paper.
        x = torch.matmul(M, x)
        y = torch.matmul(M, y_oh.float())
        return x, y


    # reconstruction and pairwise mixing are only done during training
    # at inference, nothing is modified
    def forward(self, x, y):
        if self.training:
            x = self.reconstruct(x, y)
            x, y = self.pairwise_mix(x, y)

        return x, y


if __name__ == "__main__":
    n_batch = 4
    dim = 8
    lmr = LMR(class_counts=torch.tensor([1, 10, 100, 1000]), l=0.5, d=1.0, omega=20.0, epsilon=0.1)

    x = torch.rand([n_batch, dim])
    y = torch.tensor([0, 3, 2, 3])

    print("inputs:")
    print("x (features): {}".format(x))
    print("y (labels): {}".format(y))

    x, y = lmr(x, y)
    
    print("outputs:")
    print("x (features): {}".format(x))
    print("y (labels): {}".format(y))
