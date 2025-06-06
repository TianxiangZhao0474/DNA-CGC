import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args

class my_model(nn.Module):
    def __init__(self, dims):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])

        self.beta = nn.Parameter(torch.Tensor(1, ))
        self.alpha = nn.Parameter(torch.Tensor(1, ))
        if args.dataset in ['citeseer']:
            self.beta.data = torch.sigmoid(torch.tensor(0.99999)).to(args.device)
            self.alpha.data = torch.sigmoid(torch.tensor(0.5)).to(args.device)
        else:
            self.beta.data = torch.tensor(0.99999).to(args.device)
            self.alpha.data = torch.tensor(0.5).to(args.device)

    def forward(self, x_lmh, is_train=True):
        x_lmh = x_lmh.float()
        out1 = self.layers1(x_lmh)
        out2 = self.layers2(x_lmh)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)
        if is_train:
            out3 = out2 + torch.normal(0, torch.ones_like(out2) * args.sigma).cuda()
        else:
            out3 = out2

        return out1, out2, out3

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2, z2_s):

        sim_z1_z2 = F.cosine_similarity(z1, z2)
        sim_z1_z2_s = F.cosine_similarity(z1, z2_s)
        sim_z2_z2_s = F.cosine_similarity(z2, z2_s)

        loss = -torch.mean(torch.log(torch.exp(sim_z1_z2 / self.temperature) /
                                     (torch.exp(sim_z1_z2_s / self.temperature) + torch.exp(sim_z2_z2_s / self.temperature) + torch.exp(sim_z1_z2 / self.temperature))))

        return loss
