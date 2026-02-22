import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SIMPLE_net(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2, eps=0.1):
        super().__init__()
        self.eps = eps
        self.encoder_convs = nn.ModuleList([
            GCNConv(in_features if i==0 else out_features, out_features)
            for i in range(num_layers)
        ])
        self.decoder_convs = nn.ModuleList([
            GCNConv(out_features, in_features if i==num_layers-1 else out_features)
            for i in range(num_layers)
        ])
        for i in range(num_layers):
            self.decoder_convs[i].lin.weight = nn.Parameter(
                self.encoder_convs[num_layers-1-i].lin.weight.t()
            )
        self.proj_head = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, edge_index, perturbed=False):
        # encode
        h = x
        mean_emb = 0
        for i, conv in enumerate(self.encoder_convs):
            h = conv(h, edge_index)
            if perturbed:
                random_noise = torch.rand_like(h)
                noise = F.normalize(random_noise, dim=1)
                h = h + torch.sign(h) * noise * self.eps
            mean_emb = mean_emb * i/(i+1) + h/(i+1)
        z = mean_emb
        
        # decode
        x_recon = h
        for conv in self.decoder_convs:
            x_recon = conv(x_recon, edge_index)
        return z, x_recon, self.proj_head(z)
    
    def infoNCE_loss(self,z1, z2, tau):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.T) / tau
        pos_sim = torch.diag(sim_matrix)
        loss = -pos_sim.mean() + torch.logsumexp(sim_matrix, dim=1).mean()
        return loss