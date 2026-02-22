import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

from .network import SIMPLE_net
import itertools
from .utils import fix_randomseed,unbalanced_ot_nograd, distance_matrix,create_dictionary_mnn,prepare_triplet

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

def train_SIMPLE(adata, pre_epochs=500, n_epochs=1000,
        gradient_clipping=5., weight_decay=0.0001, lr=0.005, hidden_dims=30,
        tripletNum_per_anchor=5,eps=0.1, tau=0.5, alpha=1, beta=0.1, beta2=1, beta3=1,
        ot_reg=0.1, ot_reg_m=1.0,iter_comb=None, knn_neigh=50, exclude_ratio=0.01, margin=1.0,
        random_seed=666,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
    fix_randomseed(random_seed)

    x = torch.FloatTensor(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    x = x.to(device)

    # edge_index
    edge_index = adata.uns['edge_index']
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    elif isinstance(edge_index, (list, tuple)):
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
    edge_index = edge_index.to(device)

    # model
    model = SIMPLE_net(x.shape[1], hidden_dims, eps=eps).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()),lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    print("Pretraining...")
    for epoch in tqdm(range(pre_epochs)):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            _, x_recon, _ = model(x, edge_index, perturbed=False)
            _, _, z1 = model(x, edge_index, perturbed=True)
            _, _, z2 = model(x, edge_index, perturbed=True)
            loss_recon = F.mse_loss(x_recon, x)
            loss_cl = model.infoNCE_loss(z1, z2, tau)
            loss = alpha*loss_recon + beta*loss_cl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        del x_recon,z1,z2,loss_recon,loss_cl,loss

    with torch.no_grad():
        z, _, _ = model(x, edge_index, perturbed=False)
    adata.obsm['simgcl'] = z.cpu().detach().numpy()

    # Prepare for OT
    batch_list = adata.obs['batch']
    batch_names = batch_list.unique().tolist()
    batch_info = batch_list.values
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(batch_names)), 2))
    tran_dict = {comb: None for comb in iter_comb}
    
    with torch.no_grad():
        for comb in iter_comb:
            i,j = comb
            mask_i = (batch_info == batch_names[i])
            mask_j = (batch_info == batch_names[j])
            z_i = F.normalize(z[torch.tensor(mask_i, dtype=torch.bool)])
            z_j = F.normalize(z[torch.tensor(mask_j, dtype=torch.bool)])
            _, tran = unbalanced_ot_nograd(tran=tran_dict[comb], mu1=z_i, mu2=z_j, device=device, Couple=None,reg=ot_reg, reg_m=ot_reg_m)
            tran_dict[comb] = tran

    print("Training SIMPLE...")
    for epoch in tqdm(range(pre_epochs, n_epochs)):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            z, x_recon, _ = model(x, edge_index, perturbed=False)
            _,_,z1 = model(x, edge_index, perturbed=True)
            _,_,z2 = model(x, edge_index, perturbed=True)
            loss_recon = F.mse_loss(x_recon, x)
            loss_cl = model.infoNCE_loss(z1, z2, tau)

            # OT loss
            ot_loss_total = 0.0
            for comb in iter_comb:
                i,j = comb
                mask_i = (batch_info == batch_names[i])
                mask_j = (batch_info == batch_names[j])
                zi = F.normalize(z[torch.tensor(mask_i, dtype=torch.bool)],dim=1)
                zj = F.normalize(z[torch.tensor(mask_j, dtype=torch.bool)],dim=1)
                cost = distance_matrix(zi, zj)
                tran_iter= tran_dict[comb]
                ot_loss_total += (cost * tran_iter).sum()
            del cost,zi,zj

            # Triplet loss
            if epoch % 100 == 0 or epoch == pre_epochs:
                adata.obsm['simgcl'] = z.cpu().detach().numpy()
                mnn_dict = create_dictionary_mnn(adata, use_rep='simgcl', batch_name='batch', k=knn_neigh, iter_comb=iter_comb,verbose=0)
                anchor_ind = []
                positive_ind = []
                negative_ind = []
                anchor_ind, positive_ind, negative_ind = prepare_triplet(adata, z, mnn_dict, exclude_ratio=exclude_ratio, tripletNum_per_anchor=tripletNum_per_anchor)
                del mnn_dict
            anchor_arr = z[anchor_ind]
            positive_arr = z[positive_ind]
            negative_arr = z[negative_ind]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            total_loss = alpha*loss_recon + beta*loss_cl + beta2*(ot_loss_total/len(iter_comb)) + beta3*tri_output
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        scaler.step(optimizer)
        scaler.update()
        del z1,z2, x_recon,loss_recon, loss_cl, ot_loss_total, total_loss, tri_output, anchor_arr, positive_arr, negative_arr

    model.eval()
    with torch.no_grad():
        z, _, _ = model(x, edge_index, perturbed=False)
    adata.obsm['SIMPLE'] = z.cpu().numpy()
    return adata