import warnings
warnings.filterwarnings("ignore")
import SIMPLE

# the location of R (used for the mclust clustering)
import os
os.environ['R_HOME'] = "/home/lvyz/miniconda3/envs/test/lib/R"
os.environ['R_USER'] = "/home/lvyz/miniconda3/envs/test/lib/python3.8/site-packages/rpy2"

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# font settings
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# seed setting
seed=666
SIMPLE.utils.fix_randomseed(seed)

adata_all = []
# donor 1:['151507','151508','151509','151510']; donor 2:['151669','151670','151671','151672']; donor 3:['151673','151674','151675','151676']
slice_ids = ['151673','151674','151675','151676']
edge_index_list = []
offset = 0
num_nodes_per_slice = []
for slice_id in slice_ids:
    print(slice_id)
    input_dir = os.path.join('/Data/lvyz/data/', slice_id)
    adata = sc.read_visium(path=input_dir, count_file=slice_id + '_filtered_feature_bc_matrix.h5', load_images=True)
    
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(input_dir, slice_id + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['ground_truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth'].astype('category')
    
    adata.obs_names = [x+'_'+slice_id for x in adata.obs_names]
    adata.var_names_make_unique()

    # Construct the spatial neighborhood graph for each slice
    SIMPLE.Cal_Spatial_Net(adata,k_cutoff=6)
    edge_index = adata.uns['edge_index']
    edge_index = edge_index + offset
    offset += adata.n_obs
    edge_index_list.append(edge_index)

    print(adata)
    adata_all.append(adata)
    
adata_concat = ad.concat(adata_all, label="batch", keys=slice_ids)
adata_concat.obs['ground_truth'] = adata_concat.obs['ground_truth'].astype('category')
adata_concat.obs["batch"] = adata_concat.obs["batch"].astype('category')

# Concat the spatial neighborhood graph for multiple slices
edge_index_concat = np.hstack(edge_index_list)
adata_concat.uns['edge_index'] = edge_index_concat

# preprocess
adata_concat = SIMPLE.preprocess(adata_concat, gene_num=5000)
print(adata_concat)

adata_concat=SIMPLE.train_SIMPLE(adata_concat,device=used_device,random_seed=seed)

SIMPLE.utils.mclust_R(adata_concat, num_cluster=7,random_seed=seed,refinement=True)
adata_concat = adata_concat[adata_concat.obs['ground_truth']!='unknown']
nmi_score = nmi(adata_concat.obs['ground_truth'], adata_concat.obs['mclust'])
print("Normalized Mutual Information (NMI):", nmi_score)


sc.pp.neighbors(adata_concat, use_rep='SIMPLE', random_state=seed)
sc.tl.umap(adata_concat, random_state=seed)

color = ['#ffbb78', '#98df8a', '#ff9896','#17becf']
color_dict = dict(zip(slice_ids, color))
adata_concat.uns['batch_colors'] = [color_dict[x] for x in adata_concat.obs.batch.cat.categories]
axs=sc.pl.umap(adata_concat, color=['batch', 'mclust'], ncols=2,wspace=0.3, show=False)
for ax in axs:
    ax.set_title(ax.get_title(),fontsize=20)
plt.savefig('/home/lvyz/SIMPLE-main/dlpfc3_umap.pdf', bbox_inches='tight')
plt.close()

adata_list = []
NMI_list=[]
for i,id in enumerate(slice_ids):
    adata_list.append(adata_concat[adata_concat.obs['batch'] == id])
    NMI_list.append(round(nmi(adata_list[i].obs['ground_truth'], adata_list[i].obs['mclust']),2))

spot_size = 200
title_size = 16
fig, ax = plt.subplots(1, len(slice_ids), figsize=(10, 5), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
for i in range(len(slice_ids)):
    if i ==len(slice_ids)-1:
        _sc=sc.pl.spatial(adata_list[i], img_key=None, color=['mclust'], title=[''],legend_fontsize=14, show=False, ax=ax[i], frameon=False,spot_size=spot_size)
    else:
        _sc=sc.pl.spatial(adata_list[i], img_key=None, color=['mclust'], title=[''],legend_loc=None, legend_fontsize=14, show=False, ax=ax[i], frameon=False,spot_size=spot_size)
    _sc[0].set_title("NMI=" + str(NMI_list[i]), size=title_size)
plt.savefig('/home/lvyz/SIMPLE-main/dlpfc3_spatial.pdf', bbox_inches='tight')