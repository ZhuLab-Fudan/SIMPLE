import os
import random
import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from annoy import AnnoyIndex
import itertools
import hnswlib
import scanpy as sc
import ot
import torch
from torch.backends import cudnn

def fix_randomseed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 



def preprocess(adata,gene_num=5000):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=gene_num)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']].copy()
    return adata



def Cal_Spatial_Net(adata, k_cutoff=None, max_neighbor_num=50):
    """\
    Construct spatial KNN graph and save edge_index in adata.uns.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    k_cutoff
        The number of KNN nearest neighbors
    max_neighbor_num
        Maximum neighbors searched before truncation

    Returns
    -------
    The spatial neighborhood graph is saved in adata.uns['edge_index']
    """

    print('------------Calculating spatial neighborhood graph------------')
    coor = pd.DataFrame(adata.obsm['spatial'])
    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neighbor_num + 1, algorithm='ball_tree').fit(coor)
    _,indices = nbrs.kneighbors(coor)
    
    indices = indices[:, 1:k_cutoff + 1]
    n_cells, n_neighbors = indices.shape
    cell1 = np.repeat(np.arange(n_cells), n_neighbors)
    cell2 = indices.flatten()

    edges = np.vstack((cell1, cell2))
    adata.uns['edge_index'] = edges


def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata 



def mclust_R(adata, num_cluster, used_obsm='SIMPLE', random_seed=666,knn=50,refinement=False):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, 'EEE')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    if refinement:
        new_type=refine_label_multibatch(adata, knn)
        adata.obs['mclust'] = new_type 
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def refine_label_multibatch(adata, knn=50, key='mclust', batch_key='batch'):
    """
    Multi-slice label refinement function.

    - Process each slice (batch) independently
    """
    new_type = np.empty(adata.shape[0], dtype=object)
    old_type = adata.obs[key].values
    for batch_name in adata.obs[batch_key].unique():
        batch_mask = adata.obs[batch_key] == batch_name
        batch_indices = np.where(batch_mask)[0]
        batch_position = adata.obsm['spatial'][batch_indices]
        batch_old_type = old_type[batch_indices]
        n_cell = len(batch_indices)

        batch_distance = ot.dist(batch_position, batch_position, metric='euclidean')
        batch_new_type = []
        for i in range(n_cell):
            vec = batch_distance[i, :]
            index = vec.argsort()
            neigh_types = [batch_old_type[index[j]] for j in range(1, knn+1)]
            max_type = max(neigh_types, key=neigh_types.count)
            batch_new_type.append(max_type)
        new_type[batch_indices] = batch_new_type

    return new_type.astype(float).astype(int).tolist()


# Modified from https://github.com/lkmklsmn/insct
def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names
    return(mnns)

def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual



def prepare_triplet(adata, z, positive_dict, exclude_ratio=0.1, tripletNum_per_anchor=5):
    """
        Prepare triplet samples (anchor, positive, negative) for triplet loss training.
        This function constructs triplets by treating cells with known correspondences
        (e.g., mutual nearest neighbors across batches) as anchor-positive pairs, and
        selecting negative samples from the same batch as the anchor. A proportion of
        nearest neighbors can be excluded to avoid trivial negatives.

        Parameters
        ----------
        adata : AnnData
            data matrix containing cell metadata, including batch labels.
        z : torch.Tensor
            Cell embedding matrix of shape [n_cells, embedding_dim].
        positive_dict : dict
            Dictionary mapping batch pairs to anchor-positive cell name pairs.
        exclude_ratio : float, optional (default=0.1)
            Proportion of nearest neighbors to exclude from negative candidate pool.
        tripletNum_per_anchor : int, optional (default=5)
            Number of triplets to generate per anchor.

        Returns
        -------
        anchor_ind : list of int
            Global indices of anchor cells.
        positive_ind : list of int
            Global indices of positive cells.
        negative_ind : list of int
            Global indices of negative cells.
        triplet_weights : list of float or None
            Optional weights corresponding to each (anchor, positive) pair.
    """
    anchor_ind = []
    positive_ind = []
    negative_ind = []
    name_globalID_dict = dict(zip(adata.obs_names, range(adata.shape[0])))
    
    for batch_pair in positive_dict.keys():
        batch_anchors = list(positive_dict[batch_pair].keys())
        batch_names = adata.obs['batch'][batch_anchors]
        current_batch = batch_names.iloc[0]

        # Get cells and indices from the same batch
        same_batch_mask = adata.obs['batch'] == current_batch
        same_batch_cells = adata.obs_names[same_batch_mask].values
        same_batch_indices = [name_globalID_dict[cell] for cell in same_batch_cells]
        
        # Extract embeddings and compute pairwise distances
        anchor_indices = [name_globalID_dict[anchor] for anchor in batch_anchors]
        anchor_embeddings = z[anchor_indices]
        batch_embeddings = z[same_batch_indices]
        dist_matrix = torch.cdist(anchor_embeddings, batch_embeddings)  # [num_anchors, num_batch_samples]
        
        # Select negative candidate pool (exclude nearest neighbors)
        n_samples = dist_matrix.shape[1]
        k = int(n_samples * exclude_ratio)
        sorted_indices = torch.argsort(dist_matrix, dim=1)
        valid_neg_indices = sorted_indices[:, k:]
        
        for anchor_idx, anchor in enumerate(batch_anchors):
            pos_candidates = positive_dict[batch_pair][anchor]
            if not pos_candidates:
                continue
            anchor_valid_neg = valid_neg_indices[anchor_idx]
            if len(anchor_valid_neg) == 0:
                continue
                
            # Generate triplets
            for i in range(tripletNum_per_anchor):
                if i == 0:
                    pos = pos_candidates[0]
                else:
                    pos = random.choice(pos_candidates) 
                neg_col = anchor_valid_neg[torch.randint(len(anchor_valid_neg), (1,))]
                a_idx = name_globalID_dict[anchor]
                p_idx = name_globalID_dict[pos]
                n_cell = same_batch_cells[neg_col.item()]
                n_idx = name_globalID_dict[n_cell]
                anchor_ind.append(a_idx)
                positive_ind.append(p_idx)
                negative_ind.append(n_idx)

    return anchor_ind, positive_ind, negative_ind


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        [C, D] matrix
    p
        p-norm

    Return
    ------
    [R, C] matrix
        distance matrix
    """
    distance = torch.cdist(pts_src, pts_dst, p=p) ** p
    return distance

def unbalanced_ot_nograd(tran, mu1, mu2, device, Couple, reg=0.1, reg_m=1.0, iter_range=50):
    '''
    Calculate an unbalanced optimal transport matrix between batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix.
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device
    iter_range
        number of iterations for the unbalanced OT optimization. Default: 10

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''
    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    
    with torch.no_grad():
        if Couple is not None:
            Couple = torch.tensor(Couple, dtype=torch.float).to(device)
        p_s = torch.ones(ns, 1) / ns
        p_t = torch.ones(nt, 1) / nt
        p_s = p_s.to(device)
        p_t = p_t.to(device)

        if tran is None:
            tran = torch.ones(ns, nt) / (ns * nt)
            tran = tran.to(device)

        dual = (torch.ones(ns, 1) / ns).to(device)
        f = reg_m / (reg_m + reg)

        for m in range(iter_range):
            if Couple is not None:
                cost = cost_pp*Couple
            else:
                cost = cost_pp

            kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
            b = p_t / (torch.t(kernel) @ dual)

            for i in range(iter_range):
                dual =( p_s / (kernel @ b) )**f
                b = ( p_t / (torch.t(kernel) @ dual) )**f
            tran = (dual @ torch.t(b)) * kernel
        if torch.isnan(tran).sum() > 0:
            tran = (torch.ones(ns, nt) / (ns * nt)).to(device)
    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()
