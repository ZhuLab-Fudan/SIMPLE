# SIMPLE

A multi-slice spatial omics integration method for depicting tissue complexity via joint intra- and inter-slice learning

## Overview

![Overview of SIMPLE](./SIMPLE.png)

**a.** The input data contains gene expression matrices and spatial coordinates of multiple spatial transcriptomic datasets, and then the spatial neighborhood graph was constructed based on the coordinates. **b.** SIMPLE takes preprocessed gene expression matrix and spatial neighborhood graph as input and learns consistent spot embeddings from two perspectives: intra-slice and inter-slice. For intra-slice learning, SIMPLE uses a LightGCN-inspired network and performs augmentation directly at the representation level by adding uniform random noise to the original embeddings, which enables SIMPLE to adapt to different types of spatial transcriptomics datasets. For inter-slice learning, SIMPLE adopts both global (Optimal Transport) and local (MNN triplet pairs) strategies to remove batch effect and capture consistent biological information across different slices.

## Requirements

We recommend that users install the following packages to run SIMPLE.

- anndata==0.9.2
- annoy==1.17.3
- gseapy==1.1.10
- hnswlib==0.8.0
- matplotlib==3.7.5
- numba==0.55.2
- numpy==1.22.4
- pandas==2.0.3
- POT==0.9.5
- rpy2==3.4.5
- scanpy==1.9.8
- scikit-learn==1.3.2
- scipy==1.10.1
- seaborn==0.13.2
- squidpy==1.2.3
- tqdm==4.67.1
- umap-learn==0.5.7
- louvain==0.8.2
- networkx == 3.1
- more-itertools==10.5.0
- torch==2.3.1
- torch_cluster==1.6.3
- torch_geometric==2.5.3
- torch_scatter==2.1.2
- torch_sparse==0.6.18





You can install SIMPLE with **anaconda** using the following commands:

```shell
conda create -n SIMPLE_env python=3.8
conda activate SIMPLE_env
pip install -r requirements.txt
```

## Tutorials

Please find examples of SIMPLE applications in the tutorials folder, where jupyter notebooks are provided.
- dlpfc_tutorial: Integration of SIMPLE on DLPFC dataset
- merfish_tutorial: Integration of SIMPLE on MERFISH mouse hypothalamus dataset
- starmap_tutorial: Integration of SIMPLE on STARmap mPFC dataset
- protein_tutorial: Integration of SIMPLE on spatial proteomics datasets

**NOTE:** Please update the data paths before running the code.

## Datasets

The example spatial transcriptomics datasets can be downloaded with the links below.

The human DLPFC dataset is available at the spatialLIBD package (http://spatial.libd.org/spatialLIBD).

The mouse hypothalamus dataset by MERFISH and the mouse medial prefrontal cortex dataset by STARmap are available at https://github.com/zhengli09/BASS-Analysis/tree/master/data. 

The spatial proteomics datasets of mouse spleen and mouse thymus are available at https://zenodo.org/records/10362607. 

## Contacts and bug reports

Please feel free to send bug reports or questions to Yanzhe Lv: 23110850008@m.fudan.edu.cn, Prof. Shihua Zhang: zsh@amss.ac.cn and Prof. Shanfeng Zhu: zhusf@fudan.edu.cn
