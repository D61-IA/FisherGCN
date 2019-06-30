# Fisher-Bures Adversary Graph Convolutional Networks

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a reference implementation of the paper [Fisher-Bures Adversary Graph Convolutional Networks](https://arxiv.org/abs/1903.04154)

## Outline

Based on information theory, the intrinsic shape of isotropic noise corresponds to the largest eigenvectors of the graph Laplacian. Such noise can bring a small but consistent improvement to generalization. In this paper, we discussed three different geometries of a graph that is embedded in a neural network, namely intrinsic geometry (how to define graph distance); extrinsic geometry (how perturbation of the graph affect the neural network); embedding geometry (how to measure graph embeddings). We imported new analytical tools from quantum information geometry into the domain of graph neural networks.

## Performance

The following table shows the average (20 random splits of train:dev:test data; 10 different random initialisations per split) testing loss/accuracy, based on a GCN model with one hidden layer, using a unified early stopping criterion. One can repeat these results based on [script](hpc/submit_grid.sh) (one has to translate the codes into actual commands without HPC resources). Notice that the scores have a large variation based on the how the train:dev:test datasets is selected (we use the same ratio with the Planetoid split) and one has to be careful when comparing different networks. It is highly recommendeded to run the codes on a GPU.

| Model | Cora | Citeseer | Pubmed |
| --- | --- | --- | --- | --- | --- |
| GCN |        1.07/80.52 | 1.36/69.59 | 0.75/78.17 |
| FisherGCN |  1.06/80.70 | 1.35/69.80 | 0.74/78.43 |
| GCNT |       1.04/81.20 | 1.33/70.31 | 0.70/78.99 |
| FisherGCNT | 1.03/81.46 | 1.32/70.48 | 0.69/79.34 |

The learning curve on Cora looks like this ![cora](lcurvescora.pdf)

## Requirements
- Python >=3.6.x
- Tensorflow >= 1.13

## Install dependencies
```
  pip install -r requirements.txt
```
to install all dependencies.

## Datasets

We use the same datasets as in [1][2][3]. They are stored in the folder [data](data/). Please [install](https://github.com/git-lfs/git-lfs/wiki/Installation) `git-lfs` before closing the repository with the following commands

```bash
# ...install git-lfs...
git lfs install
git clone https://github.com/stellargraph/FisherGCN
```

## Run the code

```bash
python train.py --model fishergcn
```
Check [train.py](gcn/train.py) on the parameter configurations.

## References

The following works are highlighted here because our codes and datasets are largely based on them. See our [paper](https://arxiv.org/abs/1903.04154) for the complete list of references.

[1] Z. Yang, W. W. Cohen, R. Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](http://proceedings.mlr.press/v48/yanga16.html), ICML, 2016.

[2] T. Kipf, M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR, 2017.

[3] O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann, [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868), Relational Representation Learning Workshop, NIPS 2018.

## Cite

If you apply FisherGCN in your work, please cite

```
@inproceedings{fishergcn,
  author    = {Ke Sun and Piotr Koniusz and Zhen Wang},
  title     = {Fisher-Bures Adversary Graph Convolutional Networks},
  booktitle = {Uncertainty in Artificial Intelligence},
  year      = {2019},
  pages     = {(to appear)},
  note      = {arXiv:1903.04154 [cs.LG]},
  url       = {https://arxiv.org/abs/1903.04154},
}
```
