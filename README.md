# Fisher-Bures Adversary Graph Convolutional Networks

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a reference implementation of the paper [K. Sun, P. Koniusz, Z. Wang. Fisher-Bures Adversary Graph Convolutional Network. 2019](https://arxiv.org/abs/1903.04154)

## Outline

Based on information geometry, the intrinsic shape of isotropic noise corresponds to the largest eigenvectors of the graph Laplacian. Such noise can bring a small but consistent improvement to generalization. In the paper we discussed three different geometries of a graph that is embedded in a neural network, namely intrinsic geometry (how to define graph distance); extrinsic geometry (how perturbation of the graph affect the neural network); embedding geometry (in the space of all network embeddings). We imported new tools from quantum information geometry into the domain of graph neural networks.

## Performance

The following table shows the average (20 random splits of train:dev:test data; 10 different random initialisations for each split) testing loss/accuracy, based on a GCN model with one hidden layer, using a unified early stopping criterion. One may refer to [benchmark.py](scripts/benchmark.py) for the exact evaluation protocols and hyperparameter settings.

| Model | Cora | Citeseer | Pubmed | amazon_electronics_computers | amazon_electronics_photo |
| --- | --- | --- | --- | --- | --- |
| GCN |        1.103/80.21 | 1.394/69.42 | 0.836/78.33 | 2.357/37.75 | 1.998/71.03 |
| FisherGCN |  1.084/80.48 | 1.593/69.61 | 0.826/78.47 | 2.354/40.73 | 1.992/72.34 |
| GCNT |       1.076/80.96 | 1.359/70.28 | 0.793/79.02 | 2.269/68.48 | 1.938/79.4  |
| FisherGCNT | 1.045/81.21 | 1.563/70.47 | 0.782/79.12 | 2.262/70.4  | 1.928/81.12 |

Notice that the low score of GCN/FisherGCN on amazon_electronics_computer is due to failed runs (stopping too early).

## Requirements
- Python >=3.6.x
- Tensorflow >= 1.13

## Install dependencies
```
  pip install -r requirements.txt
```
to install all dependencies.

## Datasets

Our datasets are from following papers:
  1. [T. Kipf, M. Welling. Semi-Supervised Classification with Graph Convolutional Networks. 2016.](https://arxiv.org/abs/1609.02907) [Dataset](https://github.com/stellargraph/FisherGCN/blob/master/gcn/data)
  2. [O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. Pitfalls of Graph Neural Network Evaluation. 2018](https://arxiv.org/abs/1811.05868). [Dataset](https://github.com/shchur/gnn-benchmark/tree/master/data)

Or, you can download our [ZIP data](https://s3-ap-southeast-2.amazonaws.com/stellar-research/fishergcn/data.zip) that includes the two datasets above. After downloading, please run:
```
unzip data.zip
```
All datasets will be stored in a **data** folder.

## Run the code

```bash
python train.py --model fishergcn
```

## Disclaimer

The codes are subject to changes and are not necessarily synchronized with our arxiv report.

## Cite

If you find this work useful and apply it in your work, please cite our paper

```
@article{fishergcn,
  author  = {Ke Sun and Piotr Koniusz and Zhen Wang},
  title   = {Fisher-Bures Adversary Graph Convolutional Network},
  journal = {CoRR},
  volume  = {abs/1903.04154},
  year    = {2019},
}
```
