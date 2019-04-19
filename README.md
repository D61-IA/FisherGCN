# Fisher-Bures Adversary Graph Convolutional Network (FisherGCN)

This is a reference implementation of the paper "K. Sun, P. Koniusz, Z. Wang. Fisher-Bures Adversary Graph Convolutional Network. 2019" https://arxiv.org/abs/1903.04154

## Outline

Based on information geometry, the intrinsic shape of the isometric noise corresponds to the largest eigenvectors of the graph Laplacian. Such noise can bring a small but consistent improvement to generalization. In the paper we discussed three different geometries of a graph that is embedded in a neural network, namely intrinsic geometry (how to define graph distance); extrinsic geometry (how perturbation of the graph affect the neural network); embedding geometry (in the space of all network embeddings). We imported new tools from quantum information geometry into the domain of graph neural networks.

## Performance

The folowing table shows the average (after 20 runs) testing accuracy and testing loss on the cacnonical split the datasets, based on a GCN model with one hidden layer of size 64, learning rate 0.01, dropout rate 0.5, regularization strength 0.0005, and unified early stopping criterion.

| Model | Cora | Citeseer | Pubmed |
| --- | --- | --- | --- |
| GCN | 81.27/1.103 | 71.16/1.385 | 79.04/0.747 | 
| FisherGCN | 81.80/1.085 | 71.25/1.372 | 79.08/0.738 |
| GCNT | 82.18/1.081 | 71.80/1.347 | 79.25/0.714 |
| FisherGCNT | 82.33/1.061 | 71.77/1.333 | 79.36/0.703 |

## Requirements
- Python >=3.6.x
- Tensorflow >= 1.2

## Run the code

```bash
python train.py --model fishergcn
```
No installation required.

## Data

We use the same datasets in the GCN paper. You can find those data [here](FisherGCN/data)

## Disclamer

The codes are still subject to subsential change and are not necessarily synchronized with our arxiv report.

## Cite

@article{fishergcn,
&nbsp;&nbsp;author  = {Ke Sun and Piotr Koniusz and Zhen Wang},
&nbsp;&nbsp;title   = {Fisher-Bures Adversary Graph Convolutional Network},
&nbsp;&nbsp;journal = {CoRR},
&nbsp;&nbsp;volume  = {abs/1903.04154},
&nbsp;&nbsp;year    = {2019},
}

