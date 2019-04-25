# Fisher-Bures Adversary Graph Convolutional Network

This is a reference implementation of the paper [K. Sun, P. Koniusz, Z. Wang. Fisher-Bures Adversary Graph Convolutional Network. 2019](https://arxiv.org/abs/1903.04154)

## Outline

Based on information geometry, the intrinsic shape of the isotropic noise corresponds to the largest eigenvectors of the graph Laplacian. Such noise can bring a small but consistent improvement to generalization. In the paper we discussed three different geometries of a graph that is embedded in a neural network, namely intrinsic geometry (how to define graph distance); extrinsic geometry (how perturbation of the graph affect the neural network); embedding geometry (in the space of all network embeddings). We imported new tools from quantum information geometry into the domain of graph neural networks.

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

Run
```
  pip install -r gcn/requirements.txt
```
to install all dependencies.

## Run the code

```bash
python train.py --model fishergcn
```

## Data

Our datasets are from the paper [T. Kipf, M. Welling. Semi-Supervised Classification with Graph Convolutional Networks. 2016.](https://arxiv.org/abs/1609.02907) and [O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. Pitfalls of Graph Neural Network Evaluation. 2018](https://arxiv.org/abs/1811.05868). You can find those data [here](FisherGCN/gcn/data).

## Disclaimer

The codes are subject to changes and are not necessarily synchronized with our arxiv report.

## Cite

If you find this work useful, please cite our paper

```
@article{fishergcn,
  author  = {Ke Sun and Piotr Koniusz and Zhen Wang},
  title   = {Fisher-Bures Adversary Graph Convolutional Network},
  journal = {CoRR},
  volume  = {abs/1903.04154},
  year    = {2019},
}
```
