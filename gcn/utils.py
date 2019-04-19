import numpy as np
import pickle as pkl
import random
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn import model_selection

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask( idx, l ):
    """Create boolean mask."""

    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_karate():
    """Load karate club dataset"""
    print('Loading karate club dataset...')

    G = nx.karate_club_graph()
    edges = np.array(G.edges())
    nodes = list(G.nodes())
    features = sp.eye(np.max(edges+1), dtype=np.float32).tolil()

    # True labels of the group each student (node) unded up in. Found via the original paper
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    adj = nx.adjacency_matrix(G)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    # Split nodes into train/test using stratification.
    train_nodes, test_nodes, train_targets, test_targets = model_selection.train_test_split(
        nodes, labels, train_size=int(len(nodes)/4), test_size=None, stratify=labels, random_state=55232
    )

    # Split test set into test and validation
    val_nodes, test_nodes, val_targets, test_targets = model_selection.train_test_split(
        test_nodes, test_targets, train_size=int(len(test_nodes)/2), test_size=None, random_state=523214
    )

    train_mask = sample_mask(train_nodes, labels.shape[0])
    val_mask = sample_mask(val_nodes, labels.shape[0])
    test_mask = sample_mask(test_nodes, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    y_train = y_train.reshape( (labels.shape[0],1) )
    y_val = y_val.reshape( (labels.shape[0],1) )
    y_test = y_test.reshape( (labels.shape[0],1) )

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data( dataset_str, data_seed ):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
           data_seed:  random seed to split the train/test/dev datasets
                       if None then use the original split
    :return: All data input files loaded (as well the training/test data).
    """

    if dataset_str == 'karate':
        return load_karate()

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file( "data/ind.{}.test.index".format(dataset_str) )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    _nxgraph = nx.from_dict_of_lists( graph )
    adj = nx.adjacency_matrix( _nxgraph )

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val   = range(len(y), len(y)+500)

    if data_seed is not None:
        chaos = np.random.RandomState( data_seed )
        allidx = list(idx_train) + list(idx_val) + idx_test
        chaos.shuffle( allidx )

        idx_train = allidx[:len(y)]
        idx_val   = allidx[len(y):len(y)+500]
        idx_test  = allidx[len(y)+500:]

    train_mask = sample_mask( idx_train, labels.shape[0] )
    val_mask   = sample_mask( idx_val, labels.shape[0]   )
    test_mask  = sample_mask( idx_test, labels.shape[0]  )

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # output some statistics
    print( 'nodes', adj.shape[0] )

    rows,cols = adj.nonzero()
    nedges = 0
    for row,col in zip(rows,cols):
        if adj[row,col] != 1:
            print( 'non 1:', row, col, adj[row,col] )
            sys.exit(0)

        if adj[col,row] != 1:
            print( 'link has no reverse link', row, col )
            sys.exit(0)

        if row != col:  nedges += 1
    print('nedges', nedges/2)

    # check connectivity
    n_connected = nx.number_connected_components( _nxgraph )
    print( '#components:', n_connected )

    #largest_cc = max( nx.connected_component_subgraphs( _nxgraph ), key=len )
    #print( 'diameter:', nx.diameter( largest_cc ) )

    subgraphs = np.zeros( [adj.shape[0], n_connected] )
    for i, idx in enumerate( nx.connected_components( _nxgraph ) ):
        subgraphs[list(idx),i] = 1

    #import matplotlib.pyplot as plt
    #plt.hist( subgraphs.sum(0) ); plt.show()

    print( 'dim(x)', features.shape[1] )
    print( 'dim(y)', y_train.shape[1] )
    print( 'train:valid:test={}:{}:{}'.format( train_mask.sum(), val_mask.sum(), test_mask.sum() ) )

    return adj, subgraphs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def construct_feed_dict( noise, features, support, labels, labels_mask, placeholders ):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update( {placeholders['noise']: noise} )
    feed_dict.update( {placeholders['labels']: labels} )
    feed_dict.update( {placeholders['labels_mask']: labels_mask} )
    feed_dict.update( {placeholders['features']: features} )
    feed_dict.update( {placeholders['support'][i]: support[i] for i in range(len(support))} )
    feed_dict.update( {placeholders['num_features_nonzero']: features[1].shape} )
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple( features )

def sym_normalize_adj( adj ):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_inv_sqrt = np.power( np.maximum( degree, np.finfo(float).eps ), -0.5 )
    d_mat_inv_sqrt = sp.diags( d_inv_sqrt )
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def row_normalize_adj( adj ):
    '''row normalize adjacency matrix'''

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_mat_inv = sp.diags( 1 / np.maximum( degree, np.finfo(float).eps ) )
    return d_mat_inv.dot( adj ).tocoo()

def preprocess_adj( adj, selfloop_weight=1 ):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    return sym_normalize_adj( adj + selfloop_weight * sp.eye(adj.shape[0]) )

def preprocess_high_order_adj( adj, order, eps ):
    adj = row_normalize_adj( adj )

    adj_sum = adj
    cur_adj = adj
    for i in range( 1, order ):
        cur_adj = cur_adj.dot( adj )
        adj_sum += cur_adj
    adj_sum /= order

    adj_sum.setdiag( 0 )
    adj_sum.data[adj_sum.data<eps] = 0
    adj_sum.eliminate_zeros()

    adj_sum += sp.eye( adj.shape[0] )
    return sym_normalize_adj( adj_sum + adj_sum.T )

