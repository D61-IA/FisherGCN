import numpy as np
import pickle as pkl
import random
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from pathlib import Path
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize

from data_io import *

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

def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))

def to_binary_bag_of_words(features):
    """Converts TF/IDF features to binary bag-of-words features."""
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy

def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def load_spitfall_datasets(dataset_str,
                            data_seed,
                            standardize_graph = True,
                            train_examples_per_class = 20,
                            val_examples_per_class = 30):
    """Load amazon computers/photo, ms_academic_cs, and ms_academic_phy datasets"""
    """
    :param dataset_str: Dataset name
           standardize_graph:  Standardizing includes
                                1. Making the graph undirected
                                2. Making the graph unweighted
                                3. Selecting the largest connected component (LCC)
           split: Number of samples (i.e. nodes) in the train/val/test sets.

           (Note: The default values above are the same value as pitfall paper https://arxiv.org/pdf/1811.05868.pdf)
    :return: All data input files loaded (as well the training/test data).
    """

    spitfall_data = ['amazon_electronics_computers', 'amazon_electronics_photo', 'ms_academic_cs', 'ms_academic_phy']
    if dataset_str not in spitfall_data:
        raise ValueError('Wrong dataset name!')

    parent_path = Path(__file__).resolve().parents[1]
    data_path = parent_path.joinpath( "data/{}.npz".format(dataset_str) )
    dataset_graph = load_dataset( data_path )

    # some standardization preprocessing
    if standardize_graph:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    adj, features, labels = dataset_graph.unpack()
    labels = binarize_labels(labels)

    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(features):
        features = to_binary_bag_of_words(features)

    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    assert (adj != adj.T).nnz == 0
    # features need to be binary bag-of-word vectors
    assert is_binary_bag_of_words(features), f"Non-binary node_features entry!"


    random_state = np.random.RandomState(data_seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels,
                            train_examples_per_class=train_examples_per_class,
                            val_examples_per_class=val_examples_per_class)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data( dataset_str, data_seed ):
    """
    Loads input data from gcn/data directory
    :param dataset_str: Dataset name
           data_seed:  random seed to split the train/test/dev datasets
                       if None then use the original split
    :return: All data input files loaded (as well the training/test data).
    """

    if 'amazon_electronics' in dataset_str or 'ms_academic' in dataset_str:
        adj, features, labels, idx_train, idx_val, idx_test = load_spitfall_datasets(dataset_str, data_seed)
        _nxgraph = nx.from_scipy_sparse_matrix( adj )
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        parent_path = Path(__file__).resolve().parents[1]
        for i in range( len(names) ):
            data_path = parent_path.joinpath( "data/ind.{}.{}".format(dataset_str, names[i]) )
            with open( data_path, 'rb' ) as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        data_path = parent_path.joinpath( "data/ind.{}.test.index".format(dataset_str) )
        test_idx_reorder = parse_index_file( data_path )
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

        elif 'nell.0' in dataset_str:
            # Find relation nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(allx.shape[0], len(graph))
            isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-allx.shape[0], :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-allx.shape[0], :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        if 'nell.0' in dataset_str:
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)

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
    print( '#nodes', adj.shape[0] )

    # nedges = 0
    # for _n in graph: nedges += len(graph[_n])
    # print( '#edges (raw) {:.0f}'.format( nedges/2 ) )

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
    print('#edges (no duplicates; no self-links) {:.0f}'.format( nedges/2 ) )

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
    rowsum = np.array(features.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features.eliminate_zeros()

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
