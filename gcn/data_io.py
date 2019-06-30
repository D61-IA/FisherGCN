from collections import Counter
from pathlib import Path

import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import numpy as np
import sys

PLANETOID_DATA = [
                   'cora','citeseer','pubmed',
                   'nell.0.1', 'nell.0.01', 'nell.0.001',
                 ]
PITFALL_DATA = [ 'amazon_electronics_computers',
                 'amazon_electronics_photo',
                 'ms_academic_cs',
                 'ms_academic_phy',
                 'cora_full' ]

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels

def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph

def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)

def eliminate_self_loops( G ):
    def remove_self_loop( A ):
        """Remove self-loops from the adjacency matrix."""
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()
        A.eliminate_zeros()
        return A

    G.adj_matrix = remove_self_loop( G.adj_matrix )
    return G

def remove_underrepresented_classes( g, train_examples_per_class, val_examples_per_class ):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.
    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph( g, nodes_to_keep=keep_indices )

def load_npz_to_sparse_graph( data_path ):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """

    if not str(data_path).endswith('.npz'):
        data_path = data_path.joinpath( '.npz' )

    if not Path.is_file( data_path ):
        raise ValueError( f"{data_path} doesn't exist." )

    with np.load( data_path, allow_pickle=True ) as loader:
        loader = dict(loader)

        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])
        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_sparse_graph_to_npz(filepath, sparse_graph):
    """Save a SparseGraph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    data_dict = {
        'adj_data': sparse_graph.adj_matrix.data,
        'adj_indices': sparse_graph.adj_matrix.indices,
        'adj_indptr': sparse_graph.adj_matrix.indptr,
        'adj_shape': sparse_graph.adj_matrix.shape
    }
    if sp.isspmatrix(sparse_graph.attr_matrix):
        data_dict['attr_data'] = sparse_graph.attr_matrix.data
        data_dict['attr_indices'] = sparse_graph.attr_matrix.indices
        data_dict['attr_indptr'] = sparse_graph.attr_matrix.indptr
        data_dict['attr_shape'] = sparse_graph.attr_matrix.shape
    elif isinstance(sparse_graph.attr_matrix, np.ndarray):
        data_dict['attr_matrix'] = sparse_graph.attr_matrix

    if sp.isspmatrix(sparse_graph.labels):
        data_dict['labels_data'] = sparse_graph.labels.data
        data_dict['labels_indices'] = sparse_graph.labels.indices
        data_dict['labels_indptr'] = sparse_graph.labels.indptr
        data_dict['labels_shape'] = sparse_graph.labels.shape
    elif isinstance(sparse_graph.labels, np.ndarray):
        data_dict['labels'] = sparse_graph.labels

    if sparse_graph.node_names is not None:
        data_dict['node_names'] = sparse_graph.node_names

    if sparse_graph.attr_names is not None:
        data_dict['attr_names'] = sparse_graph.attr_names

    if sparse_graph.class_names is not None:
        data_dict['class_names'] = sparse_graph.class_names

    if sparse_graph.metadata is not None:
        data_dict['metadata'] = sparse_graph.metadata

    if not filepath.endswith('.npz'):
        filepath += '.npz'

    np.savez(filepath, **data_dict)

def sample_mask( idx, l ):
    """Create boolean mask."""

    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

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

    from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
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

def load_pitfall_dataset( dataset_str,
                          data_seed,
                          standardize_graph = True,
                          train_examples_per_class = 20,
                          val_examples_per_class = 30 ):
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

    if dataset_str not in PITFALL_DATA:
        raise ValueError( 'Wrong dataset name!' )

    parent_path = Path(__file__).resolve().parents[1]
    data_path = parent_path.joinpath( "data/{}.npz".format(dataset_str) )
    dataset_graph = load_npz_to_sparse_graph( data_path )

    # some standardization preprocessing
    if standardize_graph:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    if train_examples_per_class is not None and val_examples_per_class is not None:
        if dataset_str == 'cora_full':
            # cora_full has some classes that have very few instances. We have to remove these in order for
            # split generation not to fail
            dataset_graph = remove_underrepresented_classes( dataset_graph,
                                                             train_examples_per_class, val_examples_per_class )
            dataset_graph = dataset_graph.standardize()
            # To avoid future bugs: the above two lines should be repeated to a fixpoint, otherwise code below might
            # fail. However, for cora_full the fixpoint is reached after one iteration, so leave it like this for now.

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

def load_planetoid_dataset( dataset_str, data_seed ):

    if dataset_str not in PLANETOID_DATA:
        raise ValueError( 'Wrong dataset name!' )

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

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

    adj = adj.astype(np.float32)
    features = features.tocsr()
    features = features.astype(np.float32)

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

    return _nxgraph, adj, features, labels, idx_train, idx_val, idx_test

def load_data( dataset_str, data_seed ):
    """
    Loads input data from data directory
    :param dataset_str: Dataset name
           data_seed:  random seed to split the train/test/dev datasets
                       if None then use the original split
    :return: All data input files loaded (as well the training/test data).
    """

    dataset_str = dataset_str.lower()
    if dataset_str in PITFALL_DATA:
        adj, features, labels, idx_train, idx_val, idx_test = load_pitfall_dataset( dataset_str, data_seed )
        _nxgraph = nx.from_scipy_sparse_matrix( adj )
    else:
        _nxgraph, adj, features, labels, idx_train, idx_val, idx_test = load_planetoid_dataset( dataset_str, data_seed )

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

    subgraphs = np.zeros( [adj.shape[0], n_connected], dtype=np.int )
    for i, idx in enumerate( nx.connected_components( _nxgraph ) ):
        subgraphs[list(idx),i] = 1
    subgraph_sizes = sorted( subgraphs.sum( 0 ), reverse=True )
    print( 'component size:', subgraph_sizes[:5], '...' )

    #import matplotlib.pyplot as plt
    #plt.hist( subgraphs.sum(0) ); plt.show()

    print( 'dim(x)', features.shape[1] )
    print( 'dim(y)', y_train.shape[1] )
    print( 'train:valid:test={}:{}:{}'.format( train_mask.sum(), val_mask.sum(), test_mask.sum() ) )

    return adj, subgraphs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
