import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

def sparse_to_tuple( sparse_mx ):
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

def construct_feed_dict( features, support, labels, labels_mask, placeholders,
                         noise, dropout ):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update( {placeholders['features'].indices: features[0]} )
    feed_dict.update( {placeholders['features'].values:  features[1]} )
    feed_dict.update( {placeholders['support'][i]: _sup for i, _sup in enumerate(support) } )
    feed_dict.update( {placeholders['labels']: labels} )
    feed_dict.update( {placeholders['labels_mask']: labels_mask} )

    feed_dict.update( {placeholders['noise']: noise} )
    feed_dict.update( {placeholders['dropout']: dropout } )
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = sym_normalize_adj(adj)
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
