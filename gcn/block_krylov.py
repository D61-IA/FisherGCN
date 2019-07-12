from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.sparse as sp
import scipy.linalg

'''
This is an implementation of the SVD algorithm 2

Cameron Musco and Christopher Musco.
Randomized Block Krylov Methods for Stronger and Faster Approximate Singular Value Decomposition.
NIPS 18, 2015.
'''

def block_krylov( A, k, eps=0.01, max_itrs=50 ):
    '''
       A is a nxn sparse symmetric matrix
       get the eigenvectors of A corresponding the eigenvalues with the largest manitude
       arranged in a nxk matrix
    '''

    max_itrs = min( int( np.log( A.shape[1] ) / np.sqrt( eps ) ), max_itrs )
    print( 'running block krylov for {} iterations'.format( max_itrs ) )

    X = sp.coo_matrix.dot( A, np.random.randn( A.shape[1], k ) )
    Asq = sp.coo_matrix.dot( A, A )

    K = [ X ]
    for i in range( max_itrs ):
        K.append( sp.coo_matrix.dot( Asq, K[-1] ) )
    K = np.hstack( K )

    Q, R = np.linalg.qr( K )
    M = A.dot( A.dot(Q) ).transpose().dot( Q )

    _w, Uk = scipy.linalg.eigh( M, eigvals=(M.shape[0]-k,M.shape[0]-1) )

    return Q.dot( Uk )
