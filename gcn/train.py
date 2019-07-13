#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os, gc, random, time, itertools
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import scipy.sparse as sp
import numpy as np

from utils import sparse_to_tuple, construct_feed_dict, chebyshev_polynomials, \
                  preprocess_features, preprocess_adj, preprocess_high_order_adj
from data_io import load_data, PLANETOID_DATA, PITFALL_DATA
from models import GCN, MLP
from block_krylov import block_krylov

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_enum(  'dataset', 'cora', PLANETOID_DATA + PITFALL_DATA, 'Dataset' )
flags.DEFINE_enum(  'model', 'fishergcn',
                   [ 'gcn', 'gcnT', 'gcnR', 'fishergcn', 'fishergcnT', 'mlp', 'chebynet' ],
                     'Model' )

flags.DEFINE_float(   'lrate', 0.01, 'initial learning rate.' )
flags.DEFINE_float(   'dropout', 0.5, 'Dropout rate (1 - keep probability).' )
flags.DEFINE_integer( 'epochs', 500, 'Number of epochs to train.' )
flags.DEFINE_list(    'hidden', ['64',], 'size of hidden layer(s)' )
flags.DEFINE_float(   'weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.' )
flags.DEFINE_integer( 'early_stop', 2, 'early_stop strategy' )        # 0: no stop 1: simple early stop 2: more strict conditions
flags.DEFINE_boolean( 'save', False, 'save npz file which contains the learning results' )
flags.DEFINE_integer( 'max_degree', 3, 'Maximum Chebyshev polynomial degree.' )
flags.DEFINE_integer( 'init_seed', 2019, 'random seed' )
flags.DEFINE_integer( 'data_seed', 2019, 'random seed' )
flags.DEFINE_integer( 'randomsplit', 0, 'random split of train:valid:test' ) # 0/1/2.... random split is recommended for a more complete comparison
flags.DEFINE_integer( 'repeat', 20, 'number of repeats' )

# for gcnT
flags.DEFINE_integer( 'order', 5, 'order of high-order GCN' )
flags.DEFINE_float(   'threshold', 1e-4, 'A threshold to apply nodes filtering on random walk matrix.' )

# for gcnR
flags.DEFINE_float(   'flip_prob', 1e-3, 'randomly add/remove flip_prob neighbour per node' )

# Fisher-GCN corresponds to fisher_freq=1 & fisher_adversary=1; other setting of these two parameters are varations
# in practice, one only needs to tune the fisher_noise parameter
flags.DEFINE_float(   'fisher_noise', 0.1, 'noise level' )
flags.DEFINE_integer( 'fisher_rank', 10, 'dimension of the noise' )
flags.DEFINE_integer( 'fisher_perturbation',  5, 'number of pertubations' ) # the smaller the quicker but worse
flags.DEFINE_integer( 'fisher_freq', 1, 'high frequency noise' ) # 0/1/2 for low/high/random frenquency
flags.DEFINE_integer( 'fisher_adversary', 1, 'adversary noise' ) # 0: plain noise; 1: adversary noise

def make_perturbation( V, w, noise_level, adversary ):
    if adversary > 0:
        # this is a different noise parametrization
        #_dirs = tf.get_variable( 'perturbation', shape=(FLAGS.fisher_rank, FLAGS.fisher_rank), dtype=tf.float32 )
        #_dirs = tf.matmul( _dirs, _dirs, transpose_b=True )
        #pradius = tf.sigmoid( tf.get_variable( 'perturbation_radius', dtype=tf.float32, shape=() ) )
        #_dirs =  pradius * _dirs / tf.trace( _dirs )
        #perturb = tf.matmul( _dirs, tf.random.normal( shape=(FLAGS.fisher_rank,FLAGS.fisher_perturbation), dtype=tf.float32 ) )

        perturb = tf.random.uniform( shape=(FLAGS.fisher_rank,FLAGS.fisher_perturbation), minval=-.5, maxval=.5, dtype=tf.float32 )
        _scaling = tf.sigmoid( tf.get_variable( 'perturbation_scaling', shape=(FLAGS.fisher_rank, 1), dtype=tf.float32 ) )
        perturb = _scaling * perturb

    else:
        perturb = tf.random.uniform( shape=(FLAGS.fisher_rank,FLAGS.fisher_perturbation), minval=-.5, maxval=.5, dtype=tf.float32 )

    if FLAGS.fisher_freq != 0:
        # original density matrix
        ptensor  = tf.constant( w/w.sum(), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )

        # Fisher-Bures metric
        metric_w = tf.constant( 1/np.sqrt(w/w.sum()), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )

        # perturbed density matrix
        new_p = ptensor * tf.exp( noise_level * metric_w * perturb )
        new_p = new_p / tf.reduce_sum( new_p, axis=0 )

        # the perturbation
        inc_w = ( new_p - ptensor ) * w.sum()

    else:
        # use additive noise for perturbing eigenvectors wrt small eigenvalues
        # in this case the metric may be singular or numerical instable
        #metric_w = tf.constant( np.sqrt(w/w.mean()), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )
        inc_w = noise_level * perturb

    return tf.constant( V, dtype=tf.float32, name='FisherV' ), inc_w

def build_model( adj, features, n_classes, subgraphs ):
    perturbation = None
    placeholders = {
        'features': tf.sparse_placeholder( tf.float32, shape=tf.constant(features[2],dtype=tf.int64) ),
        'labels': tf.placeholder(tf.float32, shape=(None, n_classes)),
        'labels_mask': tf.placeholder(tf.int32),
        'noise': tf.placeholder( tf.float32, shape=() ),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }

    if FLAGS.model == 'gcn':
        support = [ sparse_to_tuple( preprocess_adj(adj) ) ]
        model_func = GCN

    elif FLAGS.model == 'gcnR':
        support = [ sparse_to_tuple( adj ) ]
        model_func = GCN

    elif FLAGS.model == 'gcnT':
        support = [ sparse_to_tuple( preprocess_high_order_adj( adj, FLAGS.order, FLAGS.threshold ) ) ]
        model_func = GCN

    elif FLAGS.model == 'fishergcn' or FLAGS.model == 'fishergcnT':

        if FLAGS.model == 'fishergcn':
            A = preprocess_adj( adj )
        else:
            A = preprocess_high_order_adj( adj, FLAGS.order, FLAGS.threshold )

        N = adj.shape[0]
        L = sp.eye( N ) - A

        if FLAGS.fisher_freq == 0:
            #nsubgraphs = subgraphs.shape[1]
            #V = block_krylov( A, FLAGS.fisher_rank+nsubgraphs )
            #V = V[:,:FLAGS.fisher_rank]

            V = block_krylov( A, FLAGS.fisher_rank )
            w = ( sp.csr_matrix.dot( L, V ) * V ).sum(0)

        elif FLAGS.fisher_freq == 1:
            # if the graph contains one large component and small isolated components
            # only perturb the largest connected component
            subgraph_sizes = subgraphs.sum( 0 )
            largest_idx = np.argmax( subgraph_sizes )
            isolated = np.nonzero( 1-subgraphs[:,largest_idx] )[0]
            L = L.tolil()
            L[:,isolated] = 0
            L[isolated,:] = 0
            L = L.tocsr()

            V = block_krylov( L, FLAGS.fisher_rank )
            w = ( sp.csr_matrix.dot( L, V ) * V ).sum(0)

        elif FLAGS.fisher_freq == 2:
            V, _ = np.linalg.qr( np.random.randn(N, FLAGS.fisher_rank) )
            w = np.ones( FLAGS.fisher_rank )

        else:
            print( 'unknown frequency:', FLAGS.fisher_freq )
            sys.exit(0)

        perturbation = make_perturbation( V, w, placeholders['noise'], FLAGS.fisher_adversary )
        support    = [ sparse_to_tuple( A ) ]
        model_func = GCN

    elif FLAGS.model == 'chebynet':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        model_func = GCN

    elif FLAGS.model == 'mlp':
        support = [ sparse_to_tuple( preprocess_adj(adj) ) ]
        model_func = MLP

    else:
        raise ValueError( 'Invalid argument for model: ' + str(FLAGS.model) )

    try:
        _, _values, _shape = support[0]
        print( "sparsity: {0:.2f}%".format( 100*(_values>0).sum() / (_shape[0]*_shape[1]) ) )
    except:
        pass
    placeholders['support'] = [ tf.sparse_placeholder(tf.float32) for _ in support ]

    model = model_func( placeholders,
                        perturbation=perturbation,
                        subgraphs=subgraphs )
    return model, support, placeholders

def exp( dataset, data_seed, init_seed ):
    '''
    dataset - name of dataset
    data_seed - data_seed corresponds to train/dev/test split
    init_seed - seed for initializing NN weights
    '''
    print( 'running {} on {}'.format( FLAGS.model, dataset ) )

    tf.reset_default_graph()
    adj, subgraphs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data( dataset, data_seed )
    features = preprocess_features( features )

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #config.log_device_placement = True
    config.gpu_options.allow_growth = True

    train_loss = []
    train_acc  = []
    valid_loss = []
    valid_acc  = []
    with tf.Graph().as_default():
        random.seed( init_seed )
        np.random.seed( init_seed )
        tf.set_random_seed( init_seed )

        with tf.Session( config=config ) as sess:

            model, support, placeholders = build_model( adj, features, y_train.shape[1], subgraphs )
            sess.run( tf.global_variables_initializer() )

            def evaluate( labels, labels_mask, noise=0., dropout=0. ):
                feed_dict_val = construct_feed_dict( features, support, labels, labels_mask, placeholders, noise, dropout )
                outs_val = sess.run( [model.loss, model.accuracy], feed_dict=feed_dict_val )
                return outs_val[0], outs_val[1]

            start_t = time.time()
            for epoch in range( FLAGS.epochs ):
                feed_dict = construct_feed_dict( features, support, y_train, train_mask, placeholders,
                                                 FLAGS.fisher_noise, FLAGS.dropout )
                feed_dict.update( {tf.keras.backend.learning_phase(): 1} )
                outs = sess.run( [model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict )
                train_loss.append( outs[1] )
                train_acc.append( outs[2] )

                # Validation
                outs = evaluate( y_val, val_mask )
                valid_loss.append( outs[0] )
                valid_acc.append( outs[1] )

                if ( epoch + 1 ) % 10 == 0:
                    print("Epoch:", '%04d' % (epoch + 1),
                          "train_loss=", "{:.5f}".format(train_loss[-1]),
                          "train_acc=", "{:.5f}".format(train_acc[-1]),
                          "val_loss=", "{:.5f}".format(valid_loss[-1]),
                          "val_acc=", "{:.5f}".format(valid_acc[-1]) )
                    #print( 'perterbation radius:', sess.run( pradius ) )

                if FLAGS.early_stop == 0:
                    if epoch > 10 and ( train_loss[-1] > 1.5 * train_loss[0] or np.isnan(train_loss[-1]) ):
                        print( "Early stopping at epoch {}...".format( epoch ) )
                        break

                elif FLAGS.early_stop == 1:    # simple early stopping
                    if epoch > 20 and valid_loss[-1] > np.mean( valid_loss[-10:] ) \
                                  and valid_acc[-1] < np.mean( valid_acc[-10:] ):
                        print( "Early stopping at epoch {}...".format( epoch ) )
                        break

                elif FLAGS.early_stop == 2:    # more strict conditions
                    if epoch > 100 \
                        and np.mean( valid_loss[-10:] ) > np.mean( valid_loss[-100:] ) \
                        and np.mean( valid_acc[-10:] ) < np.mean( valid_acc[-100:] ):
                            print( "Early stopping at epoch {}...".format( epoch ) )
                            break
                else:
                    print( 'unknown early stopping strategy:', FLAGS.early_stop )
                    sys.exit(0)

            test_loss, test_acc = evaluate( y_test, test_mask )
            sec_per_epoch = ( time.time() - start_t ) / epoch
            print( "Test set results:", "loss=", "{:.5f}".format(test_loss),
                   "accuracy=", "{:.5f}".format(test_acc),
                   "epoch_secs=", "{:.2f}".format(sec_per_epoch) )

    tf.reset_default_graph()

    return {
        'train_loss': train_loss,
        'train_acc':  train_acc,
        'valid_loss': valid_loss,
        'valid_acc':  valid_acc,
        'test_loss':  test_loss,
        'test_acc':   test_acc,
    }

def analyse( dataset, result, ofilename ):
    avg_epochs = np.mean( [ len(r['train_loss']) for r in result ] )
    min_epochs = min(     [ len(r['train_loss']) for r in result ] )

    def lcurve( key ):
        return np.array( [ r[key][:min_epochs] for r in result ] )

    final_result = np.array( [ ( r['train_loss'][-1], r['train_acc'][-1],
                               r['valid_loss'][-1], r['valid_acc'][-1],
                               r['test_loss'], r['test_acc'] )
                               for r in result ] )
    _mean = np.mean( final_result, axis=0 )
    _std  = np.std( final_result, axis=0 )

    if FLAGS.save: np.savez( ofilename,
                             train_loss=lcurve('train_loss'),
                             train_acc=lcurve('train_acc'),
                             valid_loss=lcurve('valid_loss'),
                             valid_acc=lcurve('valid_acc'),
                             test_loss=final_result[:,4],
                             test_acc=final_result[:,5],
                             scores=_mean, std=_std )

    def _final_print( name, i, j ):
        print( '{} {} {} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
               FLAGS.model, dataset, name,
               np.round( _mean[i], 2 ),
               np.round( _std[i], 2 ),
               np.round( _mean[j], 2 ),
               np.round( _std[j], 2 ) ) )

    print( '{} {} final_life  {:.0f}'.format( FLAGS.model, dataset, avg_epochs ) )
    _final_print( 'final_train', 0, 1 )
    _final_print( 'final_valid', 2, 3 )
    _final_print( 'final_test',  4, 5 )

def main( argv ):
    FLAGS.hidden = [ int(h) for h in FLAGS.hidden ]

    start_t = time.time()
    result = []

    if FLAGS.randomsplit > 0:
        data_seeds = range( FLAGS.data_seed, FLAGS.data_seed+FLAGS.randomsplit )
    else:
        data_seeds = [ None ]
    init_seeds = range( FLAGS.init_seed, FLAGS.init_seed+FLAGS.repeat )

    for _data_seed, _init_seed in itertools.product( data_seeds, init_seeds ):
        result.append( exp( FLAGS.dataset, _data_seed, _init_seed ) )
        gc.collect()

    ofilename = "{}_{}_lr{}_drop{}_reg{}_hidden{}_early{}_seed{}_{}_repeat{}_{}".format(
                FLAGS.model, FLAGS.dataset, FLAGS.lrate, FLAGS.dropout,
                FLAGS.weight_decay, FLAGS.hidden, FLAGS.early_stop,
                FLAGS.init_seed, FLAGS.data_seed,
                FLAGS.randomsplit, FLAGS.repeat )

    if 'fisher' in FLAGS.model:
        ofilename += "_rank{}_perturb{}_noise{}_freq{}_adv{}".format(
                      FLAGS.fisher_rank, FLAGS.fisher_perturbation,
                      FLAGS.fisher_noise, FLAGS.fisher_freq, FLAGS.fisher_adversary )
    analyse( FLAGS.dataset, result, ofilename )

    print( 'finished in {:.2f} hours'.format( (time.time()-start_t)/3600 ) )

if __name__ == '__main__':
    app.run( main )
