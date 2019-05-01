#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os, time, shutil
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import scipy.sparse as sp
from utils import *
from models import GCN, MLP
from block_krylov import block_krylov

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string(  'dataset', 'cora', 'Dataset string.' )          # 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo'
flags.DEFINE_integer( 'randomsplit', 0, 'random split of train:valid:test' ) # 0/1/2.... random split is recommended for a more complete comparison

flags.DEFINE_string(  'model',   'fishergcn',    'Model string.' )    # 'gcn', 'gcnR', 'gcnT', 'fishergcn', 'fishergcnT', 'gcn_cheby', 'dense'
flags.DEFINE_float(   'learning_rate', 0.01, 'Initial learning rate.' )
flags.DEFINE_float(   'dropout', 0.5, 'Dropout rate (1 - keep probability).' )
flags.DEFINE_integer( 'epochs', 500, 'Number of epochs to train.' )
flags.DEFINE_integer( 'hidden1', 64, 'Number of units in hidden layer 1.' )
flags.DEFINE_float(   'weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.' )
flags.DEFINE_integer( 'early_stop', 2, 'early_stop strategy' )        # 0: no stop 1: simple early stop 2: more strict conditions
flags.DEFINE_boolean( 'save', False, 'save npz file which contains the learning results' )
flags.DEFINE_boolean( 'retrace', False, 'recover the model with the minimum validation loss' )
flags.DEFINE_integer( 'max_degree', 3, 'Maximum Chebyshev polynomial degree.' )
flags.DEFINE_integer( 'seed',   2019, 'random seed' )
flags.DEFINE_integer( 'repeat', 20, 'number of repeats' )

# for gcnT
flags.DEFINE_integer( 'order', 5, 'order of high-order GCN' )
flags.DEFINE_float(   'threshold', 1e-4, 'A threshold to apply nodes filtering on random walk matrix.' )

# for gcnR
flags.DEFINE_float(   'mask_prob', 0.0001, 'corruption rate of the adjacency matrix' )

# Fisher-GCN corresponds to fisher_freq=1 & fisher_adversary=1; other setting of these two parameters are varations
# in practice, one only needs to tune the fisher_noise parameter
flags.DEFINE_float(   'fisher_noise', 0.1, 'noise level' )
flags.DEFINE_integer( 'fisher_rank', 10, 'dimension of the noise' )
flags.DEFINE_integer( 'fisher_perturbation',  5, 'number of pertubations' ) # the smaller the quicker but worse
flags.DEFINE_integer( 'fisher_freq', 1, 'high frequency noise' ) # 0/1/2 for low/high/random frenquency
flags.DEFINE_integer( 'fisher_adversary', 1, 'adversary noise' ) # 0: plain noise; 1: adversary noise

def exp( run, dataset, diag_tensor=False, data_seed=None ):
    tf.reset_default_graph()
    np.random.seed( FLAGS.seed + run )
    tf.set_random_seed( FLAGS.seed + run )

    adj, subgraphs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data( dataset, data_seed )
    features = preprocess_features(features)
    perturbation = None

    placeholders = {
        'features': tf.sparse_placeholder( tf.float32, shape=tf.constant(features[2],dtype=tf.int64) ),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'noise': tf.placeholder( tf.float32, shape=() ),
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
            nsubgraphs = subgraphs.shape[1]
            V = block_krylov( A, FLAGS.fisher_rank+nsubgraphs )
            V = V[:,:FLAGS.fisher_rank]
            w = ( sp.csr_matrix.dot( L, V ) * V ).sum(0)
            #w, V = sp.linalg.eigsh( A, k=FLAGS.fisher_rank+nsubgraphs )

        elif FLAGS.fisher_freq == 1:
            V = block_krylov( L, FLAGS.fisher_rank )
            w = ( sp.csr_matrix.dot( L, V ) * V ).sum(0)

        elif FLAGS.fisher_freq == 2:
            V, _ = np.linalg.qr( np.random.randn(N, FLAGS.fisher_rank) )
            w = np.ones( FLAGS.fisher_rank )

        else:
            print( 'unknown frequency:', FLAGS.fisher_freq )
            sys.exit(0)

        if FLAGS.fisher_adversary > 0:
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

        ptensor  = tf.constant( w/w.sum(), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )
        metric_w = tf.constant( 1/np.sqrt(w/w.sum()), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )
        new_p = ptensor * tf.exp( placeholders['noise'] * metric_w * perturb )
        new_p = new_p / tf.reduce_sum( new_p, axis=0 )
        inc_w = ( new_p - ptensor ) * w.sum()

        # additive noise
        #metric_w = tf.constant( np.sqrt(w/w.mean()), dtype=tf.float32, shape=(FLAGS.fisher_rank,1) )
        #inc_w = placeholders['noise'] * metric_w * perturb

        perturbation = ( tf.constant( V, dtype=tf.float32, name='FisherV' ), inc_w )
        support    = [ sparse_to_tuple( A ) ]
        model_func = GCN

    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN

    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj) + 1]  # Not used
        model_func = MLP

    else:
        raise ValueError( 'Invalid argument for model: ' + str(FLAGS.model) )

    _, _values, _shape = support[0]
    print( 'running {} on {} (trial {})'.format( FLAGS.model, dataset, run+1 ) )
    print( "sparsity: {0:.2f}%".format( 100*(_values>0).sum() / (_shape[0]*_shape[1]) ) )
    placeholders['support'] = [ tf.sparse_placeholder(tf.float32) for _ in support ]
    model = model_func( placeholders, input_dim=features[2][1], input_rows=features[2][0], diag_tensor=diag_tensor, perturbation=perturbation, logging=True, subgraphs=subgraphs )

    if FLAGS.retrace:
        saver = tf.train.Saver( max_to_keep=FLAGS.early_stop )
        runid = '.{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                  dataset, FLAGS.model, FLAGS.learning_rate,
                  FLAGS.weight_decay, FLAGS.dropout, FLAGS.fisher_noise,
                  time.time(), np.random.randint( 0, int(1e18) ), run )
        os.mkdir( runid )

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #config.log_device_placement = True
    config.gpu_options.allow_growth = True

    if 'COLAB_TPU_ADDR' not in os.environ:
        tpu_addr = None
        sess = tf.Session( config=config )
    else:
        tpu_addr = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        sess = tf.Session( tpu_addr, config=config )
        sess.run( tf.contrib.tpu.initialize_system() )

    # Define model evaluation function
    def evaluate( noise, features, support, labels, mask, placeholders ):
        t_test = time.time()
        feed_dict_val = construct_feed_dict( noise, features, support, labels, mask, placeholders )
        outs_val = sess.run( [model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val )
        return outs_val[0], outs_val[1], (time.time() - t_test)

    sess.run( tf.global_variables_initializer() )

    history = []
    history_files = []
    for epoch in range( FLAGS.epochs ):
        t = time.time()

        feed_dict = construct_feed_dict( FLAGS.fisher_noise, features, support, y_train, train_mask, placeholders )
        feed_dict.update( {placeholders['dropout']: FLAGS.dropout} )

        outs = sess.run( [model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict )

        # Validation
        if FLAGS.retrace:
            _file = saver.save( sess, '{}/gcn'.format( runid ), global_step=epoch )
        else:
            _file = None
        val_cost, val_acc, duration = evaluate( 0.0, features, support, y_val, val_mask, placeholders )

        history_files.append( _file )
        history.append( (outs[1], outs[2], val_cost, val_acc) )
        # tuples in the form ( train_cost, train_acc, val_cost, val_acc )

        if ( epoch + 1 ) % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]),
                  "val_loss=", "{:.5f}".format(val_cost),
                  "val_acc=", "{:.5f}".format(val_acc),
                  "time=", "{:.5f}".format(time.time() - t))
            #print( 'perterbation radius:', sess.run( pradius ) )

        if FLAGS.early_stop == 0:
            pass

        elif FLAGS.early_stop == 1:    # simple early stopping
            if epoch > 20 and history[-1][2] > np.mean( [r[2] for r in history[-11:-1]] ) \
                          and history[-1][3] < np.mean( [r[3] for r in history[-11:-1]] ):
                print( "Early stopping at epoch {}...".format( epoch ) )
                break

        elif FLAGS.early_stop == 2:    # more strict conditions
            if epoch > 100 \
                and np.mean( [r[2] for r in history[-10:]] ) > np.mean( [r[2] for r in history[-100:]] ) \
                and np.mean( [r[3] for r in history[-10:]] ) < np.mean( [r[3] for r in history[-100:]] ):
                    print( "Early stopping at epoch {}...".format( epoch ) )
                    break
        else:
            print( 'unknown early stopping strategy:', FLAGS.early_stop )
            sys.exit(0)

    if FLAGS.retrace:
        history = [ _r for _r in zip(history_files,history) if os.access( _r[0]+".meta", os.R_OK ) ]
        history.sort( key=lambda r:r[4], reverse=True )
        sess = tf.Session()
        saver.restore( sess, history[0][0] )
        shutil.rmtree( runid )

    test_cost, test_acc, test_duration = evaluate( 0.0, features, support, y_test, test_mask, placeholders )
    print( "Test set results:", "cost=", "{:.5f}".format(test_cost),
           "accuracy=", "{:.5f}".format(test_acc),
           "time=", "{:.5f}".format(test_duration) )

    if tpu_addr is not None: sess.run( tf.contrib.tpu.shutdown_system() )
    sess.close()

    return history, test_cost, test_acc

def analyse( dataset, result, ofilename ):
    min_epochs = min( [ len(r[0]) for r in result ] )
    lcurves    = np.array( [ r[0][:min_epochs] for r in result ] )

    final_result = np.array( [ r[0][-1]+(r[1], r[2]) for r in result ] )
    scores = np.mean( final_result, axis=0 )
    std    = np.std( final_result, axis=0 )

    if FLAGS.save: np.savez( ofilename, lcurves=lcurves, scores=scores, std=std )

    print( '{} {} final_train {:.3f} {:.3f}'.format( FLAGS.model, dataset, scores[0], scores[1] ) )
    print( '{} {} final_valid {:.3f} {:.3f}'.format( FLAGS.model, dataset, scores[2], scores[3] ) )
    print( '{} {} final_test  {:.3f} {:.3f}'.format( FLAGS.model, dataset, scores[4], scores[5] ) )

def main( argv ):
    if FLAGS.dataset == 'all':
        datasets = [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]
    else:
        datasets = [ FLAGS.dataset ]

    for _dataset in datasets:
        start_t = time.time()
        result = []
        if FLAGS.randomsplit > 0:
            for data_seed in range( FLAGS.seed, FLAGS.seed+FLAGS.randomsplit ):
                for i in range( FLAGS.repeat ):
                    result.append( exp( i, _dataset, data_seed=data_seed ) )
        else:
            for i in range( FLAGS.repeat ):
                result.append( exp(i, _dataset) )

        ofilename = "{}_{}_lr{}_drop{}_reg{}_hidden{}_early{}".format(
                    FLAGS.model, _dataset, FLAGS.learning_rate, FLAGS.dropout,
                    FLAGS.weight_decay, FLAGS.hidden1, FLAGS.early_stop )
        if 'fisher' in FLAGS.model:
            ofilename += "_rank{}_perturb{}_noise{}_freq{}_adv{}".format(
                          FLAGS.fisher_rank, FLAGS.fisher_perturbation,
                          FLAGS.fisher_noise, FLAGS.fisher_freq, FLAGS.fisher_adversary )
        analyse( _dataset, result, ofilename )

        print( 'finished in {:.2f} hours'.format( (time.time()-start_t)/3600 ) )

if __name__ == '__main__':
    app.run( main )
