#!/usr/bin/env python

'''
wrapper of train.py

which can do grid search of hyperparameters
'''

import subprocess, shlex
import sys, gc, itertools, re, time, argparse
from pathlib import Path

FISHERGCN_PATH = Path(__file__).resolve().parents[1]
sys.path.append( str(FISHERGCN_PATH.joinpath( "gcn/" )) )
from data_io import PLANETOID_DATA, PITFALL_DATA

MODELS   = [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]
DATAS    = PLANETOID_DATA + PITFALL_DATA

LRATE                = [ 0.001, 0.003, 0.01, 0.03 ]
DROPOUT              = [ 0.5, 0.8 ]
WEIGHT_DECAY         = [ 0.002, 0.001, 0.0005 ]
HIDDEN               = [ 32, 64 ]
FISHER_NOISE         = [ 0.03, 0.1, 0.3, 1.0 ]
FISHER_RANK          = [ 50 ]

def _config_grid( data, model, default_config ):

    _lrate        = LRATE        if default_config[0] is None else ( default_config[0], )
    _dropout      = DROPOUT      if default_config[1] is None else ( default_config[1], )
    _weight_decay = WEIGHT_DECAY if default_config[2] is None else ( default_config[2], )
    _hidden       = HIDDEN       if default_config[3] is None else ( default_config[3], )
    _fisher_noise = FISHER_NOISE if default_config[4] is None else ( default_config[4], )
    _fisher_rank  = FISHER_RANK  if default_config[5] is None else ( default_config[5], )

    if 'fisher' in model:
        _config_arr = itertools.product( _lrate, _dropout, _weight_decay, _hidden, _fisher_noise, _fisher_rank )
    else:
        _config_arr = itertools.product( _lrate, _dropout, _weight_decay, _hidden )
    return list( _config_arr )

def run_single( ds, model, config, early, epochs, split, repeat, data_seed, init_seed, save ):
    pattern_v = re.compile( 'final_valid\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)' )
    pattern_t = re.compile(  'final_test\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)' )

    if 'fisher' in model:
        lr, dropout, reg, hidden, fisher_noise, fisher_rank = config
    else:
        lr, dropout, reg, hidden = config

    cmd_path = FISHERGCN_PATH.joinpath( "gcn/train.py" )

    cmd = 'python {} --dataset {} --model {} --lrate {} --dropout {}' \
          ' --weight_decay {} --hidden1 {} --early_stop {} --epochs {}' \
          ' --randomsplit {} --repeat {} --data_seed {} --init_seed {}'.format(
          cmd_path, ds, model, lr, dropout, reg, hidden, early, epochs,
          split, repeat, data_seed, init_seed )
    if 'fisher' in model:
        cmd += ' --fisher_noise {} --fisher_rank {}'.format( fisher_noise, fisher_rank )
    if save: cmd += ' --save'

    try:
        out = subprocess.check_output( shlex.split(cmd), stderr=subprocess.STDOUT ).decode( 'utf-8' )
    except subprocess.CalledProcessError as e:
        print( '[stdout]' )
        print( e.output.decode('utf-8') )
        sys.exit(1)

    valid_loss, valid_loss_std, valid_acc, valid_acc_std = pattern_v.search( out ).groups()
    test_loss,  test_loss_std,  test_acc,  test_acc_std  = pattern_t.search( out ).groups()
    return valid_loss, valid_acc, \
          test_loss, test_loss_std, test_acc, test_acc_std

def print_result( data, model, config, loss_and_acc, start_t ):
    print( '{:30s} {:15s}'.format( data, model ), end=" " )
    print( 'lr={:<5.3g} drop={:<3.1g} reg={:<6.4g} hidden={:<3d}'.format( *config ), end=" " )
    print( 'val {:6s} {:6s} test {:6s} {:6s} {:6s} {:6s}'.format( *loss_and_acc ), end=" " )
    print( '({:.2f}h)'.format( (time.time()-start_t)/3600 ) )

    if 'fisher' in model:
        print( '{:30s} {:15s}'.format( data, model ), end=" " )
        print( 'noise={:<5.2g} rank={:<3d}'.format( *config[4:] ) )

    sys.stdout.flush()

def run_repeated_exp( data, model, default_config, epochs, early, randomsplit, repeat, data_seed, init_seed ):
    config_arr = _config_grid( data, model, default_config )
    if len( config_arr ) <= 1:
        best_config = config_arr[0]

    else:
        # "light" runs to select the best configuration;
        # use less random splits for a small number of iterations without early stopping

        if randomsplit > 0:
            LIGHT_NUM_SPLITS = 5
            LIGHT_REPEAT = 5
            LIGHT_EPOCHS = 100
        else:
            LIGHT_NUM_SPLITS = 0
            LIGHT_REPEAT = 5
            LIGHT_EPOCHS = 100

        results = []
        for config in config_arr:
            _tick = time.time()
            loss_and_acc = run_single( data, model, config, 0, LIGHT_EPOCHS, LIGHT_NUM_SPLITS, LIGHT_REPEAT, data_seed=data_seed, init_seed=init_seed, save=False )
            gc.collect()

            results.append( ( loss_and_acc[4], config ) )
            print_result( data, model, config, loss_and_acc, _tick )

        results.sort( key=lambda x: x[0], reverse=True )
        best_config = results[0][1]

    # re-run the best config
    start_t = time.time()
    loss_and_acc = run_single( data, model, best_config, early, epochs, randomsplit, repeat, data_seed=data_seed, init_seed=init_seed, save=True )
    print( "best {}x{}".format( randomsplit, repeat ), end=" " )
    print_result( data, model, best_config, loss_and_acc, start_t )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'dataset', choices=DATAS+['all'] )
    parser.add_argument( 'model',   choices=MODELS+['all'] )
    parser.add_argument( '--randomsplit',  type=int,   default=30,   help="#random splits" )
    parser.add_argument( '--repeat',       type=int,   default=10,   help="#repeats" )
    parser.add_argument( '--data_seed',    type=int,   default=2019, help="random seed for data split" )
    parser.add_argument( '--init_seed',    type=int,   default=2019, help="random seed for initialization" )
    parser.add_argument( '--epochs',       type=int,   default=500,  help="#epochs" )
    parser.add_argument( '--early_stop',   type=int,   default=2,    help="early stop strategy" )
    parser.add_argument( '--lrate',        type=float, default=None, help='learning rate' )
    parser.add_argument( '--dropout',      type=float, default=None, help='dropout rate' )
    parser.add_argument( '--weight_decay', type=float, default=None, help='weight decay' )
    parser.add_argument( '--hidden',       type=int,   default=None, help='hidden layer size' )
    parser.add_argument( '--fisher_noise', type=float, default=None, help='noise level' )
    parser.add_argument( '--fisher_rank',  type=int,   default=None, help='rank' )
    args = parser.parse_args()

    if args.dataset == 'all':
        data_arr = DATAS
    else:
        data_arr = [ args.dataset ]

    if args.model == 'all':
        model_arr = MODELS
    else:
        model_arr = [ args.model ]

    default_config = ( args.lrate, args.dropout, args.weight_decay, args.hidden, args.fisher_noise, args.fisher_rank )
    for data, model in itertools.product( data_arr, model_arr ):
        run_repeated_exp( data, model, default_config, args.epochs, args.early_stop, args.randomsplit, args.repeat, args.data_seed, args.init_seed )

if __name__ == '__main__':
    main()
