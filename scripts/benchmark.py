#!/usr/bin/env python

import subprocess, shlex
import sys, itertools, re, time, argparse

MODELS   = [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]
DATAS    = [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]

LRATES        = [ 0.003, 0.01, 0.03 ]
LRATES_SUBSET = [ 0.01 ]
DROPOUTS      = [ 0.5 ]
REGS          = [ 0.002, 0.001, 0.0005 ]
REGS_SUBSET   = [ 0.001 ]
HIDDEN        = [ 16, 32, 64 ]
HIDDEN_SUBSET = [ 64 ]
FISHER_NOISE  = [ 0.03, 0.1, 0.3, 1.0 ]
FISHER_NOISE_SUBSET = [ 0.1 ]
FISHER_RANK   = [ 10 ]

def construct_config_arr( data, model ):
    if data in [ 'cora', 'citeseer' ]:
        if 'fisher' in model:
            _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN, FISHER_NOISE, FISHER_RANK )
        else:
            _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN )

    else: # use some default parameters for large datasets
        if 'fisher' in model:
            _config_arr = itertools.product( LRATES_SUBSET, DROPOUTS, REGS_SUBSET, HIDDEN_SUBSET, FISHER_NOISE_SUBSET, FISHER_RANK )
        else:
            _config_arr = itertools.product( LRATES_SUBSET, DROPOUTS, REGS_SUBSET, HIDDEN_SUBSET )

    return list(_config_arr)

def run_single( ds, model, config, early, epochs, split, repeat, seed, save ):
    pattern_v = re.compile( 'final_valid\s+([\d\.]+)\s+([\d\.]+)' )
    pattern_t = re.compile( 'final_test\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)' )

    if 'fisher' in model:
        lr, dropout, reg, hidden, fisher_noise, fisher_rank = config
    else:
        lr, dropout, reg, hidden = config

    cmd = 'python gcn/train.py --dataset {} --model {} --learning_rate {} --dropout {}' \
          ' --weight_decay {} --hidden1 {} --early_stop {} --epochs {}' \
          ' --randomsplit {} --repeat {} --seed {}'.format( ds, model, lr, dropout, reg, hidden, early, epochs, split, repeat, seed )
    if 'fisher' in model:
        cmd += ' --fisher_noise {} --fisher_rank {}'.format( fisher_noise, fisher_rank )
    if save: cmd += ' --save'

    out = subprocess.check_output( shlex.split(cmd), stderr=subprocess.STDOUT ).decode( 'utf-8' )

    val_loss,  val_acc  = pattern_v.search( out ).groups()
    test_loss, test_acc, test_loss_std, test_acc_std = pattern_t.search( out ).groups()
    return val_loss, val_acc, test_loss, test_loss_std, test_acc, test_acc_std

def print_result( data, model, config, loss_and_acc, start_t ):
    print( '{:30s} {:15s}'.format( data, model ), end=" " )
    print( 'lr={:<5.3g} drop={:<3.1g} reg={:<6.4g} hidden={:<3d}'.format( *config ), end=" " )
    print( 'val {:6s} {:6s} test {:6s}-{:6s} {:6s}-{:6s}'.format( *loss_and_acc ), end=" " )
    print( '({:.2f}h)'.format( (time.time()-start_t)/3600 ) )

    if 'fisher' in model:
        print( '{:30s} {:15s}'.format( data, model ), end=" " )
        print( 'noise={:<5.2g} rank={:<3d}'.format( config[4], config[5] ) )

def run_repeated_exp( data, model, randomsplit, repeat, seed ):
    config_arr = construct_config_arr( data, model )
    if len( config_arr ) <= 1:
        best_config = config_arr[0]

    else:
        # "light" runs to select the best configuration;
        # repeat 2 time on 20 random splits for small number of iterations without early stopping

        if randomsplit > 0:
            LIGHT_NUM_SPLITS = 5
            LIGHT_REPEAT = 5
        else:
            LIGHT_NUM_SPLITS = 0
            LIGHT_REPEAT = 5

        results = []
        for config in config_arr:
            _tick = time.time()
            loss_and_acc = run_single( data, model, config, 0, 100, LIGHT_NUM_SPLITS, LIGHT_REPEAT, seed=seed, save=False )
            results.append( ( loss_and_acc[4], config ) )

            print_result( data, model, config, loss_and_acc, _tick )
            sys.stdout.flush()

        results.sort( key=lambda x: x[0], reverse=True )
        best_config = results[0][1]

    # re-run the best config 5 times using 20 different random splits
    start_t = time.time()
    loss_and_acc = run_single( data, model, best_config, 1, 500, randomsplit, repeat, seed=seed, save=True )
    print( "best", end=" " )
    print_result( data, model, best_config, loss_and_acc, start_t )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'dataset', choices=DATAS+['all'] )
    parser.add_argument( 'model',   choices=MODELS+['all'] )
    parser.add_argument( '--randomsplit', type=int, default=20,   help="#random splits" )
    parser.add_argument( '--repeat',      type=int, default=5,    help="#repeats" )
    parser.add_argument( '--seed',        type=int, default=2019, help="random seed" )
    args = parser.parse_args()

    if args.dataset == 'all':
        data_arr = DATAS
    else:
        data_arr = [ args.dataset ]

    if args.model == 'all':
        model_arr = MODELS
    else:
        model_arr = [ args.model ]

    for data in data_arr:
        for model in model_arr:
            run_repeated_exp( data, model, args.randomsplit, args.repeat, args.seed )

if __name__ == '__main__':
    main()
