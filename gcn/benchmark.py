#!/usr/bin/env python

import subprocess, shlex
import sys, itertools, re, time

MODELS   = [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]
DATAS    = [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]

LRATES   = [ 0.003, 0.01 ]
DROPOUTS = [ 0.5 ]
REGS     = [ 0.002, 0.001, 0.0005 ]
HIDDEN   = [ 32, 64 ]
FISHER_NOISE = [ 0.1, 0.3, 1.0, 3.0 ]
FISHER_RANK  = [ 10 ]

def run( ds, model, config, early, epochs, split, repeat, save ):
    pattern_v = re.compile( 'final_valid\s+([\d\.]+)\s+([\d\.]+)' )
    pattern_t = re.compile( 'final_test\s+([\d\.]+)\s+([\d\.]+)' )

    if 'fisher' in model:
        lr, dropout, reg, hidden, fisher_noise, fisher_rank = config
    else:
        lr, dropout, reg, hidden = config

    cmd = 'python train.py --dataset {} --model {} --learning_rate {} --dropout {}' \
          ' --weight_decay {} --hidden1 {} --early_stop {} --epochs {}' \
          ' --randomsplit {} --repeat {}'.format( ds, model, lr, dropout, reg, hidden, early, epochs, split, repeat )
    if 'fisher' in model:
        cmd += ' --fisher_noise {} --fisher_rank {}'.format( fisher_noise, fisher_rank )
    if save: cmd += ' --save'

    out = subprocess.check_output( shlex.split(cmd), stderr=subprocess.STDOUT ).decode( 'utf-8' )

    val_loss,  val_acc  = pattern_v.search( out ).groups()
    test_loss, test_acc = pattern_t.search( out ).groups()
    return val_loss, val_acc, test_loss, test_acc

def run_exp( data, model ):
    start_t = time.time()

    # "light" runs to select the best configuration;
    # repeat 5 time on 5 random splits for small number of iterations without early stopping
    if 'fisher' in model:
        _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN, FISHER_NOISE, FISHER_RANK )
    else:
        _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN )

    results = []
    for config in _config_arr:
        _tick = time.time()
        print( '{} {}'.format( data, model ), config, end=" " )
        loss_and_acc = run( data, model, config, 0, 100, 5, 5, False )
        print( loss_and_acc, '({:.1f}m)'.format( (time.time()-_tick)/60 ) )
        results.append( ( loss_and_acc[3], config ) )
    results.sort( key=lambda x: x[0], reverse=True )

    # re-run the best config 5 times using 20 different random splits
    best_config = results[0][1]
    print( 'best {} {}'.format( data, model ), best_config, end=" " )
    loss_and_acc = run( data, model, best_config, 1, 500, 20, 5, True )
    print( loss_and_acc )

    _time_cost = time.time() - start_t
    print( 'finished in {:.1f} hours'.format( _time_cost/3600 ) )
    sys.stdout.flush()

def main():
    if len( sys.argv ) < 3:
        print( 'usage: {} [cora|citeseer|pubmed|amazon_electronics_computers|amazon_electronics_photo] [gcn|fishergcn|gcnT|fishergcnT]'.format( sys.argv[0] ) )
    else:
        if sys.argv[1] == 'all':
            data_arr = DATAS
        else:
            assert( sys.argv[1] in DATAS )
            data_arr = [ sys.argv[1] ]

        if sys.argv[2] == 'all':
            model_arr = MODELS
        else:
            assert( sys.argv[2] in MODELS )
            model_arr = [ sys.argv[2] ]

        for data in data_arr:
            for model in model_arr:
                run_exp( data, model )

if __name__ == '__main__':
    main()
