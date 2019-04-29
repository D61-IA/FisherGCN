#!/usr/bin/env python

import subprocess, shlex
import sys, itertools, re, time

MODELS   = [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]
DATAS    = [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]

LRATES   = [ 0.003, 0.01 ]
DROPOUTS = [ 0.5 ]
REGS     = [ 0.002, 0.001, 0.0005 ]
HIDDEN   = [ 32, 64 ]
EARLY    = [ 1, 2 ]
EPOCHS   = [ 500 ]

FISHER_NOISE = [ 0.1, 0.3, 1.0, 3.0 ]
FISHER_RANK  = [ 10 ]

def run( ds, model, config, repeat ):
    pattern_v = re.compile( 'final_valid\s+([\d\.]+)\s+([\d\.]+)' )
    pattern_t = re.compile( 'final_test\s+([\d\.]+)\s+([\d\.]+)' )

    if 'fisher' in model:
        lr, dropout, reg, hidden, early, epochs, fisher_noise, fisher_rank = config
    else:
        lr, dropout, reg, hidden, early, epochs = config

    cmd = './train.py --dataset {} --model {} --learning_rate {} --dropout {}' \
          ' --weight_decay {} --hidden1 {} --early_stop {} --epochs {}' \
          ' --repeat {} --randomsplit 1'.format( ds, model, lr, dropout, reg, hidden, early, epochs, repeat )
    if 'fisher' in model:
        cmd += ' --fisher_noise {} --fisher_rank {}'.format( fisher_noise, fisher_rank )

    out = subprocess.check_output( shlex.split(cmd), stderr=subprocess.STDOUT ).decode( 'utf-8' )

    val_loss,  val_acc  = pattern_v.search( out ).groups()
    test_loss, test_acc = pattern_t.search( out ).groups()
    return val_loss, val_acc, test_loss, test_acc

def main( data, model ):
    start_t = time.time()

    if 'fisher' in model:
        _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN, EARLY, EPOCHS, FISHER_NOISE, FISHER_RANK )
    else:
        _config_arr = itertools.product( LRATES, DROPOUTS, REGS, HIDDEN, EARLY, EPOCHS )

    # run 2 time per 10 splits
    results = []
    for config in _config_arr:
        print( '{} {}'.format( data, model ), config, end=" " )
        loss_and_acc = run( data, model, config, 2 )
        print( loss_and_acc )
        results.append( ( loss_and_acc[1], config ) )
    results.sort( key=lambda x: x[0], reverse=True )

    # re-run the best config 10 times per split
    best_config = results[0][1]
    print( 'best config:', best_config, end=" " )
    loss_and_acc = run( data, model, best_config, 10 )
    print( loss_and_acc )

    _time_cost = time.time() - start_t
    print( 'finished in {:.1f} hours'.format( _time_cost/3600 ) )
    sys.stdout.flush()

if __name__ == '__main__':
    if len( sys.argv ) < 3:
        print( 'usage: {} [cora|citeseer|pubmed|amazon_electronics_computers|amazon_electronics_photo] [gcn|fishergcn|gcnT|fishergcnT]'.format( sys.argv[0] ) )
    else:
        assert( sys.argv[1] in DATAS )
        assert( sys.argv[2] in MODELS )
        main( sys.argv[1], sys.argv[2] )

