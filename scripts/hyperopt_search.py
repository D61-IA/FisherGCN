#!/usr/bin/env python

import sys, re, time, argparse
import numpy as np
import subprocess, shlex
import pickle

from pathlib import Path
FISHERGCN_PATH = Path(__file__).resolve().parents[1]

from hyperopt import hp, fmin, tpe, Trials
EARLY  = 1
EPOCHS = 300
NSPLIT = 10
REPEAT = 10
NEVALS = 64
MAGIC = 123
NCPUS  = 8

SPACE = {
    'gcn': {
           'model': 'gcn',
           'lrate': hp.choice( 'lrate', (0.01,) ),
           'dropout': hp.choice( 'dropout', (0.5,0.6,0.7,0.8) ),
           'weight_decay': hp.choice( 'weight_decay', (5e-4,1e-3,2e-3) ),
           'hidden': hp.choice( 'hidden', (64,) ),
    },
    'fishergcn': {
           'model': 'fishergcn',
           'lrate': hp.choice( 'lrate', (0.01,) ),
           'dropout': hp.choice( 'dropout', (0.5,0.6,0.7) ),
           'weight_decay': hp.choice( 'weight_decay', (5e-4,1e-3) ),
           'hidden': hp.choice( 'hidden', (64,) ),
           'fisher_noise': hp.loguniform( 'fisher_noise', np.log(1e-3), np.log(0.3) ),
           'fisher_rank': hp.choice( 'fisher_rank', (10,30,50,) ),
    }
}

def run_exp( dataset, config ):
    cmd_path = FISHERGCN_PATH.joinpath( "gcn/train.py" )

    cmd = 'python {} --dataset {} --model {} --lrate {} --dropout {}' \
          ' --weight_decay {} --hidden {} --early_stop {} --epochs {}' \
          ' --randomsplit {} --repeat {} --seed {}'.format(
          cmd_path, dataset, config['model'],
          config['lrate'], config['dropout'],
          config['weight_decay'], config['hidden'],
          EARLY, EPOCHS, NSPLIT, REPEAT, MAGIC )

    if 'fisher' in config['model'].lower():
        cmd += ' --fisher_noise {} --fisher_rank {}'.format(
                config['fisher_noise'], config['fisher_rank'] )

    try:
        out = subprocess.check_output( shlex.split(cmd), stderr=subprocess.STDOUT ).decode( 'utf-8' )
    except subprocess.CalledProcessError as e:
        print( '[stdout]' )
        print( e.output.decode('utf-8') )
        sys.exit(1)

    pattern_t = re.compile(  'final_test\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)' )
    test_loss,  test_loss_std,  test_acc,  test_acc_std  = pattern_t.search( out ).groups()
    return float(test_acc)

def benchmark( config ):
    _acc = 0
    _acc += run_exp( 'cora',     config )
    _acc += run_exp( 'citeseer', config )
    _acc /= 2
    return -_acc

def benchmark_tune( config, reporter ):
    _acc = -benchmark( config )
    reporter( accuracy=_acc )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'model',   choices=('gcn','fishergcn'), help="model to tune" )
    parser.add_argument( '--tune',  action='store_true',         help="using ray.tune" )
    args = parser.parse_args()

    start_t = time.time()
    if args.tune:
        import ray
        from ray.tune import run
        from ray.tune.suggest.hyperopt import HyperOptSearch

        ray.init()
        algo = HyperOptSearch( SPACE[args.model], max_concurrent=NCPUS, reward_attr="accuracy" )
        run( benchmark_tune, search_alg=algo, num_samples=NEVALS )

    else:
        trials=Trials()
        best = fmin( benchmark, space=SPACE[args.model], algo=tpe.suggest, max_evals=NEVALS, trials=trials )
        print( 'best:', best )

        with open("{}.pkl".format(args.model), "wb") as f:
            pickle.dump( trials, f )

    print( "finished in {:.1f} hours".format( (time.time()-start_t)/3600 ) )

if __name__ == '__main__':
    main()
