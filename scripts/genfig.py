#!/usr/bin/env python

import sys, os
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['text.usetex']  = True
import matplotlib.pyplot as plt

'''
npz files -> pdf plot of learning curves
'''

MODEL_NAMES = {
    'gcn'       : 'GCN',
    'fishergcn' : 'FisherGCN',
    'gcnT'      : 'GCN$^T$',
    'fishergcnT': 'FisherGCN$^T$',
}

DATA_NAMES = {
    'cora'    : 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed'  : 'PubMed',
}


def print_result( data, model, results ):
    scores = np.array( [ r[1] for r in results ] )

    mean   = np.mean( scores, axis=0 )
    #median = np.median( scores, axis=0 )
    print( f'{data:10s} {model:10s} loss={mean[4]:.3f} accuracy={mean[5]:.3f}' )

def extract_lcurve( results, key ):
    lcurves = []
    for r in results:
        raw = np.load( r[0] )
        lcurves.append( raw[key] )

    epochs = min( [ c.shape[1] for c in lcurves ] )
    lcurves = np.vstack( [ c[:,:epochs] for c in lcurves ] )
    return lcurves

def main():
    # usage: python genfig.py path [keys...]
    #
    # path is the folder containing the npz files
    # keys is a list of keywords which should appear in the npz filename
    if len( sys.argv ) > 1:
        root = sys.argv[1]
    else:
        root = "."

    if len( sys.argv ) > 2:
        keys = sys.argv[2:]
    else:
        keys = ['']

    result = {}

    for dirs, subdirs, files in os.walk( root ):
        for filename in files:
            haskey = True
            for key in keys:
                if not key in filename: haskey = False
            if not haskey: continue

            fullname = os.path.join( root, filename )
            if not fullname.endswith( 'npz' ): continue
            if not os.access( fullname, os.R_OK ): continue

            raw = np.load( fullname )
            scores = raw['scores']
            std    = raw['std']

            model = filename.split('_')[0]
            data  = filename.split('_')[1]
            if data == 'amazon' or data == 'ms':
                data += '_' + filename.split('_')[2]
                data += '_' + filename.split('_')[3]

            if (data,model) in result:
                result[(data,model)].append( ( fullname, scores, std, ) )
            else:
                result[(data,model)] = [ ( fullname, scores, std, ) ]


    for data in [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo', 'ms_academic_cs' ]:
        lcurves = []

        for model in [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]:
            if not (data,model) in result: continue
            print_result( data, model, result[data,model] )
            lcurves.append( ( model, extract_lcurve( result[data,model], 'train_loss' ),
                                     extract_lcurve( result[data,model], 'valid_acc' ) ) )

        if len(lcurves) == 0: continue

        colors     = [ 'r', 'tab:blue', 'darkorange', 'purple' ]
        markers    = [ 'o', 'v', 's', '*' ]

        fig, ax1 = plt.subplots( 1,1,figsize=(8,6) )
        ax2 = ax1.twinx()
        step = 0
        for (model, tlc, vlc), c, mk in zip( lcurves, colors, markers ):
            tlc_mean = np.mean( tlc, axis=0 )
            tlc_std  = np.std( tlc, axis=0 )
            vlc_mean = np.mean( vlc, axis=0 )
            vlc_std  = np.std( vlc, axis=0 )
            x = np.arange( tlc_mean.shape[0] ) + step
            step += 0.8

            ax1.errorbar( x, tlc_mean, tlc_std,
                          color=c, ls='-', lw=2, marker=mk, label='{}'.format( MODEL_NAMES[model] ),
                          markevery=5, errorevery=5, capsize=1, alpha=0.5, elinewidth=1 )

            ax2.errorbar( x, vlc_mean, vlc_std,
                          color=c, ls='--', lw=2, marker=mk, label='{}'.format( MODEL_NAMES[model] ),
                          markevery=5, errorevery=5, capsize=1, alpha=0.5, elinewidth=1 )

        ax1.set_xlim( 0, 100 )
        ax1.set_ylim( 0.3, 1.0 )
        ax2.set_ylim( 55, 80 )

        ax1.set_xlabel( 'Epoch' )
        ax1.set_ylabel( 'Training Loss' )
        ax2.set_ylabel( 'Testing Accuracy' )
        ax1.set_title( DATA_NAMES[data] if data in DATA_NAMES else data.replace('_','') )
        ax1.grid()
        ax1.legend( loc=5 )

        figname = data + '.svg'
        fig.savefig( figname, bbox_inches='tight', transparent=True, pad_inches=0 )

if __name__ == '__main__':
    main()
