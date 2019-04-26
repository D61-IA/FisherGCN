#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import itertools

def output( query, key='' ):
    query = [ r for r in query if key in r[0] ]
    query = sorted( query, key=lambda r:r[1][5], reverse=True )

    if len(query) > 0:
        filename, scores, std = query[0]
        print( '{:.2f}\pm{:.1f} {:.3f}\pm{:.2f} {} ({} runs)'.format( scores[5], std[5], scores[4], std[4], filename, len(query) ) )

        raw = np.load( filename )
        lcurves = raw['lcurves']
    else:
        lcurves = None

    return lcurves

def main():
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

            model, data = filename.split('_')[:2]
            if (data,model) in result:
                result[(data,model)].append( ( fullname, scores, std ) )
            else:
                result[(data,model)] = [ ( fullname, scores, std ) ]

    for data in [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]:
        lcurves = []
        linestyles = ['--', '-.', '-', ':']
        markers = ['o', 'v', 's', '*']

        for model in [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]:
            if not (data,model) in result: continue
            lcurves.append( ( model, output( result[(data,model)] ) ) )
        if len( lcurves ) == 0 : continue

        figname = data + '.pdf'
        fig, ax = plt.subplots( 2,1,figsize=(12,10) )

        styles = list(itertools.product(markers, linestyles))
        shuffle(styles)
        styles = styles[:len(lcurves)]
        for (model, lc), style in zip(lcurves, styles):
            lc_mean = np.mean( lc, axis=0 )
            lc_std = np.std( lc, axis=0 )
            ax[0].errorbar( range(lc_mean.shape[0]), lc_mean[:,1], lc_std[:,1],
                    fmt=style[0], ls=style[1], markevery=5, errorevery=5,
                    label='Training {}'.format(model) )
            ax[0].errorbar( range(lc_mean.shape[0]), lc_mean[:,3], lc_std[:,3],
                    fmt=style[0], ls=style[1], markevery=5, errorevery=5,
                    label='validation {}'.format(model) )
            ax[1].errorbar( range(lc_mean.shape[0]), lc_mean[:,0], lc_std[:,0],
                    fmt=style[0], ls=style[1], markevery=5, errorevery=5,
                    label='Training {}'.format(model) )
            ax[1].errorbar( range(lc_mean.shape[0]), lc_mean[:,2], lc_std[:,2],
                    fmt=style[0], ls=style[1], markevery=5, errorevery=5,
                    label='validation {}'.format(model) )
        ax[0].set_xlabel( 'Epoch' )
        ax[0].set_ylabel( 'Accuracy' )
        ax[0].set_title( 'Learning Curves (Accuracy)' )
        ax[0].grid()
        ax[0].legend()
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Learning Curves (Loss) ')
        ax[1].grid()
        ax[1].legend()
        fig.savefig( figname )


        print( '' )

if __name__ == '__main__':
    main()
