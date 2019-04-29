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

            model = filename.split('_')[0]
            data  = filename.split('_')[1]
            if data == 'amazon':
                data += '_' + filename.split('_')[2]
                data += '_' + filename.split('_')[3]

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
        fig, ax = plt.subplots( 3,1,figsize=(8,16) )

        styles = list(itertools.product(markers, linestyles))
        shuffle(styles)
        styles = styles[:len(lcurves)]

        def axs_errorbar(axs, ylabels, titles):
            for (model, lc), style in zip(lcurves, styles):
                lc_mean = np.mean( lc, axis=0 )
                lc_std = np.std( lc, axis=0 )
                x = range(lc_mean.shape[0])

                for idx, ax in zip([0,1,1], reversed(axs)):
                    ax.errorbar(x, lc_mean[:,idx], lc_std[:,idx],
                                fmt=style[0], ls=style[1],  label='Training {}'.format(model),
                                markevery=5, errorevery=5, capsize=5, alpha=0.5)
                    ax.errorbar(x, lc_mean[:,idx+2], lc_std[:,idx+2],
                                fmt=style[0], ls=style[1], label='Validation {}'.format(model),
                                markevery=5, errorevery=5, capsize=5, alpha=0.5)
            axs[1].set_xlim(0, 40)
            axs[1].set_ylim(60, 100)
            for ax, t, yl in zip(axs, titles, ylabels):
                ax.set_xlabel('Epoch')
                ax.set_ylabel(yl)
                ax.set_title(t)
                ax.grid()
                ax.legend()

        titles = ['Learning Curves (Accuracy)', 'Learning Curves - Zoom in (Accuracy)', 'Learning Curves (Loss)']
        ylabels = ['Accuracy', 'Accuracy', 'Loss']
        axs_errorbar(ax, ylabels, titles)
        fig.savefig( figname )

if __name__ == '__main__':
    main()
