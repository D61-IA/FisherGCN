#!/usr/bin/env python

import numpy as np
import sys, os, re

def print_result( result ):
    DATAS    = [ 'cora', 'citeseer', 'pubmed', 'amazon_electronics_computers', 'amazon_electronics_photo' ]
    MODELS   = [ 'gcn', 'fishergcn', 'gcnT', 'fishergcnT' ]

    for data in DATAS:
        for model in MODELS:
            if not (data, model) in result: continue
            if len(result[(data,model)])==0: continue

            r = np.array( result[(data,model)] )
            #print( data, model , r )
            avg_loss, avg_acc, avg_std = r.mean(0)
            print( '{:30s} {:20s} {:6.4g} {:6.4g}'.format( data, model, avg_loss, avg_acc ) )
        print()

def main():
    re_test = re.compile( r'''test\s+
                              ([\d\.]+)\s*-([\d\.]+)
                              \s+
                              ([\d\.]+)\s*-([\d\.]+)
                           ''', re.X )

    if len( sys.argv ) > 1:
        root = sys.argv[1]
    else:
        root = "."

    result = {}
    for dirs, subdirs, files in os.walk( root ):
        for filename in files:
            for line in open( filename ):
                if line.startswith( 'best' ):
                    data   = line.split()[1]
                    model  = line.split()[2]
                    test_loss, _, test_acc, test_acc_std = re_test.search( line ).groups()

                    if not (data,model) in result: result[(data,model)] = []
                    result[(data,model)].append( ( float(test_loss), float(test_acc), float(test_acc_std) ) )

    print_result( result )

if __name__ == '__main__':
    main()
