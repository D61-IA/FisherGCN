#!/usr/bin/env python

import sys, os, re
import numpy as np

def output( model, data, key='' ):
    if not (data,model) in result: return
    query = result[(data,model)]

    query = [ r for r in query if key in r[0] ]
    query = sorted( query, key=lambda r:r[2], reverse=True )

    if len(query) > 0:
        filename, valid_acc, test_acc, test_acc_std, test_c = query[0]
        print( '{:.2f}\pm{:.1f} {:.3f} val{:.2f} {} ({} runs)'.format( test_acc, test_acc_std, test_c, valid_acc, filename, len(query) ) )

pattern_v = re.compile( 'final_valid\s+\d+' )
pattern_t = re.compile( 'final_test\s+\d+' )

if len( sys.argv )>1:
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
        if not os.access( fullname, os.R_OK ): continue

        for line in open( fullname ):
            if pattern_v.search( line ):
                try:
                    data = line.split()[1]
                    model = line.split()[0]
                    _val = float( line.split()[5] )
                    if (data,model) in result:
                        result[(data,model)].append( [ filename, _val ] )
                    else:
                        result[(data,model)] = [ [ filename, _val ] ]
                except ValueError:
                    print( 'error parsing', filename )
                    break

            elif pattern_t.search( line ):
                try:
                    data = line.split()[1]
                    model = line.split()[0]
                    _test    = float( line.split()[5] )
                    _teststd = float( line.split()[6] )
                    _test_c  = float( line.split()[3] )
                    result[(data,model)][-1] += [ _test, _teststd, _test_c ]
                except ValueError:
                    print( 'error parsing', filename )
                    break

for data in ['cora', 'citeseer', 'pubmed']:
    for model in ['gcn', 'fishergcn', 'gcnT', 'fishergcnT']:
        if 'fisher' in model:
            output( model, data, 'freq0' )
            output( model, data, 'freq1' )
            output( model, data, 'freq2' )
            #output( model, data )
        else:
            output( model, data )

    print( '' )

