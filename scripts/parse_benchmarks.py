#!/usr/bin/env python

from benchmark_grid import DATAS, MODELS
from pathlib import Path
import sys, os, re, math, itertools

def print_result( result ):
    for data, model in itertools.product( DATAS, MODELS ):
        if not (data, model) in result: continue
        if len(result[(data,model)])==0: continue

        loss_m  = 0
        loss_m2 = 0
        acc_m   = 0
        acc_m2  = 0
        count   = 0

        for w,a,b,c,d in result[ (data,model) ]:
            loss_m  += a * w
            loss_m2 += ( a**2 + b**2 ) * w
            acc_m   += c * w
            acc_m2  += ( c**2 + d**2 ) * w
            count   += w

        loss_m  /= count
        loss_m2 /= count
        loss_m2 -= loss_m**2

        acc_m  /= count
        acc_m2 /= count
        acc_m2 -= acc_m**2

        print( '{:30s} {:20s} {:3d}runs {:6.2f}\pm{:<6.2f} {:6.2f}\pm{:<6.1f}'.format(
                data, model, count, loss_m, math.sqrt(loss_m2), acc_m, math.sqrt(acc_m2) ) )
    print()

def main():
    re_runs = re.compile( r'''best\s+
                              (\d+)x(\d+)\s
                           ''', re.X )
    re_test = re.compile( r'''test\s+
                              ([\d\.]+)\s+([\d\.]+)
                              \s+
                              ([\d\.]+)\s+([\d\.]+)
                           ''', re.X )

    if len( sys.argv ) > 1:
        root = sys.argv[1]
    else:
        root = "."
    root_path = Path( root )

    result = {}
    for dirs, subdirs, files in os.walk( root ):
        for filename in files:
            if not ( filename.endswith('.out') or filename.endswith('.log') ): continue
            for line in open( root_path.joinpath(filename) ):
                line = line.strip()
                if line.startswith( 'best' ):
                    _repeat1, _repeat2 = re_runs.search( line ).groups()
                    runs = max( int(_repeat1) * int(_repeat2), 1 )
                    data   = line.split()[2]
                    model  = line.split()[3]
                    loss_and_acc = re_test.search( line ).groups()
                    loss_and_acc = [ float(s) for s in loss_and_acc ]

                    if not (data,model) in result: result[(data,model)] = []
                    result[(data,model)].append( [runs,] + loss_and_acc )

    print_result( result )

if __name__ == '__main__':
    main()
