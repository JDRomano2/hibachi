#!/usr/bin/env python
#==============================================================================
#
#          FILE:  IO.py
# 
#         USAGE:  import IO (from hib.py)
# 
#   DESCRIPTION:  graphing and file i/o routines.  
# 
#       UPDATES:  170213: added subset() function
#                 170214: added getfolds() function
#                 170215: added record shuffle to getfolds() function
#                 170216: added addnoise() function
#                 170217: modified create_file() to name file uniquely
#                 170302: added plot_hist() to plot std
#                 170313: added get_arguments()
#                 170319: added addone()
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.1
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Mon Mar 20 09:57:07 EDT 2017
#==============================================================================
import pandas as pd
import csv
import numpy as np
import argparse
import sys
import os
###############################################################################
def get_arguments():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run hibachi evaluations on your data")

    parser.add_argument('-e', '--evaluation', type=str,
            help='name of evaluation [normal|folds|subsets|noise]' +
                 ' (default=normal)')
    parser.add_argument('-f', '--file', type=str,
            help='name of training data file (REQ)' +
                 ' filename of random will create all data')
    parser.add_argument("-g", "--generations", type=int, 
            help="number of generations (default=40)")
    parser.add_argument("-i", "--information_gain", type=int, 
            help="information gain 2 way or 3 way (default=2)")
    parser.add_argument("-p", "--population", type=int, 
            help="size of population (default=100)")
    parser.add_argument("-r", "--random_data_files", type=int, 
            help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--seed", type=int, 
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-R", "--rows", type=int, 
            help="random data rows (default=1000)")
    parser.add_argument("-C", "--columns", type=int, 
            help="random data columns (default=3)")
    parser.add_argument("-S", "--statistics", 
            help="plot statistics",action='store_true')
    parser.add_argument("-T", "--trees", 
            help="plot best individual trees",action='store_true')
    parser.add_argument("-F", "--fitness", 
            help="plot fitness results",action='store_true')


    args = parser.parse_args()

    if(args.file == None):
        print('filename required')
        sys.exit()
    else:
        options['file'] = args.file
        options['basename'] = os.path.basename(args.file)
        options['dir_path'] = os.path.dirname(args.file)

    if(args.seed == None):
        options['seed'] = -999
    else:
        options['seed'] = args.seed

    if(args.population == None):
        options['population'] = 100
    else:
        options['population'] = args.population

    if(args.information_gain == None):
        options['information_gain'] = 2
    else:
        options['information_gain'] = args.information_gain

    if(args.random_data_files == None):
        options['random_data_files'] = 0
    else:
        options['random_data_files'] = args.random_data_files

    if(args.generations == None):
        options['generations'] = 40
    else:
        options['generations'] = args.generations

    if(args.evaluation == None):
        options['evaluation'] = 'normal'
    else:
        options['evaluation'] = args.evaluation

    if(args.rows == None):
        options['rows'] = 1000
    else:
        options['rows'] = args.rows

    if(args.columns == None):
        options['columns'] = 3
    else:
        options['columns'] = args.columns

    if(args.statistics):
        options['statistics'] = True
    else:
        options['statistics'] = False

    if(args.trees):
        options['trees'] = True
    else:
        options['trees'] = False

    if(args.fitness):
        options['fitness'] = True
    else:
        options['fitness'] = False

    return options
###############################################################################
def get_random_data(rows, cols):
    data = np.random.randint(0,3,size=(rows,cols))
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def create_file(data,label,outfile):
    """ append label as column to data then write out file with header """
    for i in range(len(data)):
        data[i].append(label[i])
        
    # create header
    header = []
    for i in range(len(data[0])-1):
        header.append('X' + str(i))
    header.append('Class')
        
    # print data to results.tsv
    datadf = pd.DataFrame(data,columns=header) # convert to DF
    datadf.to_csv(outfile, sep='\t', index=False)
###############################################################################
def read_file(fname):
    """ UNUSED: return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    with open(fname) as data:
        dataReader = csv.reader(data, delimiter='\t')
        data = list(list(int(elem) for elem in row) for row in dataReader)
    #transpose into x
    inst_length = len(data[0])
    x = []
    for i in range(inst_length):
        y = [col[i] for col in data]
        x.append(y)
    del y
    return data, x
###############################################################################
def read_file_np(fname):
    """ return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    data = np.genfromtxt(fname, dtype=np.int, delimiter='\t') 
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def addone(d):
    a = np.array(d)
    a += 1
    return a.tolist()
###############################################################################
def xpose(d):
    """ transpose list of lists """
    x = []
    for i in range(len(d[0])):
        y = [col[i] for col in d]
        x.append(y)
    return x
###############################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    import sys
    sys.stdout.write(format % args)
    sys.stdout.flush()