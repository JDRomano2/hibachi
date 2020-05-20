"""
Tools for handling file input and output.
"""

import argparse

def parse_args():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run hibachi to generate a dataset with feature interactions."
    )

    parser.add_argument('-e', '--evaluation', type=str,
            help='name of evaluation [normal|folds|subsets|noise|oddsratio]' +
                 ' (default=normal) note: oddsratio sets columns == 10')
    parser.add_argument('-f', '--file', type=str,
            help='name of training data file (REQ)' +
                 ' filename of random will create all data')
    parser.add_argument("-g", "--generations", type=int,
            help="number of generations (default=40)")
    parser.add_argument("-i", "--information_gain", type=int,
            help="information gain 2 way or 3 way (default=2)")
    parser.add_argument("-m", "--model_file", type=str,
            help="model file to use to create Class from; otherwise \
                  analyze data for new model.  Other options available \
                  when using -m: [f,o,s,P]")
    parser.add_argument('-o', '--outdir', type=str,
            help='name of output directory (default = .)' +
            ' Note: the directory will be created if it does not exist')
    parser.add_argument("-p", "--population", type=int,
            help="size of population (default=100)")
    parser.add_argument("-r", "--random_data_files", type=int,
            help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--seed", type=int,
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-A", "--showallfitnesses",
            help="show all fitnesses in a multi objective optimization",
            action='store_true')
    parser.add_argument("-C", "--columns", type=int,
            help="random data columns (default=3) note: " +
                 "evaluation of oddsratio sets columns to 10")
    parser.add_argument("-F", "--fitness",
            help="plot fitness results", action='store_true')
    parser.add_argument("-P", "--percent", type=int,
            help="percentage of case for case/control (default=25)")
    parser.add_argument("-R", "--rows", type=int,
            help="random data rows (default=1000)")
    parser.add_argument("-S", "--statistics",
            help="plot statistics",action='store_true')
    parser.add_argument("-T", "--trees",
            help="plot best individual trees", action='store_true')

    args = parser.parse_args()
    options = vars(args)

    return options