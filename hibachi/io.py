"""
Tools for handling file input and output.
"""

from dataclasses import dataclass

import ipdb

import argparse

@dataclass
class InputOptions:
    """Class for keeping track of parameters passed by the user."""
    infile: str
    model_file: str
    outdir: str
    num_generations: int
    pop_size: int
    case_control_ratio: int
    evaluation: str  # {'normal', 'folds', 'subsets', 'noise', 'oddsratio'}
    rand_file_count: int  # "number of random data to use instead of files" ???
    rand_seed: int  # default rand int in [1,1000]
    num_columns: int  # number of columns if random data are used
    num_rows: int  # number of rows if random data are used
    inf_gain_type: int  # 2-way or 3-way
    show_all_fitness: bool
    plot_fitness: bool
    plot_statistics: bool
    plot_best_trees: bool
    verbose: bool

def parse_args():
    options = dict()

    parser = argparse.ArgumentParser(
        description = "Run hibachi to generate a dataset with feature interactions.",
        prog="HIBACHI"
    )

    parser.add_argument('-e', '--evaluation', type=str,
        help='name of evaluation [normal|folds|subsets|noise|oddsratio]' +
             ' (default=normal) note: oddsratio sets columns == 10')
    parser.add_argument('-f', '--infile', type=str,
        help='name of training data file (REQ)' +
             ' filename of random will create all data')
    parser.add_argument("-g", "--num_generations", type=int,
        help="number of generations (default=40)")
    parser.add_argument("-i", "--inf_gain_type", type=int,
        help="information gain 2 way or 3 way (default=2)")
    parser.add_argument("-m", "--model_file", type=str,
        help="model file to use to create Class from; otherwise \
              analyze data for new model.  Other options available \
              when using -m: [f,o,s,P]")
    parser.add_argument('-o', '--outdir', type=str,
        help='name of output directory (default = .)' +
        ' Note: the directory will be created if it does not exist')
    parser.add_argument("-p", "--pop_size", type=int,
        help="size of population (default=100)")
    parser.add_argument("-r", "--rand_file_count", type=int,
        help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--rand_seed", type=int,
        help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-A", "--show_all_fitness",
        help="show all fitnesses in a multi objective optimization",
        action='store_true')
    parser.add_argument("-C", "--num_columns", type=int,
        help="random data columns (default=3) note: " +
             "evaluation of oddsratio sets columns to 10")
    parser.add_argument("-F", "--plot_fitness",
        help="plot fitness results", action='store_true')
    parser.add_argument("-P", "--case_control_ratio", type=int,
        help="percentage of case for case/control (default=25)")
    parser.add_argument("-R", "--num_rows", type=int,
        help="random data rows (default=1000)")
    parser.add_argument("-S", "--plot_statistics",
        help="plot statistics",action='store_true')
    parser.add_argument("-T", "--plot_best_trees",
        help="plot best individual trees", action='store_true')
    parser.add_argument("-V", "--verbose",
        help="Print more information to screen while running", action='store_true')

    args = parser.parse_args()
    options = vars(args)

    options = InputOptions(
        **vars(args)
    )

    return options