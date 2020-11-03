"""
Tools for handling file input and output.
"""

from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd


@dataclass
class InputOptions:
    """Class for keeping track of parameters passed by the user."""

    infile: str
    model_file: str
    outdir: str
    num_generations: int
    pop_size: int
    case_control_ratio: int
    mode: str  # {'normal', 'folds', 'subsets', 'noise', 'oddsratio'}
    rand_file_count: int  # "number of random data to use instead of files" ???
    rand_seed: int  # default rand int in [1,1000]
    columns: int  # number of columns if random data are used
    rows: int  # number of rows if random data are used
    inf_gain_type: int  # 2-way or 3-way
    show_all_fitness: bool
    plot_fitness: bool
    plot_statistics: bool
    plot_best_trees: bool
    verbose: bool


def _parse_args():
    options = dict()

    parser = argparse.ArgumentParser(
        prog="Hibachi",
        description="Hibachi is a tool for generating datasets with interactions between the features using genetic programming. As an added benefit, Hibachi's output also includes an interpretable generative model that can be used to either introspect the dataset or to generate more data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-e",
        "--mode",
        type=str,
        default="normal",
        help="Sets the mode for evaluation [normal|folds|subsets|noise|oddsratio]"
        + " (note: oddsratio sets columns == 10",
    )
    parser.add_argument(
        "-f",
        "--infile",
        type=str,
        default="random",
        help="Name of file containing the data used to train Hibachi (if `random`, a random dataset will be generated). Default: 'random'."
    )
    parser.add_argument(
        "-g",
        "--num_generations",
        type=int,
        default=40,
        help="Number of generations over which to run the evolutionary algorithm.",
    )
    parser.add_argument(
        "-i",
        "--inf_gain_type",
        type=int,
        default=2,
        help="Information gain type (2- or 3-way). Options are `2` or `3`.",
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        default=None,
        help="model file to use to create Class from; otherwise \
              analyze data for new model. Other options available \
              when using -m: [f,o,s,P]",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        help="Name of the directory where all output files will be written (the directory will be created if it does not already exist)."
    )
    parser.add_argument(
        "-p", "--pop_size", type=int, default=100, help="Size of the population of individual models to be held during each generation of the algorithm."
    )
    parser.add_argument(
        "-r",
        "--rand_file_count",
        type=int,
        default=0,
        help="Number of random data files (with class labels) to generate using the winning model.",
    )
    parser.add_argument(
        "-s",
        "--rand_seed",
        type=int,
        help="Random seed used in stochastic portions of the Hibachi algorithm (e.g., generating random data).",
    )
    parser.add_argument(
        "-A",
        "--show_all_fitness",
        help="show all fitnesses in a multi objective optimization",
        action="store_true",
    )
    parser.add_argument(
        "-C",
        "--columns",
        type=int,
        default=3,
        help="Number of columns in generated random data (only used when `-f random` is passed). Note: Number of columns is automatically set to 10 when evaluation is 'oddsratio'.",
    )
    parser.add_argument(
        "-R",
        "--rows",
        type=int,
        default=1000,
        help="Number of rows in generated random data (only used when `-f random` is passed).",
    )
    parser.add_argument(
        "-F", "--plot_fitness", help="plot fitness results", action="store_true"
    )
    parser.add_argument(
        "-P",
        "--case_control_ratio",
        type=int,
        default=25,
        help="Percent of rows in the output data that will be labeled as 'case', rather than 'control'.",
    )
    parser.add_argument(
        "-S", "--plot_statistics", help="Enable plotting of summary statistics.", action="store_true"
    )
    parser.add_argument(
        "-T",
        "--plot_best_trees",
        help="Enable plotting tree diagrams for the best individual models.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print more information to screen while running.",
        action="store_true",
    )

    args = parser.parse_args()
    options = vars(args)

    options = InputOptions(**vars(args))

    return options


def create_file(x, result, outfile):
    d = np.array(x).transpose()
    columns = [0]*len(x)
    for i in range(len(x)):
        columns[i] = 'X'+str(i)

    df = pd.DataFrame(d, columns=columns)

    df['Class'] = result
    df.to_csv(outfile, sep='\t', index=False)


def make_random_data(n_rows, n_cols, rseed=None):
    if rseed != None:
        np.random.seed(rseed)
    data = np.random.randint(0, 3, size=(n_rows, n_cols))
    return data


def get_input_data(infile, n_rows=None, n_cols=None, rseed=42):
    if infile == "random":
        if (n_rows is None) or (n_cols is None):
            raise ValueError(
                "Number of rows and columns must be provided when generating random data."
            )
        data = make_random_data(n_rows, n_cols, rseed)
    else:
        data = np.genfromtxt(infile, dtype=np.int, delimiter="\t")

    return data.tolist(), data.transpose().tolist()


def read_model(in_file):
    with open(in_file, 'r') as fp:
        m = fp.read()
        m = m.rstrip()
    return m

def write_model(outfile, best):
    """ write top individual out to model file """
    f = open(outfile, 'w')
    f.write(str(best[0]))
    f.write('\n')
    f.close()

def create_OR_table(best,fitness,seed,outdir,rowxcol,popstr,
                    genstr,evaluate,ig):
    """ write out odd_ratio and supporting data """
    fname = outdir + "or_sod_igsum-" + rowxcol + '-' 
    fname += 's' + str(seed).zfill(3) + '-'
    fname += popstr + '-' 
    fname += genstr + '-' 
    fname += evaluate + '-ig' + str(ig) + 'way.txt'
    f = open(fname, 'w')
    f.write("Individual\tFitness\tSOD\tigsum\tOR_list\tModel\n")
    for i in range(len(best)):
        f.write(str(i))
        f.write('\t')
        f.write(str(fitness[i][0]))
        f.write('\t')
        f.write(str(best[i].SOD))
        f.write('\t')
        f.write(str(best[i].igsum))
        f.write('\t')
        f.write(str(best[i].OR.tolist()))
        f.write('\t')
        f.write(str(best[i]))
        f.write('\n')

    f.close()
