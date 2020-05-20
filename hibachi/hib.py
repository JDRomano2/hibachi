"""
Hibachi - Data simulation software that creates data sets with particular
characteristics
"""

import os
import random
import numpy as np

from deap import algorithms, base, creator, tools, gp

from . import io as hibachi_io

def eval_individual(individual, xdata, xtranspose):
    pass

def pareto_eq(ind1, ind2):
    """Determine whether two individuals are equal on the Pareto front.

    Parameters
    ----------
    ind1
    ind2

    Returns
    -------
    type
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)

def hibachi(pop, gen, rseed, showall):
    """Set up stats and population size, then start the hibachi process.

    Parameters
    ----------
    pop
    gen
    rseed : int
    showall
    """
    pass

class Run:
    """Run the Hibachi application to generate a dataset.
    """
    def __init__(self):
        options = hibachi_io.parse_args()

        self.infile = options['file']
        self.evaluate = options['evaluation']
        self.population = options['population']
        self.generations = options['generations']
        self.rdf_count = options['random_data_files']
        self.ig = options['information_gain']
        self.rows = options['rows']
        self.cols = options['columns']
        self.stats = options['statistics']
        self.trees = options['trees']
        self.fitness = options['fitness']
        self.prcnt = options['percent']
        self.outdir = options['outdir']
        self.showall = options['showallfitnesses']
        self.model_file = options['model_file']

        if options['seed'] is None:
            self.rseed = -999
        else:
            self.rseed = options['seed']

        random.seed(self.rseed)
        np.random.seed(self.rseed)

        if self.infile == 'random':
            self.infile_base = 'random'
        else:
            self.infile_base = os.path.splitext(os.path.basename(self.infile))[0]

        self.rowxcol = str(self.rows) + 'x' + str(cols)
        self.popstr = 'p' + str(self.population)
        self.genstr = 'g' + str(self.generations)

        self.out_file = '-'.join([
            "results",
            self.infile_base,
            self.rowxcol,
            's' + str(self.rseed),
            self.popstr,
            self.genstr,
            self.evaluate,
            'ig' + str(self.ig) + "way.txt"
        ])

        self._run_algorithm()

    def _run_algorithm(self):
        pass
        # Read data into list of lists

        # Define primitive set for strongly-typed GP

        # Set up stats and population size

        # Start the process
