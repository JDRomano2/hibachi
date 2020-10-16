"""
Hibachi - Data simulation software that creates data sets with particular
characteristics
"""

import os
import random
import numpy as np

from deap import algorithms, base, creator, tools, gp

import ipdb

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


class Hibachi:
    """Run the Hibachi application to generate a dataset.
    """

    def __init__(self):
        self.options = hibachi_io.parse_args()

        ipdb.set_trace()

        if self.options.seed is None:
            self.rseed = -999
        else:
            self.rseed = options["seed"]

        random.seed(self.rseed)
        np.random.seed(self.rseed)

        if self.infile == "random":
            self.infile_base = "random"
        else:
            self.infile_base = os.path.splitext(os.path.basename(self.infile))[0]

        self.rowxcol = str(self.rows) + "x" + str(cols)
        self.popstr = "p" + str(self.population)
        self.genstr = "g" + str(self.generations)

        self.out_file = "-".join(
            [
                "results",
                self.infile_base,
                self.rowxcol,
                "s" + str(self.rseed),
                self.popstr,
                self.genstr,
                self.evaluate,
                "ig" + str(self.ig) + "way.txt",
            ]
        )

        self._run()

    def _run(self):
        pass
        # Read data into list of lists

        # Define primitive set for strongly-typed GP

        # Set up stats and population size

        # Start the process
