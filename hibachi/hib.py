"""
Hibachi - Data simulation software that creates data sets with particular
characteristics
"""

import os
import random
import itertools
import operator as op
import numpy as np

from deap import algorithms, base, creator, tools, gp

from . import io as hibachi_io
from . import operators as ops


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


def build_primitive_set(inst_length):
    """Define a new primitive set for strongly typed GP."""
    pset = gp.PrimitiveSetTyped(
        "MAIN", itertools.repeat(float, inst_length), float, "X"
    )
    # basic operators
    pset.addPrimitive(ops.addition, [float, float], float)
    pset.addPrimitive(ops.subtract, [float, float], float)
    pset.addPrimitive(ops.multiply, [float, float], float)
    pset.addPrimitive(ops.safediv, [float, float], float)
    pset.addPrimitive(ops.modulus, [float, float], float)
    pset.addPrimitive(ops.plus_mod_two, [float, float], float)
    # logic operators
    pset.addPrimitive(ops.equal, [float, float], float)
    pset.addPrimitive(ops.not_equal, [float, float], float)
    pset.addPrimitive(ops.gt, [float, float], float)
    pset.addPrimitive(ops.lt, [float, float], float)
    pset.addPrimitive(ops.AND, [float, float], float)
    pset.addPrimitive(ops.OR, [float, float], float)
    pset.addPrimitive(ops.xor, [float, float], float)
    # bitwise operators
    pset.addPrimitive(ops.bitand, [float, float], float)
    pset.addPrimitive(ops.bitor, [float, float], float)
    pset.addPrimitive(ops.bitxor, [float, float], float)
    # unary operators
    pset.addPrimitive(op.abs, [float], float)  # Should we implement natively?
    pset.addPrimitive(ops.NOT, [float], float)
    pset.addPrimitive(ops.factorial, [float], float)
    pset.addPrimitive(ops.left, [float, float], float)
    pset.addPrimitive(ops.right, [float, float], float)
    # large operators
    pset.addPrimitive(ops.power, [float, float], float)
    pset.addPrimitive(ops.logAofB, [float, float], float)
    pset.addPrimitive(ops.permute, [float, float], float)
    pset.addPrimitive(ops.choose, [float, float], float)
    # misc operators
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)
    # terminals
    randval = "rand" + str(random.random())[2:]  # so it can rerun from ipython
    pset.addEphemeralConstant(randval, lambda: random.random() * 100, float)
    pset.addTerminal(0.0, float)
    pset.addTerminal(1.0, float)

    return pset


class Hibachi:
    """Run the Hibachi application to generate a dataset.
    """

    def __init__(self):
        self.options = hibachi_io.parse_args()

        if self.options.rand_seed is None:
            self.rseed = random.randint(0, 1000)
        else:
            self.rseed = self.options.rand_seed

        random.seed(self.rseed)
        np.random.seed(self.rseed)

        if self.options.infile == "random":
            self.infile_base = "random"
        else:
            self.infile_base = os.path.splitext(os.path.basename(self.infile))[0]

        self.rowxcol = str(self.options.num_rows) + "x" + str(self.options.num_columns)
        self.popstr = "p" + str(self.options.pop_size)
        self.genstr = "g" + str(self.options.num_generations)

        self.out_file = "-".join(
            [
                "results",
                self.infile_base,
                self.rowxcol,
                "s" + str(self.rseed),
                self.popstr,
                self.genstr,
                self.options.evaluation,
                "ig" + str(self.options.inf_gain_type) + "way.txt",
            ]
        )

        self._run()

        self.read_data()

        self.pset = build_primitive_set(self.data.shape[-1])

        # set up a DEAP individual and register it in the toolbox
        if self.options.evaluation == "oddsratio":
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        else:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def read_data(self):
        if self.options.infile == "random":
            self.data = hibachi_io.get_input_data(
                "random",
                self.options.num_rows,
                self.options.num_columns,
                self.rseed,
            )
        else:
            self.data = hibachi_io.get_input_data(self.options.infile)

    def _run(self):
        pass
        # Read data into list of lists

        # Define primitive set for strongly-typed GP

        # Set up stats and population size

        # Start the process


def run_hibachi():
    h = Hibachi()
