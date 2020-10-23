"""
Hibachi - Data simulation software that creates data sets with particular
characteristics
"""

import os
import sys
import random
import itertools
import operator as op
import numpy as np

from deap import algorithms, base, creator, tools, gp

from . import io as hibachi_io
from . import operators as ops
from . import eval_utils
from . import utils

import ipdb


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


class Hibachi():
    """Run the Hibachi application to generate a dataset.
    """

    def __init__(self):
        self.options = hibachi_io.parse_args()

        self.labels = []
        self.all_ig_sums = []

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

        self.read_data()

        if self.options.infile != 'random':
            self.num_rows = len(self.data)
            self.num_cols = len(self.x)

        self.inst_length = len(self.x)

        self.pset = build_primitive_set(self.inst_length)

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
        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        if self.options.verbose:
            print("input data:  {0}".format(self.options.infile))
            print("data shape:  {0} X {1}".format(self.options.num_rows, self.options.num_columns))
            print("random seed: {0}".format(self.rseed))
            print("pcnt. cases: {0}".format(self.options.case_control_ratio))
            print("output dir:  {0}".format(self.options.outdir))
            if self.options.model_file is None:
                print("population size:  {0}".format(self.options.pop_size))
                print("num. generations: {0}".format(self.options.num_generations))
                print("evaluation type:  {0}".format(self.options.evaluation))
                print("ign 2/3 way:      {0}".format(self.options.inf_gain_type))
            print()

        if self.options.model_file:
            # Finish implementing!
            raise Exception

        pop, stats, hof, log = self._run()

    def read_data(self):
        if self.options.infile == "random":
            self.data, self.x = hibachi_io.get_input_data(
                "random",
                self.options.num_rows,
                self.options.num_columns,
                self.rseed,
            )
        else:
            self.data, self.x = hibachi_io.get_input_data(self.options.infile)
        self.inst_length = len(self.x)

    def eval_individual(self, individual):
        """Evaluate an individual."""
        result = []
        ig_sums = np.array([])

        # TODO: Fix polarity of these!!!
        X = self.x
        data = self.data

        eval_method = self.options.evaluation

        inst_length = self.inst_length

        func = self.toolbox.compile(expr=individual)

        try:
            result = [(func(*inst[:inst_length])) for inst in data]
        except:  #TODO: Tighten this up - what exceptions can we expect?
            return -sys.maxsize, sys.maxsize

        if len(np.unique(result)) == 1:
            return -sys.maxsize, sys.maxsize

        if eval_method == 'normal' or eval_method == 'oddsratio':
            rangeval = 1
        elif eval_method == 'folds':
            rangeval = numfolds = 10
            folds = eval_utils.getfolds(X, numfolds)
        elif eval_method == 'subsets':
            rangeval = 10
            percent = 25
        elif eval_method == 'noise':
            rangeval = 10
            percent = 10

        result = eval_utils.reclass_result(X, result, self.options.case_control_ratio)

        for m in range(rangeval):
            ig_sum = 0

            if eval_method == 'folds':
                xsub = list(folds[m])
            elif eval_method == 'subsets':
                xsub = eval_utils.subsets(X, self.options.case_control_ratio)
            elif eval_method == 'noise':
                xsub = eval_utils.addnoise(X, self.options.case_control_ratio)
            else:
                xsub = X

            # Compute information gain between columns and result; return mean
            # information gain across all samples
            if self.options.inf_gain_type == 2:
                for i, j in itertools.combinations(range(inst_length), 2):
                    ig_sum += utils.two_way_information_gain(xsub[i], xsub[j], result)
            elif self.options.inf_gain_type == 3:
                for i, j, k in itertools.combinations(range(inst_length), 3):
                    ig_sum += utils.three_way_information_gain(xsub[i], xsub[j], xsub[k], result)

            ig_sums = np.append(ig_sums, ig_sum)

        if eval_method == 'oddsratio':
            sum_of_diffs, OR = eval_utils.oddsRatio(xsub, result, inst_length)
            individual.OR = OR
            individual.SOD = sum_of_diffs
            individual.ig_sum = ig_sum

        ig_sum_avg = np.mean(ig_sums)
        self.labels.append((ig_sum_avg, result))
        self.all_ig_sums.append(ig_sums)

        if len(individual) <= 1:
            return -sys.maxsize, sys.maxsize
        else:
            if eval_method == 'oddsratio':
                individual_str = str(individual)
                uniq_col_count = 0
                for col_num in range(len(X)):
                    col_name = 'X{}'.format(col_num)
                    if col_name in individual_str:
                        uniq_col_count += 1

                return igsum, len(individual) / float(uniq_col_count), sum_of_diffs
            
            elif eval_method == 'normal':
                return ig_sum, len(individual)
            else:
                return ig_sum_avg, len(individual)

    def _run(self):
        # can't use 'lambda' as a variable name
        mu = self.options.pop_size
        lm = self.options.pop_size
        n_gen = self.options.num_generations
        np.random.seed(self.rseed)
        random.seed(self.rseed)
        pop = self.toolbox.population(n=mu)
        hof = tools.ParetoFront(similar=pareto_eq)
        if self.options.show_all_fitness:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
        else:
            stats = tools.Statistics(lambda ind: max(ind.fitness.values[0], 0))
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

        pop, log = algorithms.eaMuPlusLambda(
            pop, self.toolbox, mu=mu, lambda_=lm, cxpb=0.7,
            mutpb=0.3, ngen=n_gen, stats=stats, verbose=True,
            halloffame=hof
        )

        return pop, stats, hof, log



def run_hibachi():
    h = Hibachi()
