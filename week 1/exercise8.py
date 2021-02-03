from deap import gp
from deap import tools, base, creator, algorithms
import operator
import math


POP_SIZE = 1000
GENS = 50
CROSS = 0.7
MUTAT = 0


def protectedDiv(x, y):
    if y == 0:
        return 0
    return x / y


def protectedLog(x):
    if x == 0:
        return -999
    if x < 0:
        return 0
    else:
        return math.log(x)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


pset = gp.PrimitiveSet('main', 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(protectedDiv, 2)
pset.renameArguments(ARG0='x')


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


good_function = {-1.0: 0.0000, -0.9: -0.1629, -0.8: -0.2624, -0.7: -0.3129, -0.6: -0.3264, -0.5: -0.3125, -0.4: -0.2784, -0.3: -0.2289, -0.2: -0.1664, -0.1: -0.0909, 0: 0.0, 0.1: 0.1111, 0.2: 0.2496, 0.3: 0.4251, 0.4: 0.6496, 0.5: 0.9375, 0.6: 1.3056, 0.7: 1.7731, 0.8: 2.3616, 0.9: 3.0951, 1.0: 4.0000}


def evalFunction(individual, points):
    func = toolbox.compile(expr=individual)
    error = 0
    for i in points:
        error += abs(func(i) - good_function[i])
    return error,


toolbox.register("evaluate", evalFunction, points=[x/10. for x in range(-10, 11)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)

algorithms.eaSimple(pop, toolbox, CROSS, MUTAT, GENS, halloffame=hof, verbose=True)

print(hof[0])
