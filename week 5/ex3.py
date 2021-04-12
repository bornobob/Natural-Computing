import operator as op
from functools import reduce
import matplotlib.pyplot as plt
from itertools import combinations
import math


def prod(iterable):
    return reduce(op.mul, iterable, 1)


def apply_weights(ps, weights):
    if not weights:
        weights = [1] * len(ps)
    return sum(([p] * w for p, w in zip(ps, weights)), [])


def calc_prob(ps):
    mapping = {i: ps[i] for i in range(len(ps))}
    indices = set(range(len(ps)))
    lower_i = (len(ps) // 2) + 1
    result = 0.0
    for i in range(lower_i, len(ps) + 1):
        combs = combinations(indices, i)
        for c in combs:
            negatives = indices.difference(set(c))
            result += prod(list(mapping[j] for j in c) + list(1 - mapping[j] for j in negatives))
    return result


def calc_lower_bound(weights):
    weight_sum = sum(weights)
    i, r = 0, 0
    while r <= weight_sum / 2:
        r += weights[i]
        i += 1
    return i


def valid_combination(comb, weights):
    needed_sum = sum(weights) / 2
    return sum(weights[c] for c in comb) > needed_sum


def calc_prob_weighted(ps, weights):
    mapping = {i: ps[i] for i in range(len(ps))}
    indices = set(range(len(ps)))
    lower_i = calc_lower_bound(weights)
    result = 0.0
    for i in range(lower_i, len(ps) + 1):
        combs = combinations(indices, i)
        for c in combs:
            if valid_combination(c, weights):
                negatives = indices.difference(set(c))
                result += prod(list(mapping[j] for j in c) + list(1 - mapping[j] for j in negatives))
    return result

def weight(p):
    err = 1 - p
    return math.log((1-err)/err)


if __name__ == '__main__':
    x_1 = list(range(1, 16))
    y_1 = list(calc_prob_weighted([0.75] + [0.6] * 10, [w] + [1] * 10) for w in range(1, 16))
    plt.ylabel('Chance of correct decision')
    plt.xlabel('Weight of the strong classifier')
    plt.title('Chance of correct decision per weight')
    plt.plot(x_1, y_1)
    plt.show()

    x_2 = list(e/10 for e in range(1,10))
    y_2 = list(weight(e) for e in x_2)
    plt.ylabel('Weight')
    plt.xlabel('Classifier competence')
    plt.title('Weight given to base-learner')
    plt.plot(x_2, y_2)
    plt.show()
