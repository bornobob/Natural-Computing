import math
import itertools
import random
import matplotlib.pyplot as plt


def read_file(filename):
    result = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            split = line.split()
            result[i] = (float(split[0]), float(split[1]))
    return result


N = 8
GENS = 150
MAPPING = read_file('file-tsp-2.txt')
LOCAL_SEARCH = True


def distance(source, goal):
    s_x, s_y = MAPPING[source]
    g_x, g_y = MAPPING[goal]
    return math.sqrt((s_x - g_x)**2 + (s_y - g_y)**2)


def get_ordering(route, exclude, start):
    double_route = itertools.chain.from_iterable(itertools.repeat(route, 2))
    order = list(double_route)[start:start + len(route)]
    return [i for i in order if i not in exclude]


def order_crossover(route1, route2, cut1, cut2):
    rout_len = len(route1)

    if cut2 < cut1:
        cut1, cut2 = cut2, cut1

    cross1 = route1[cut1:cut2]
    cross2 = route2[cut1:cut2]

    os1 = [-1]*cut1 + cross1 + [-1]*(rout_len - cut2)
    os2 = [-1]*cut1 + cross2 + [-1]*(rout_len - cut2)

    order1 = get_ordering(route1, cross2, cut2)
    order2 = get_ordering(route2, cross1, cut2)

    for i, dx in enumerate(range(rout_len - (cut2 - cut1))):
        os1[(cut2 + dx) % rout_len] = order2[i]

    for i, dx in enumerate(range(rout_len - (cut2 - cut1))):
        os2[(cut2 + dx) % rout_len] = order1[i]

    return os1, os2


def crossover_points(max):
    r1, r2 = 1, 1
    while r1 == r2:
        r1 = random.randint(0, max)
        r2 = random.randint(0, max)
    return r1, r2


def mutate(route):
    if random.random() < 0.5:
        r1, r2 = crossover_points(len(route) - 1)
        route[r1], route[r2] = route[r2], route[r1]
    return route


def fitness(route):
    fitness = 0
    for i, city in enumerate(route[:-1]):
        fitness += distance(city, route[i+1])
    return fitness + (distance(route[0], route[-1]))


def random_init(n):
    res = []
    for _ in range(n):
        res.append(list(random.shuffle(list(range(len(MAPPING.keys()))))))


def tournament_selection(population, k):
    r_pop = list(population)
    random.shuffle(r_pop)
    k_ps = r_pop[:k]
    return sorted(k_ps, key=lambda x: fitness(x))[0]


def bin_tournament_selection(population, k=2):
    for _ in range(2):
        yield tournament_selection(population, k)


def two_opt_swap(route, i, k):
    new_route = route[:i]
    new_route += list(reversed(route[i:k]))
    new_route += route[k:]
    return new_route


def local_search(route):
    if LOCAL_SEARCH:
        best_fitness = fitness(route)
        for i in range(len(route) - 2):
            for k in range(i + 2, len(route)):
                new_route = two_opt_swap(route, i, k)
                new_fitness = fitness(new_route)
                if new_fitness < best_fitness:
                    return new_route
    return route


def ea():
    fitnesses = []
    pops = [list(range(len(MAPPING.keys()))) for _ in range(N)]
    for i in range(N):
        random.shuffle(pops[i])
        pops[i] = local_search(pops[i])
    for _ in range(GENS):
        cps = crossover_points(len(pops[0]) - 1)
        ps = bin_tournament_selection(pops, k=7)
        c1, c2 = order_crossover(*ps, *cps)
        c1 = mutate(c1)
        c2 = mutate(c2)
        for i in range(N):
            pops[i] = local_search(pops[i])
        pops = sorted(pops + [c1, c2], key=lambda x: fitness(x))[:N]
        fitnesses.append(fitness(pops[0]))
    return pops[0], fitnesses


if __name__ == '__main__':
    route, fits = ea()
    xs = [MAPPING[x][0] for x in route]
    ys = [MAPPING[x][1] for x in route]
    plt.subplot(1, 2, 1)
    plt.title('TS Path')
    plt.plot(xs, ys)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title('Fitness over generations')
    plt.plot(fits)
    plt.show()
