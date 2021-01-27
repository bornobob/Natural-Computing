import random
import matplotlib.pyplot as plt


L = 100
P = 1/L
ITERS = 1500
REPLACE_ALWAYS = False


def fitness(x):
    res = 0
    for i in x:
        if i == '1':
            res += 1
    return res


def generate_bitstring():
    return '{0:100b}'.format(random.randint(0, 2**L - 1)).replace(' ', '0')


def attack_bitstring(x):
    res = ''
    for i in x:
        if random.random() < P:
            res += '1' if i == '0' else '0'
        else:
            res += i
    return res


def ga():
    x = generate_bitstring()
    fitnesses = [fitness(x)]
    for _ in range(ITERS):
        xm = attack_bitstring(x)
        if REPLACE_ALWAYS or fitness(xm) > fitness(x):
            x = xm
        fitnesses.append(fitness(x))
    return fitnesses


if __name__ == '__main__':
    fitness_results = ga()
    plt.plot(range(ITERS + 1), fitness_results)
    plt.title('Counting Ones Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    print('Max:', max(fitness_results))
    plt.show()
