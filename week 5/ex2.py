import operator as op
from functools import reduce
import matplotlib.pyplot as plt


# https://stackoverflow.com/a/4941932
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def calculate_probablity(c, p):
    lower_i = (c // 2) + 1
    return sum(ncr(c, i) * p**i * (1 - p)**(c - i) for i in range(lower_i, c+1))


if __name__ == '__main__':
    # Answers for ex2.d
    print(f'Chance the radiologist is right: {calculate_probablity(1, 0.85)}')
    print(f'Chance the doctors are right: {calculate_probablity(3, 0.75)}')
    print(f'Chance the students are right: {calculate_probablity(31, 0.6)}')

    for c in range(23, 28):
        print(f'Chance the {c} students are right: {calculate_probablity(c, 0.6)}')

    print(f'Converges to one? with 1000 students we get: {calculate_probablity(1000, 0.6)}')

    # CODE FOR GRAPH: ex2.c
    p = 0.65
    x_1 = list(range(60))
    y_1 = list(calculate_probablity(c, p) for c in range(60))
    ax1 = plt.subplot(1, 2, 1)
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.ylabel('Chance of correct decision')
    plt.xlabel('Group size')
    plt.title('Chance of correct decision per group size')
    plt.plot(x_1, y_1)

    c = 11
    x_2 = list(range(101))
    y_2 = list(calculate_probablity(c, p / 100) for p in range(100 + 1))
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    plt.title('Chance of correct decision per competence level')
    plt.xlabel('Competence level')
    plt.plot(x_2, y_2)
    plt.show()
h