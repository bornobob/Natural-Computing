from matplotlib.cm import get_cmap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def shuffle_in_unison(a, b):
    c = np.arange(len(a))
    np.random.shuffle(c)
    return a[c], b[c]


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    data, targets = shuffle_in_unison(*load_breast_cancer(return_X_y=True))
    train_samples = 450
    x_train = data[:train_samples]
    x_test = data[train_samples:]
    y_train = targets[:train_samples]
    y_test = targets[train_samples:]

    estimators = list(range(1, 21, 2))
    learning_rates = list(np.arange(0.01, 1.5, 0.05))
    points = []

    fig = plt.figure()
    ax = Axes3D(fig)

    for e in estimators:
        for l in learning_rates:
            a = AdaBoostClassifier(n_estimators=e, learning_rate=l, random_state=seed)
            a.fit(x_train, y_train)
            points.append((e, l, a.score(x_test, y_test)))

    xs = list(p[0] for p in points)
    ys = list(p[1] for p in points)
    zs = list(p[2] for p in points)
    ax.scatter(xs, ys, zs, c=zs, cmap='hsv')
    plt.xlabel('# Estimators')
    plt.ylabel('Learning rate')
    ax.set_zlabel('Accuracy')
    plt.show()
