from sklearn.cluster import KMeans
import math
import numpy as np


ITERS = 1000
CLUSTERS = 3
POPULATION_SIZE = 10
C_1 = 1.49
C_2 = 1.49
W = 0.72
FEATURES = 4
DATAPOINTS = 150
RANDOM_MATRIX = np.random.uniform(low=0.001, high=1.0, size=(2, ITERS, FEATURES))
CLUSTER_BELONGINGS = np.zeros((ITERS, POPULATION_SIZE, DATAPOINTS))


def read_dataset(path):
    data = []
    known_labels = dict()
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if line:
                data_line = line.split(',')
                data.append([float(x) for x in data_line[:1]])
                label = data_line[-1]
                label_value = len(known_labels)
                if label in known_labels:
                    label_value = known_labels[label]
                else:
                    known_labels[label] = label_value
                labels.append(label_value)
    return data, labels, known_labels


population = None
data_vectors, labels, label_names = read_dataset('./iris.data')
data_vectors = np.array(data_vectors)


def distance(z, m):
    return math.sqrt(sum((z[k] - m[k])**2 for k in range(len(z))))


def velocity(v, x, y, yh, i):
    result = []
    for k in range(FEATURES):
        result.append(W * v[k] + C_1 * RANDOM_MATRIX[0, i, k] * (y[k] - x[k]) + C_2 * RANDOM_MATRIX[1, i, k] * (yh[k] - x[k]))
    return result


def quantization_error(p_i, p, iter):
    cluster_belong = CLUSTER_BELONGINGS[iter, p_i]
    data_vectors_per_cluster = [data_vectors[cluster_belong == i] for i in range(CLUSTERS)]
    return (sum(sum(distance(z, p[c]) for z in data_vectors_per_cluster[c]) / len(data_vectors_per_cluster[c])) for c in range(CLUSTERS)) / CLUSTERS


def initialize_particles():
    for _ in range(POPULATION_SIZE):
        pass


def pso_clustering():
    # 1. initialize each particle to contain Nc randomly selected cluster
    #    centroids
    particles = initialize_particles()
    velocities = np.zeros((POPULATION_SIZE, CLUSTERS, FEATURES), dtype=np.float)
    local_bests = np.zeros((POPULATION_SIZE, CLUSTERS, FEATURES), dtype=np.float)
    global_best = np.zeros((CLUSTERS, FEATURES), dtype=np.float)
    global_best_score = 10000
    local_best_scores = np.zeros((POPULATION_SIZE), dtype=np.float) + 10000

    for i in range(ITERS):
        for p_i, p in enumerate(particles):
            for z_i, z in enumerate(data_vectors):
                dist = []
                for m in p:
                    dist.append(distance(z, m))
                CLUSTER_BELONGINGS[i, p_i, z_i] = dist.index(min(dist))
            fitness = quantization_error(p_i, p, i)
            # Update global best and local best positions
            if fitness < local_best_scores[p_i]:
                local_best_scores[p_i] = fitness
                local_bests[p_i] = p
            if fitness < global_best_score:
                global_best_score = fitness
                global_best = p
        # Update the cluster centroids using equations (3) and (4):
        #   def velocity() and inline thing
        if i > 0:
            for p_i, p in enumerate(particles):
                velocities[p_i] = velocity(velocities[p_i], p, local_bests[p_i], global_best, i - 1)
                particles[p_i] += velocities[p_i]
    return global_best


if __name__ == '__main__':
    print(pso_clustering())
