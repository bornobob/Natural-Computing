from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


ITERS = 100
CLUSTERS = 2
POPULATION_SIZE = 25
C_1 = 1.49
C_2 = 1.49
W = 0.72
FEATURES = 2
DATAPOINTS = 400
RANDOM_MATRIX = np.random.uniform(low=0.001, high=1.0, size=(2, ITERS, CLUSTERS, FEATURES))
CLUSTER_BELONGINGS = np.zeros((ITERS, POPULATION_SIZE, DATAPOINTS))


def read_dataset():
    data = np.zeros((DATAPOINTS, FEATURES))
    labels = np.zeros(DATAPOINTS)
    for i in range(DATAPOINTS):
        zs = np.random.uniform(-1, 1, (2,))
        z1, z2 = zs[0], zs[1]
        label = int(z1 >= 0.7 or (z1 <= 0.3 and z2 >= -0.2 - z1))
        data[i] = zs
        labels[i] = label
    return data, labels


population = None
data_vectors, labels = read_dataset()
data_vectors = np.array(data_vectors)


def distance(z, m):
    return np.sqrt(np.sum(z - m)**2)


def velocity(v, x, y, yh, i):
    result = np.zeros((CLUSTERS, FEATURES))
    for k in range(CLUSTERS):
        result[k] = W * v[k] + C_1 * RANDOM_MATRIX[0, i, k] * (y[k] - x[k]) + C_2 * RANDOM_MATRIX[1, i, k] * (yh[k] - x[k])
    return result


def quantization_error(p_i, p, iter):
    cluster_belong = CLUSTER_BELONGINGS[iter, p_i]
    data_vectors_per_cluster = [data_vectors[cluster_belong == i] for i in range(CLUSTERS)]
    outer_sum = 0
    for c in range(CLUSTERS):
        inner_sum = 0
        for z in data_vectors_per_cluster[c]:
            dist = distance(z, p[c])
            nr_data_vectors_for_C = len(data_vectors_per_cluster[c]) 
            inner_sum += (dist / nr_data_vectors_for_C)
        outer_sum += inner_sum
    return outer_sum / CLUSTERS


def initialize_particles(hybrid):
    res = np.random.uniform(low=0, high=1, size=(POPULATION_SIZE, CLUSTERS, FEATURES))
    if hybrid:
        res[0] = k_means()
    return res


def pso_clustering(hybrid):
    # 1. initialize each particle to contain Nc randomly selected cluster
    #    centroids
    particles = initialize_particles(hybrid)
    velocities = np.zeros((POPULATION_SIZE, CLUSTERS, FEATURES), dtype=np.float)
    local_bests = np.zeros((POPULATION_SIZE, CLUSTERS, FEATURES), dtype=np.float)
    global_best = np.zeros((CLUSTERS, FEATURES), dtype=np.float)
    global_best_score = 10000
    local_best_scores = np.zeros((POPULATION_SIZE), dtype=np.float) + 10000
    quantization_errors = []

    for i in range(ITERS):
        for p_i, p in enumerate(particles):
            for z_i, z in enumerate(data_vectors):
                dist = []
                for m in p:
                    dist.append(distance(np.copy(z), np.copy(m)))
                CLUSTER_BELONGINGS[i, p_i, z_i] = dist.index(min(dist))
            # print(CLUSTER_BELONGINGS[i])
            fitness = quantization_error(p_i, np.copy(p), i)
            # print(fitness)
            # Update global best and local best positions
            if fitness < local_best_scores[p_i]:
                local_best_scores[p_i] = fitness
                local_bests[p_i] = np.copy(p)
            if fitness < global_best_score:
                global_best_score = fitness
                global_best = np.copy(p)
        quantization_errors.append(global_best_score)
        # Update the cluster centroids using equations (3) and (4):
        #   def velocity() and inline thing
        if i > 0:
            for p_i, p in enumerate(particles):
                velocities[p_i] = velocity(np.copy(velocities[p_i]), np.copy(p), np.copy(local_bests[p_i]), np.copy(global_best), i - 1)
                particles[p_i] += np.copy(velocities[p_i])
    return global_best, quantization_errors


def k_means():
    kmeans = KMeans(n_clusters=CLUSTERS, max_iter=ITERS)
    y_kmeans = kmeans.fit(data_vectors)
    return y_kmeans.cluster_centers_


if __name__ == '__main__':
    _, errors_hybrid = pso_clustering(hybrid=True)
    _, errors_nonhybrid = pso_clustering(hybrid=False)
    plt.title('Quantization error over iterations')
    plt.subplot(1, 2, 1)
    plt.gca().set_title('Without Hybrid')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.plot(list(range(ITERS)), errors_nonhybrid)
    plt.subplot(1, 2, 2)
    plt.gca().set_title('With Hybrid')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.plot(list(range(ITERS)), errors_hybrid)
    plt.show()
