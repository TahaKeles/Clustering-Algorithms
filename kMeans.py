import random

import matplotlib.pyplot as plt
import numpy as np


def euclidean(XA, XB):
    XA = np.array(XA)
    XB = np.array(XB)
    return np.sqrt(np.sum(np.square(XA - XB)))


def graph(data, y_kmeans, centers):
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    new_centers = []
    for center in centers:
        new_centers.append([center[0], center[1]])
    # print(new_centers)
    # exit()
    for cent in new_centers:
        plt.scatter(cent[0], cent[1], c='black', s=200, alpha=0.5);
    plt.savefig("part2.png")
    plt.clf()


def random_centers(dim, k):
    centers = []
    for i in range(k):
        center = []
        for d in range(dim):
            rand = random.randint(0, 100)
            center.append(rand)
        centers.append(center)
    return centers


def point_clustering(data, centers, dims, first_cluster=False):
    for point in data:
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = 0
            for d in range(0, dims):
                dist = abs(point[d] - centers[i][d])
                euclidean_dist += dist
            euclidean_dist = np.sqrt(euclidean_dist)
            if nearest_center_dist == None:
                nearest_center_dist = euclidean_dist
                nearest_center = i
            elif nearest_center_dist > euclidean_dist:
                nearest_center_dist = euclidean_dist
                nearest_center = i
        if first_cluster:
            point.append(nearest_center)
        else:
            point[-1] = nearest_center
    return data


def mean_center(data, centers, dims):
    new_centers = []
    for i in range(len(centers)):
        new_center = []
        n_of_points = 0
        total_of_points = []
        for point in data:
            if point[-1] == i:
                n_of_points += 1
                for dim in range(0, dims):
                    if dim < len(total_of_points):
                        total_of_points[dim] += point[dim]
                    else:
                        total_of_points.append(point[dim])
        if len(total_of_points) != 0:
            for dim in range(0, dims):
                print(total_of_points, dim)
                new_center.append(total_of_points[dim] / n_of_points)
            new_centers.append(new_center)
        else:
            new_centers.append(centers[i])
    return new_centers


def train_k_means_clustering(data, k=2, epochs=5):
    dims = len(data[0])
    print('data[0]:', data[0])
    centers = random_centers(dims, k)

    clustered_data = point_clustering(data, centers, dims, first_cluster=True)

    for i in range(epochs):
        centers = mean_center(clustered_data, centers, dims)
        clustered_data = point_clustering(data, centers, dims, first_cluster=False)

    return centers


def predict_k_means_clustering(point, centers):
    dims = len(point)

    nearest_center = None
    nearest_dist = None

    for i in range(len(centers)):
        euclidean_dist = 0
        for dim in range(1, dims):
            dist = point[dim] - centers[i][dim]
            euclidean_dist += dist ** 2
        euclidean_dist = np.sqrt(euclidean_dist)
        if nearest_dist == None:
            nearest_dist = euclidean_dist
            nearest_center = i
        elif nearest_dist > euclidean_dist:
            nearest_dist = euclidean_dist
            nearest_center = i

    return nearest_center


aa = {
    "euclidean": euclidean
}


class KMeans():

    def __init__(self, k=3, init_centroid="random", distance="euclidean"):
        self.k = k
        self.init_centroid = init_centroid
        self.distance = aa[distance]

    def choose_random_point(self, X):

        min_val = np.min(X)
        max_val = np.max(X)
        return np.random.uniform(low=min_val, high=max_val, size=(self.n_features,))

    def random_init(self, X):

        initial_centroids = []
        for _ in range(self.k):
            rand_centroid = self.choose_random_point(X)
            initial_centroids.append(rand_centroid)
        return initial_centroids

    def train(self, X, max_iteration=100):

        X = np.array(X)

        self.n_features = X[0].shape[0]
        self.centroids = []
        self.centroids = self.random_init(X)

        self.cluster_members = [[] for _ in range(self.k)]

        iteration = 0
        while iteration < max_iteration:

            current_inertia = float(0.0)
            current_cluster_members = [[] for _ in range(self.k)]
            for data_point in X:
                min_distance = float("inf")
                cluster = 0
                for cluster_idx, centroid_i in enumerate(self.centroids):
                    distance = self.distance(centroid_i, data_point)
                    if distance <= min_distance:
                        cluster = cluster_idx
                        min_distance = distance
                current_cluster_members[cluster].append(data_point)
                current_inertia = current_inertia + min_distance

            new_centroids = [[] for _ in range(self.k)]
            for cluster_i in range(self.k):
                new_centroid_i = np.zeros(self.n_features)
                members_of_current_cluster = current_cluster_members[cluster_i]
                if len(members_of_current_cluster) > 0:
                    for member in current_cluster_members[cluster_i]:
                        new_centroid_i = new_centroid_i + member
                    new_centroid_i = new_centroid_i / len(
                        members_of_current_cluster)
                else:
                    new_centroid_i = self.choose_random_point(X)

                new_centroids[cluster_i] = new_centroid_i

            total_diff = float(0.0)
            for cluster_i in range(self.k):
                total_diff = total_diff + self.distance(self.centroids[cluster_i], new_centroids[cluster_i])

            self.centroids = new_centroids
            self.cluster_members = current_cluster_members
            self.inertia = current_inertia

            iteration = iteration + 1

        self.n_iteration = iteration
        return self.predict(X)

    def predict(self, X):
        result = []
        for data_point in X:
            # calculate distance to each centroids
            min_distance = float("inf")
            cluster = None
            for cluster_idx, centroid_i in enumerate(self.centroids):
                distance = self.distance(centroid_i, data_point)
                if distance <= min_distance:
                    cluster = cluster_idx
                    min_distance = distance
            result.append(cluster)
        return result


def elbow_method(X, k_range=range(1, 9), init_centroid="random", distance="euclidean"):
    X = np.array(X)
    distortion_scores = []
    silhouete_scores = []
    for k in k_range:
        model = KMeans(k=k, init_centroid=init_centroid, distance=distance)
        model.train(X)
        distortion_scores.append(model.inertia)
        silhouete_scores.append(silhouete_score(X, model.centroids))
        #print(k)

    plt.plot(k_range, distortion_scores)
    plt.xlabel('K')
    plt.ylabel('Avg. Obj. Func. Value')
    plt.savefig("elbowmethod.png")
    plt.clf()

    return distortion_scores


def silhouete_coef(x, centroids):
    distances = []
    if len(centroids) == 1:
        return 0
    for centroid in centroids:
        distances.append(euclidean(x, centroid))
    distances.sort()
    return (distances[1] - distances[0]) / max(distances[0], distances[1])


def silhouete_score(X, centroids):
    X = np.array(X)
    silhouete_coefs = []
    for point in X:
        silhouete_coefs.append(silhouete_coef(point, centroids))
    return sum(silhouete_coefs) / len(silhouete_coefs)


if __name__ == '__main__':
    clustering1 = np.load("./data/kmeans/clustering1.npy")
    clustering2 = np.load("./data/kmeans/clustering2.npy")
    clustering3 = np.load("./data/kmeans/clustering3.npy")
    clustering4 = np.load("./data/kmeans/clustering4.npy")

    # uncomment for model change

    # kmeans_model = KMeans(k=2, init_centroid="random", distance="euclidean")
    # kmeans_model.train(clustering1, max_iteration=100)

    # kmeans_model = KMeans(k=3, init_centroid="random", distance="euclidean")
    # kmeans_model.train(clustering2, max_iteration=100)
    #
    # kmeans_model = KMeans(k=4, init_centroid="random", distance="euclidean")
    # kmeans_model.train(clustering3, max_iteration=100)
    #
    kmeans_model = KMeans(k=5, init_centroid="random", distance="euclidean")
    kmeans_model.train(clustering4, max_iteration=100)

    centers = kmeans_model.centroids
    # for plotting cluster points
    # graph(clustering1, kmeans_model.predict(clustering1), centers)
    # graph(clustering2, kmeans_model.predict(clustering2), centers)
    # graph(clustering3,kmeans_model.predict(clustering3), centers)
    graph(clustering4, kmeans_model.predict(clustering4), centers)

    # for elbow method
    # elbow_method(clustering1,range(1,10),"random","euclidean")
    # elbow_method(clustering2, range(1, 10), "random", "euclidean")
    # elbow_method(clustering3, range(1, 10), "random", "euclidean")
    #elbow_method(clustering4, range(1, 10), "random", "euclidean")
