import math

import matplotlib.pyplot as plt
import numpy as np
import sys


def euclidean_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def distance_matrix(data):

    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    #print(distance_matrix)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            distance_matrix[i][j] = euclidean_distance(data[i],data[j])
    np.fill_diagonal(distance_matrix,sys.maxsize)
    return distance_matrix

def find_clusters(input, linkage,data):
    clusters = {}
    row_index = -1
    col_index = -1
    array = []

    for n in range(input.shape[0]):
        array.append(n)

    clusters[0] = array.copy()

    for k in range(1, input.shape[0]):
        min_val = sys.maxsize

        for i in range(0, input.shape[0]):
            for j in range(0, input.shape[1]):
                if input[i][j] <= min_val:
                    min_val = input[i][j]
                    row_index = i
                    col_index = j
        if linkage == "single":
            for i in range(0, input.shape[0]):
                if i != col_index:
                    # we calculate the distance of every data point from newly formed cluster and update the matrix.
                    temp = min(input[col_index][i], input[row_index][i])
                    # we update the matrix symmetrically as our distance matrix should always be symmetric
                    input[col_index][i] = temp
                    input[i][col_index] = temp
        # for Complete Linkage
        elif linkage == "complete":
            for i in range(0, input.shape[0]):
                if i != col_index and i != row_index:
                    # temp = min(input[col_index][i], input[row_index][i])
                    temp = max(input[col_index][i], input[row_index][i])
                    input[col_index][i] = temp
                    input[i][col_index] = temp
        # for Average Linkage
        elif linkage == "average":
            for i in range(0, input.shape[0]):
                if i != col_index and i != row_index:
                    temp = (input[col_index][i] + input[row_index][i]) / 2
                    input[col_index][i] = temp
                    input[i][col_index] = temp
        elif linkage == "centroid":
            for i in range(0, input.shape[0]):
                if i != col_index and i != row_index:
                    dist_centroid = (input[col_index][i] + input[row_index][i]) / 2
                    input[col_index][i] = dist_centroid
                    input[i][col_index] = dist_centroid
        for i in range(0, input.shape[0]):
            input[row_index][i] = sys.maxsize
            input[i][row_index] = sys.maxsize
        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(array)):
            if array[n] == maximum:
                array[n] = minimum
        clusters[k] = array.copy()
        #print(clusters[k])

    return clusters


def hierarchical_clustering(data, linkage, no_of_clusters):
    color = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
    a = distance_matrix(data)
    clusters = find_clusters(a,linkage,data)

    # plotting the clusters
    iteration_number = a.shape[0] - no_of_clusters
    clusters_to_plot = clusters[iteration_number]
    #print(clusters_to_plot)
    arr = np.unique(clusters_to_plot)

    indices_to_plot = []
    fig = plt.figure()
    fig.suptitle('Scatter Plot for clusters')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for x in np.nditer(arr):
        indices_to_plot.append(np.where(clusters_to_plot == x))
    p = 0

    #print(clusters_to_plot)
    for i in range(0, len(indices_to_plot)):
        for j in np.nditer(indices_to_plot[i]):
            ax.scatter(data[j, 0], data[j, 1], c=color[p])
        p = p + 1
    plt.savefig("part3.png")
    #plt.show()



if __name__ == "__main__":

    data1 = np.load( "data/hac/data1.npy")
    data2 = np.load( "data/hac/data2.npy")
    data3 = np.load( "data/hac/data3.npy")
    data4 = np.load( "data/hac/data4.npy")

    # uncomment for clustering and plotting graph second parameter is linkage criteria third parameter is number of cluster

    #hierarchical_clustering(data1,"single",2)
    #hierarchical_clustering(data1,"complete",2)
    #hierarchical_clustering(data1,"average",2)
    #hierarchical_clustering(data1,"centroid",2)



    #hierarchical_clustering(data2,"centroid",2)
    #hierarchical_clustering(data2,"average",2)
    #hierarchical_clustering(data2,"complete",2)
    #hierarchical_clustering(data2,"single",2)


    #hierarchical_clustering(data3,"average",2)
    #hierarchical_clustering(data3,"complete",2)
    #hierarchical_clustering(data3,"single",2)
    #hierarchical_clustering(data3,"centroid",2)


    hierarchical_clustering(data4,"average", 4)
    #hierarchical_clustering(data4,"complete", 4)
    #hierarchical_clustering(data4,"single", 4)
    #hierarchical_clustering(data4,"centroid", 4)
