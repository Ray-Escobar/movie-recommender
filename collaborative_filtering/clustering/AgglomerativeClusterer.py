from typing import List
import numpy as np


class Cluster:

    def __init__(self, item_indices: List[int], sim_matrix: np.array):

        if len(item_indices) == 0:
            raise Exception('At least one item is necessary inside a cluster')

        self.item_indices = item_indices
        self.sim_matrix = sim_matrix
        self.clutroid = self.compute_clustroid()

    def get_clustroid(self):
        return self.clutroid

    def get_containing_item_indices(self):
        return self.item_indices

    def compute_clustroid(self):

        clustroid = 0
        current_distance = float('inf')

        for i in range(len(self.item_indices)):
            distance_to_others = 0 # the clustroid will be the node with the meanimum squared distance to all the other nodes
            for j in range(len(self.item_indices)):
                distance_to_others += self.sim_matrix[i][j] ** 2

            if distance_to_others < current_distance:
                clustroid = i
                current_distance = distance_to_others

        return clustroid


class AgglomerativeClusterer:
    def __init__(self, item_indices: List[int], k_clusters: int, sim_matrix: np.array):
        self.item_indices = item_indices
        self.k_clusters = k_clusters
        self.sim_matrix = sim_matrix
        self.partitioned_indices = self.__compute_clusters()

    def get_partitioned_item_indices(self):
        return self.partitioned_indices

    def __compute_clusters(self):
        clusters = []

        for item_index in self.item_indices:
            clusters.append(Cluster([item_index], self.sim_matrix))


        print('Computing clusters agglomeratively...')


        while len(clusters) > self.k_clusters:

            print('Progress {} / {}'.format(len(self.item_indices)  - len(clusters) + 1, len(self.item_indices) - self.k_clusters))

            cluster_index_1, cluster_index_2 = self.__find_min_dist_cluster_pair(clusters)
            cluster1 = clusters[cluster_index_1]
            cluster2 = clusters[cluster_index_2]

            merged_cluster = self.__merge_clusters(cluster1, cluster2)

            clusters.remove(cluster1)
            clusters.remove(cluster2)

            clusters.append(merged_cluster)

        # return the list of item indices partitioned into clusters
        item_indices_partitioned = []

        for cluster in clusters:
            item_indices_partitioned.append(cluster.get_containing_item_indices())

        return item_indices_partitioned

    def __merge_clusters(self, cluster1: Cluster, cluster2: Cluster):
        item_union = list(set(cluster1.get_containing_item_indices() + cluster2.get_containing_item_indices()))
        return Cluster(item_union, self.sim_matrix)


    def __find_min_dist_cluster_pair(self, clusters: List[Cluster]):
        if len(clusters) < 2:
            raise Exception('At least 2 clusters are necessary for the operation')
        pair = (0, 1)
        min_dist = self.sim_matrix[0][1]

        for i in range(0, len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]

                if self.sim_matrix[cluster1.get_clustroid()][cluster2.get_clustroid()] < min_dist:
                    min_dist = self.sim_matrix[cluster1.get_clustroid()][cluster2.get_clustroid()]
                    pair = (i, j)

        return pair











