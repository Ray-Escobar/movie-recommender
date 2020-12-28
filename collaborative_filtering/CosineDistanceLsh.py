from typing import List, Set, Tuple

from collaborative_filtering.LocalitySensitiveHashTable import LocalitySensitiveHashTable
import numpy as np

class CosineDistanceLsh(LocalitySensitiveHashTable):

    def __init__(self, data_matrix: np.array, signiture_length: int, random_seed: int):

        np.random.seed(random_seed)

        self.data_matrix: np.array = data_matrix


        self.signiture_length: int = signiture_length

        # generate the planes used in the locality sensitive hashing
        num_movies: int = self.data_matrix.shape[1]
        self.planes = self.__generate_k_random_planes(k=self.signiture_length, dim=num_movies)

        self.lsh_map, self.signatures = self.__generate_locality_sensitive_hash_table(self.planes)


    def query_neighbors(self, row: int, k_neighbors: int, target_column: int, max_distance: int) -> List[int]:

        neighbors: List[int] = []

        target_row_signiture = self.signatures[row]

        distance = 0

        while len(neighbors) < k_neighbors and distance < max_distance:
            # get the neighbors at the currently selected distance in the hash_map
            nearby_neighbors: Set[np.array] = set()

            if target_row_signiture - distance in self.lsh_map.keys():
                nearby_neighbors.update(self.lsh_map[target_row_signiture - distance])

            if target_row_signiture + distance in self.lsh_map.keys():
                nearby_neighbors.update(self.lsh_map[target_row_signiture + distance])

            # add only the neighbors that contain ratings for the target movie
            for nearby_neighbor in nearby_neighbors:
                if self.__row_has_value_in_column(nearby_neighbor, target_column):
                    neighbors.append(nearby_neighbor)

                # if the maximum number of neighbors has been reached, stop adding neighbors
                if len(neighbors) >= k_neighbors:
                    break

            # increase the search distance by 1
            distance += 1

        return neighbors

    def __row_has_value_in_column(self, row: int, column: int) -> bool:


        row_data = self.data_matrix[row, :]

        if np.abs(row_data[column]) < 0.01:
            return False

        return True

    def __generate_locality_sensitive_hash_table(self, planes: List[np.array]) -> Tuple[dict, dict]:


        print("Generating cosine lsh signatures...")

        num_rows: int = self.data_matrix.shape[0]

        bins: dict = dict()
        signitures: dict = dict()

        for row in range(num_rows):
            print('Progress ' + str(row + 1) + " / " + str(num_rows))

            row_data: np.array = self.data_matrix[row, :]

            row_signiture: int = self.__generate_signiture(row_data, planes)

            signitures[row] = row_signiture

            if row_signiture not in bins.keys():
                bins[row_signiture] = list()

            bins[row_signiture].append(row)

        print("Finished generating signatures.")

        return bins, signitures

    def __generate_k_random_planes(self, k: int, dim: int) -> List[np.array]:


        planes = []


        for _ in range(k):
            plane = np.random.randn(dim)
            planes.append(plane)


        return planes

    def __generate_signiture(self, row_data: np.array, planes: List[np.array]) -> int:

        signiture = 0


        for plane in planes:
            signiture = signiture << 1

            if np.dot(row_data, plane) > 0:
                signiture |= 1


        return signiture
