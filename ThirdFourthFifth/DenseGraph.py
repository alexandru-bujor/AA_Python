import random

class DenseGraph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.matrix = [[100000000 if i != j else 0 for j in range(vertices)] for i in range(vertices)]
        self.adj_list = [[] for _ in range(vertices)]             # For DFS/BFS
        self.adj_list_weighted = [[] for _ in range(vertices)]    # For Dijkstra
        self._create_dense_weighted_graph()

    def _create_dense_weighted_graph(self):
        for i in range(self.vertices):
            for j in range(i + 1, self.vertices):
                if random.random() < 0.8:
                    weight = random.randint(1, 20)
                    # Matrix (for Floyd-Warshall)
                    self.matrix[i][j] = weight
                    self.matrix[j][i] = weight
                    # Unweighted list (for DFS/BFS)
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    # Weighted list (for Dijkstra)
                    self.adj_list_weighted[i].append((j, weight))
                    self.adj_list_weighted[j].append((i, weight))

    def get_adj_matrix(self):
        return self.matrix

    def get_adj_list(self):
        return self.adj_list

    def get_weighted_adj_list(self):
        return self.adj_list_weighted
