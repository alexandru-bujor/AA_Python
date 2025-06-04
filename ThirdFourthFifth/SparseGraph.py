import random

class SparseGraph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.matrix = [[100000000 if i != j else 0 for j in range(vertices)] for i in range(vertices)]
        self.adj_list = [[] for _ in range(vertices)]
        self.adj_list_weighted = [[] for _ in range(vertices)]
        self._create_sparse_weighted_graph()

    def _create_sparse_weighted_graph(self):
        for i in range(self.vertices):
            for j in range(i + 1, self.vertices):
                if random.random() < 0.2:
                    weight = random.randint(1, 20)
                    self.matrix[i][j] = weight
                    self.matrix[j][i] = weight
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    self.adj_list_weighted[i].append((j, weight))
                    self.adj_list_weighted[j].append((i, weight))

    def get_adj_matrix(self):
        return self.matrix

    def get_adj_list(self):
        return self.adj_list

    def get_weighted_adj_list(self):
        return self.adj_list_weighted
