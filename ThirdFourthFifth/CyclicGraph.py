import random

class CyclicGraph:
    def __init__(self, num_vertices, weighted=False, max_weight=10):
        self.V = num_vertices
        self.weighted = weighted
        self.max_weight = max_weight
        self.adj_matrix = [[0] * self.V for _ in range(self.V)]
        self.adj_list = [[] for _ in range(self.V)]
        self._build_cyclic_graph()

    def _add_edge(self, u, v):
        if self.adj_matrix[u][v] == 0:
            weight = random.randint(1, self.max_weight) if self.weighted else 1
            self.adj_matrix[u][v] = weight
            self.adj_matrix[v][u] = weight
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def _build_cyclic_graph(self):
        for i in range(self.V):
            self._add_edge(i, (i + 1) % self.V)  # create a cycle

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_adj_list(self):
        return self.adj_list
