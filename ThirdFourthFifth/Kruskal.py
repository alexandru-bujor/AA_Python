def kruskal_algorithm(size, edges):
    parent = list(range(size))
    rank = [0] * size

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(x, y):
        xroot = find(x)
        yroot = find(y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    mst = []
    edges = sorted(edges, key=lambda x: x[2])  # Sort by weight

    for u, v, weight in edges:
        if find(u) != find(v):
            mst.append((u, v, weight))
            union(u, v)

    return mst


def extract_edges_from_adj_matrix(adj_matrix):
    size = len(adj_matrix)
    edges = []
    for i in range(size):
        for j in range(i + 1, size):  # avoid double-counting (undirected graph)
            weight = adj_matrix[i][j]
            if weight != 0 and weight != float('inf'):
                edges.append((i, j, weight))
    return edges


graph_adj_matrix = [
    [0, 2, 3, 0, 0],
    [2, 0, 4, 3, 0],
    [3, 4, 0, 5, 0],
    [0, 3, 5, 0, 1],
    [0, 0, 0, 1, 0]
]
num_vertices = len(graph_adj_matrix)

print("Adjacency Matrix:")
for row in graph_adj_matrix:
    print(row)

edges = extract_edges_from_adj_matrix(graph_adj_matrix)
print("\nExtracted Edges (sorted by weight for clarity):")
# Sort for printing, Kruskal's will sort them anyway
for edge in sorted(edges, key=lambda x: x[2]):
    print(f"({edge[0]} - {edge[1]}, Weight: {edge[2]})")

mst_edges_ = kruskal_algorithm(num_vertices, edges)
total_weight = sum(edge[2] for edge in mst_edges_)

print("\nMinimum Spanning Tree Edges for Graph 1:")
for edge in mst_edges_:
    print(f"Edge: ({edge[0]} - {edge[1]}), Weight: {edge[2]}")
print(f"Total MST Weight for Graph 1: {total_weight}")