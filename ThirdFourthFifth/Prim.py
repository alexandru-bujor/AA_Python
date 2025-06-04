def prim_algorithm(adj_matrix):
    size = len(adj_matrix)
    in_mst = [False] * size
    key = [float('inf')] * size
    parent = [-1] * size

    key[0] = 0  # Start from vertex 0

    for _ in range(size):
        u = min((v for v in range(size) if not in_mst[v]), key=lambda v: key[v], default=-1)
        if u == -1:
            break
        in_mst[u] = True

        for v in range(size):
            if 0 < adj_matrix[u][v] < key[v] and not in_mst[v]:
                key[v] = adj_matrix[u][v]
                parent[v] = u

    mst = [(parent[i], i, adj_matrix[parent[i]][i]) for i in range(1, size) if parent[i] != -1]
    return mst

graph1_adj_matrix = [
    [0, 2, 3, 0, 0],
    [2, 0, 4, 3, 0],
    [3, 4, 0, 5, 0],
    [0, 3, 5, 0, 1],
    [0, 0, 0, 1, 0]
]

print("Adjacency Matrix:")
for row in graph1_adj_matrix:
    print(row)

mst_edges_1 = prim_algorithm(graph1_adj_matrix)
total_weight_1 = sum(edge[2] for edge in mst_edges_1)

print("\nMST Edges for Graph:")
for edge in mst_edges_1:
    print(f"Edge: ({edge[0]} - {edge[1]}), Weight: {edge[2]}")
print(f"Total MST Weight for Graph: {total_weight_1}")