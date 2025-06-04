def dijkstra(adj_matrix, start_vertex):
    size = len(adj_matrix)
    distances = [float('inf')] * size
    distances[start_vertex] = 0
    visited = [False] * size

    for _ in range(size):
        min_distance = float('inf')
        u = None
        for i in range(size):
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                u = i

        if u is None:
            break

        visited[u] = True

        for v in range(size):
            if adj_matrix[u][v] != 0 and not visited[v]:
                alt = distances[u] + adj_matrix[u][v]
                if alt < distances[v]:
                    distances[v] = alt

    return distances

graph_matrix = [
    [0, 10, 5, 0, 0],  # From 0 to 1 (10), 0 to 2 (5)
    [0, 0, 2, 1, 0],   # From 1 to 2 (2), 1 to 3 (1)
    [0, 9, 0, 3, 2],   # From 2 to 1 (9), 2 to 3 (3), 2 to 4 (2)
    [0, 0, 0, 0, 4],   # From 3 to 4 (4)
    [6, 0, 0, 0, 0]    # From 4 to 0 (6)
]
start_node = 0

shortest_distances = dijkstra(graph_matrix, start_node)

# Print the results
print(f"Shortest distances from vertex {start_node}:")
for i, dist in enumerate(shortest_distances):
    if dist == float('inf'):
        print(f"  To vertex {i}: Unreachable")
    else:
        print(f"  To vertex {i}: {dist}")