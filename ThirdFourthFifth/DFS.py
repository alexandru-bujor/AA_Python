def dfs(adj_matrix):
    visited = [False] * len(adj_matrix)
    result = []

    def dfs_recursive(node):
        visited[node] = True
        result.append(node)
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs_recursive(neighbor)

    dfs_recursive(0)
    return result


matrix = [
 [0, 1, 1, 0, 0],  # Node 0 → connected to 1, 2
 [1, 0, 0, 1, 1],  # Node 1 → connected to 0, 3, 4
 [1, 0, 0, 0, 0],  # Node 2 → connected to 0
 [0, 1, 0, 0, 0],  # Node 3 → connected to 1
 [0, 1, 0, 0, 0]   # Node 4 → connected to 1
]

print(dfs(matrix))