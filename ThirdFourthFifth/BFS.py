from collections import deque

def bfs(adj_list):
    visited = [False] * len(adj_list)
    result = []
    queue = deque([0])
    visited[0] = True

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return result


lst = [
 [1, 2],  # Node 0 → neighbors 1, 2
 [0, 3, 4],  # Node 1 → neighbors 0, 3, 4
 [0],  # Node 2 → neighbor 0
 [1],  # Node 3 → neighbor 1
 [1]   # Node 4 → neighbor 1
]

print(bfs(lst))