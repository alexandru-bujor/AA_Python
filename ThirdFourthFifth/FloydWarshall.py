def floyd_warshall(dist):
    V = len(dist)
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] != 100000000 and dist[k][j] != 100000000:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
