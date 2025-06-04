from matplotlib import pyplot as plt
import time
from DenseGraph import DenseGraph
from SparseGraph import SparseGraph
from CyclicGraph import CyclicGraph
from DFS import dfs
from BFS import bfs
from FloydWarshall import floyd_warshall
from Dijkstra import dijkstra
from Kruskal import kruskal_algorithm, extract_edges_from_adj_matrix
from Prim import prim_algorithm

# --- Configuration ---
sizes = [10, 50, 100, 200, 300, 500]

# Define graph types to test
graph_types = [
    (SparseGraph, "Sparse"),
    (DenseGraph, "Dense"),
    (CyclicGraph, "Cyclic")
]

# --- Helper Functions for Timing and Plotting ---

def time_algorithm(algorithm, *args, **kwargs): # Accepts multiple positional and keyword arguments
    """Measures the execution time of a given algorithm with its arguments."""
    start = time.perf_counter()
    algorithm(*args, **kwargs) # Unpacks args and kwargs for the algorithm
    end = time.perf_counter()
    return end - start

def plot_graph_times(sizes, alg1_times, alg2_times, title, alg1_label, alg2_label, alg1_color, alg2_color):
    """Generates and displays a plot comparing two algorithm's execution times."""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, alg1_times, label=alg1_label, marker='o', color=alg1_color)
    plt.plot(sizes, alg2_times, label=alg2_label, marker='x', color=alg2_color)
    plt.title(f"{title} Execution Time")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_combined_graphs(sizes, all_results, title, alg1_suffix, alg2_suffix):
    """Generates and displays a combined plot for multiple graph types and algorithms."""
    plt.figure(figsize=(12, 8))
    for label, alg1_times, alg2_times in all_results:
        plt.plot(sizes, alg1_times, label=f"{label} {alg1_suffix}", linestyle='-', marker='o')
        plt.plot(sizes, alg2_times, label=f"{label} {alg2_suffix}", linestyle='--', marker='x')

    plt.title(title)
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- BFS and DFS Performance Testing ---

def test_bfs_dfs_performance(graph_class, sizes, label):
    """Tests BFS and DFS performance for a given graph class."""
    dfs_times = []
    bfs_times = []

    print(f"\n--- Testing BFS and DFS for {label} Graph ---")
    for size in sizes:
        graph = graph_class(size)
        dfs_time = time_algorithm(dfs, graph.get_adj_matrix())
        bfs_time = time_algorithm(bfs, graph.get_adj_list())

        dfs_times.append(dfs_time)
        bfs_times.append(bfs_time)

        print(f"{label} | Size: {size} | DFS: {dfs_time:.6f}s | BFS: {bfs_time:.6f}s")

    return dfs_times, bfs_times

# --- Execute BFS and DFS Tests and Plot Results ---

all_bfs_dfs_results = []

for graph_class, label in graph_types:
    dfs_times, bfs_times = test_bfs_dfs_performance(graph_class, sizes, label)
    plot_graph_times(sizes, dfs_times, bfs_times, f"{label} Graph - DFS vs BFS", "DFS", "BFS", 'blue', 'green')
    all_bfs_dfs_results.append((label, dfs_times, bfs_times))

# Plotting summary for BFS and DFS
plot_combined_graphs(sizes, all_bfs_dfs_results, "DFS vs BFS Execution Time Across Graph Types", "DFS", "BFS")

# --- Shortest Path Algorithms Performance Testing (Floyd-Warshall and Dijkstra) ---

def test_floyd_warshall_performance(graph_class, sizes, label):
    """Tests Floyd-Warshall performance for a given graph class."""
    fw_times = []

    print(f"\n--- Testing Floyd-Warshall for {label} Graph ---")
    for size in sizes:
        graph = graph_class(size)
        # Deep copy of the matrix to avoid modifying the original during Floyd-Warshall
        matrix_copy = [row[:] for row in graph.get_adj_matrix()]

        elapsed = time_algorithm(floyd_warshall, matrix_copy)
        fw_times.append(elapsed)

        print(f"{label} | Size: {size} | Floyd-Warshall: {elapsed:.6f}s")
    return fw_times

def test_dijkstra_performance(graph_class, sizes, label):
    """Tests Dijkstra's algorithm performance for a given graph class."""
    dijkstra_times = []

    print(f"\n--- Testing Dijkstra for {label} Graph ---")
    for size in sizes:
        graph = graph_class(size)
        # Assuming Dijkstra takes an adjacency matrix and a start node (e.g., 0)
        elapsed = time_algorithm(lambda g: dijkstra(g, 0), graph.get_adj_matrix())
        dijkstra_times.append(elapsed)
        print(f"{label} | Size: {size} | Dijkstra: {elapsed:.6f}s")
    return dijkstra_times

# --- Execute Shortest Path Algorithm Tests and Plot Results ---

all_shortest_path_results = []

for graph_class, label in graph_types:
    fw_times = test_floyd_warshall_performance(graph_class, sizes, label)
    dijkstra_times = test_dijkstra_performance(graph_class, sizes, label)
    plot_graph_times(sizes, fw_times, dijkstra_times, f"{label} Graph - Floyd-Warshall vs Dijkstra", "Floyd-Warshall", "Dijkstra", 'red', 'purple')
    all_shortest_path_results.append((label, fw_times, dijkstra_times))

# Plotting summary for shortest path algorithms
plot_combined_graphs(sizes, all_shortest_path_results, "Floyd-Warshall vs Dijkstra - All Graph Types", "Floyd-Warshall", "Dijkstra")

# --- Minimum Spanning Tree (MST) Algorithms Performance Testing (Kruskal and Prim) ---

def test_kruskal_performance(graph_class, sizes, label):
    """Tests Kruskal's algorithm performance for a given graph class."""
    times = []
    print(f"\n--- Testing Kruskal for {label} Graph ---")
    for size in sizes:
        graph = graph_class(size)
        # Kruskal typically needs a list of edges, assuming extract_edges_from_adj_matrix handles this
        edges = extract_edges_from_adj_matrix(graph.get_adj_matrix())
        elapsed = time_algorithm(kruskal_algorithm, size, edges)
        times.append(elapsed)
        print(f"{label} | Size: {size} | Kruskal: {elapsed:.6f}s")
    return times

def test_prim_performance(graph_class, sizes, label):
    """Tests Prim's algorithm performance for a given graph class."""
    times = []
    print(f"\n--- Testing Prim for {label} Graph ---")
    for size in sizes:
        graph = graph_class(size)
        # Prim typically needs an adjacency matrix
        elapsed = time_algorithm(prim_algorithm, graph.get_adj_matrix())
        times.append(elapsed)
        print(f"{label} | Size: {size} | Prim: {elapsed:.6f}s")
    return times

# --- Execute MST Algorithm Tests and Plot Results ---

all_mst_results = []

for graph_class, label in graph_types:
    prim_times = test_prim_performance(graph_class, sizes, label)
    kruskal_times = test_kruskal_performance(graph_class, sizes, label)
    plot_graph_times(sizes, prim_times, kruskal_times, f"{label} Graph - Prim vs Kruskal", "Prim's Algorithm", "Kruskal's Algorithm", 'orange', 'brown')
    all_mst_results.append((label, prim_times, kruskal_times))

# Plotting summary for MST algorithms
plot_combined_graphs(sizes, all_mst_results, "Prim vs Kruskal - All Graph Types", "Prim", "Kruskal")