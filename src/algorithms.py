import copy
import networkx as nx

from collections import defaultdict

def insertionSort(arr):
    """
    Copy-pasted from GeeksForGeeks and adapted to my needs. Sorts a list in ascending order by list length.
    :param arr:
    :return:
    """
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and len(arr[i]) < len(arr[j]):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def outer_density(g, set_x, set_y):
    """
    Calculation of outer density as defined on page 4 in "Density-friendly Graph Decomposition" by Tatti (2017).
    :param g: The graph.
    :param set_x: array of node indices (referring to the index from iri_to_index)
    :param set_y: array of node indices (referring to the index from iri_to_index)
    :return:
    """
    for element in set_y:
        if element in set_x:
            set_x.remove(element)

    num_edges_from_x_to_y = 0
    num_edges_inside_x = 0
    for element in set_x:
        for neighbor in g.neighbors[element]:
            if neighbor in set_y:
                num_edges_from_x_to_y += 1
            else:
                num_edges_inside_x += 1
    num_edges_inside_x = num_edges_inside_x / 2  # Since each edge within x was counted twice during the for loops
    big_e_delta = num_edges_inside_x + num_edges_from_x_to_y
    d_xy = big_e_delta / len(set_x)
    return d_xy


def greedyLD_part1(g):
    temp_neighbors = {}
    for i in range(len(g.neighbors)):
        temp = []
        for j in range(len(g.neighbors[i])):
            temp.append(g.neighbors[i][j])
        temp_neighbors[i] = temp

    w = []
    while len(temp_neighbors) != 0:
        # Find node with the smallest degree and save its index
        key_smallest_degree = list(temp_neighbors.keys())[0]
        for key in temp_neighbors:
            if len(temp_neighbors[key]) <= len(temp_neighbors[key_smallest_degree]):
                key_smallest_degree = key
        w.append(key_smallest_degree)

        # Remove this node from graph
        for neighbor in temp_neighbors[key_smallest_degree]:
            temp_neighbors[neighbor].remove(key_smallest_degree)
        temp_neighbors.pop(key_smallest_degree)

        w.reverse()
    return w


def greedyLD_part2(g, w):
    n = len(g.neighbors)
    c = []
    j = 0
    # TODO: why do the authors take the set X to be exactly set y and then some? could't you just say set X starts after set y? the overlap is ignored anyways... removing this could increase performance.
    while j < n - 1:
        max_density_i = 1
        max_density = 0
        for i in range(j + 1, n):
            dij = outer_density(g, w[:i], w[:j])
            if dij >= max_density:
                max_density_i = i
                max_density = dij
        c.append(w[:max_density_i])
        j = max_density_i
    return c


def greedyLD_full(g):
    """
    Implements the Algorithm GreedyLD(G) from the paper "Density-friendly Graph Decomposition" by Tatti (2017), based on
    the Graph class of the file graph.py .
    :param g: The graph G = (V,E) representing the knowledge graph without arcs, but edges.
    :return:
    """
    w = greedyLD_part1(g)
    c = greedyLD_part2(g, w)
    return c


def k_core(g, k):
    """
    https://www.geeksforgeeks.org/find-k-cores-graph/
    :param g: Graph G
    :return: Collection c of k-cores
    """
    def DFSUtil(g, v, visited, vDegree, k):
        visited.add(v)

        for i in g.neighbors[v]:
            if vDegree[v] < k:
                vDegree[i] -= 1
                if vDegree[i] == k-1 and i in visited:
                    visited.remove(i)

            if i not in visited:
                DFSUtil(g, i, visited, vDegree, k)
                if vDegree[i] < k:
                    vDegree[v] -= 1

    visit = set()
    degree = defaultdict(lambda: 0)

    for i in range(len(g.neighbors)):
        degree[i] = len(g.neighbors[i])
    for i in range(len(g.neighbors)):
        if i not in visit:
            DFSUtil(g, i, visit, degree, k)

    allocated = set()
    k_cores = []
    for i in range(len(g.neighbors)):
        if i not in allocated:
            if degree[i] >= k:
                new_core = []
                allocated.add(i)
                new_core.append(i)
                # for j in range(len(new_core)):
                j = 0
                while j in range(len(new_core)):
                    for l in range(len(g.neighbors[new_core[j]])):
                        node = g.neighbors[new_core[j]][l]
                        if (node not in new_core) and (degree[node] >= k):
                            new_core.append(node)
                            allocated.add(node)
                    j += 1
                k_cores.append(new_core)
    return k_cores


def k_core1(g, k=None):
    G = nx.Graph()

    for i in range(len(g.neighbors)):
        G.add_node(i, iri=g.index_to_iri[i])

    for i in range(len(g.neighbors)):
        for j in range(len(g.neighbors[i])):
            G.add_edge(i, g.neighbors[i][j])
    G.remove_edges_from(nx.selfloop_edges(G))
    H = nx.k_core(G, k=k)

    return H







