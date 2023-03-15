import copy
import networkx as nx
import random
import graph

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


def custom_to_networkX_graph(g):
    """
    :param g: custom graph from graph.py
    :return: Graph object from NetworkX package
    """
    G = nx.Graph()

    for i in range(len(g.neighbors)):
        G.add_node(i, iri=g.index_to_iri[i])

    for i in range(len(g.neighbors)):
        for j in range(len(g.neighbors[i])):
            G.add_edge(i, g.neighbors[i][j], capacity=1)

    return G


def networkX_to_custom_graph(G, original_g):
    """
    :param G: networkX graph object
    :param original_g: original custom graph object from graph.py
    :return: custom graph object from graph.py with only those triples that remain after decomposition through networkX
    """
    g = copy.deepcopy(original_g)
    g.neighbors = []    # neighbors is removed, since we care about the triples from now on. neighbors does not matter anymore.
    g.triples = []
    G_nodes = list(G.nodes)
    for triple in original_g.triples:
        subj = triple[0]
        pred = triple[1]
        obj = triple[2]
        if (subj in G_nodes) and (obj in G_nodes):
            g.triples.append(triple)
    return g



def k_core1(G, k=None):
    G.remove_edges_from(nx.selfloop_edges(G))
    H = nx.k_core(G, k=k)

    return H


def min_cut(G, s_node=None, t_node=None):
    if s_node==None:
        s_node = list(G.nodes)[random.randint(0, len(G.nodes))]   # list(G.nodes)[0]
    if t_node==None:
        t_node = list(G.nodes)[random.randint(0, len(G.nodes))]  # list(G.nodes)[len(G.nodes) - 1]

    cut_value, partition = nx.minimum_cut(G, s_node, t_node, "capacity")
    reachable, non_reachable = partition

    return reachable, non_reachable


def con_comps_and_largest(G):
    comps = list(nx.connected_components(G))
    max_size = 0
    index_max_size = 0
    for i in range(len(comps)):
        if max_size <= len(comps[i]):
            max_size = len(comps[i])
            index_max_size = i
    return comps, index_max_size


def remove_smaller_connected_components(G, comps=None, largest=None):
    """
    Computes the connected components of graph G, creates a copy H of G, and removes all of connected components
    from H except the largest one.
    :param G:
    :return:
    """
    if comps==None and largest==None:
        comps, largest = con_comps_and_largest(G)
    H = copy.deepcopy(G)
    for c in range(len(comps)):
        if c != largest:
                H.remove_nodes_from(comps[c])

    return H


def global_w(g):
    """
    Implements Algorithm 1: Global Weak summarization of a graph
    :param g: Graph object from graph.py. NOT NetworkX graph.
    :return:
    """
    def fuse(array):
        return min(array)

    op = {}
    ip = {}
    s = {}
    t = {}
    # 1
    for triple in g.triples:
        if triple[0] not in op:
            op[triple[0]] = []
            ip[triple[0]] = []
        if triple[2] not in op:
            op[triple[2]] = []
            ip[triple[2]] = []
        op[triple[0]].append(triple[1])
        ip[triple[2]].append(triple[1])
    # 2
    for n in list(op):
        # 2.1
        x = fuse([s[p] for p in op[n]])
        if len(x) == 0:
            x = n
        # 2.2
        y = fuse([t[p] for p in ip[n]])
        if len(y) == 0:
            y = n
        # 2.3
        z = fuse([x, y])
        # 2.4
        for p in ip[n]:
            s[p] = z
        # 2.5
        for p in op[n]:
            t[p] = z
    # 3
    for n in list(ip):
        # 2.1
        x = fuse([t[p] for p in ip[n]])
        if len(x) == 0:
            x = n
        # 2.2
        y = fuse([s[p] for p in op[n]])
        if len(y) == 0:
            y = n
        # 2.3
        z = fuse([x, y])
        # 2.4
        for p in op[n]:
            t[p] = z
        # 2.5
        for p in ip[n]:
            s[p] = z
    # 4
    # these two dictionaries contain information about which original nodes have been summarized into s[p] and t[p].
    covered_by_s = {}
    covered_by_t = {}
    h = graph.Graph()
    for triple in g.triples:
        subj = triple[0]
        pred = triple[1]
        obj = triple[2]
        if pred not in covered_by_s:
            covered_by_s[pred] = []
        if pred not in covered_by_t:
            covered_by_t[pred] = []
        covered_by_s[pred].append(subj)
        covered_by_t[pred].append(obj)
        h.triples.append([s[pred], pred, t[pred]])



