import copy
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import random
import time
import graph
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'yellow', 'k', 'tab:orange', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:pink', 'chartreuse', 'gold']

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


def custom_to_networkX_undir_graph(g):
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
        # pred = triple[1]
        obj = triple[2]
        if (subj in G) and (obj in G):
            g.triples.append(triple)
    return g


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


def connected_comps_and_largest(G):
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
        comps, largest = connected_comps_and_largest(G)
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


def clustering_using_embedding(G, k, seed):
    timestamp = time.strftime("%d-%m_%H:%M")
    print("Current timestamp: " + timestamp)

    # Generate walks
    node2vec = Node2Vec(G, dimensions=20, walk_length=60, num_walks=80, workers=4)
    # Learn embeddings
    model = node2vec.fit(window=10, min_count=1)
    # model.wv.most_similar('1')
    model.wv.save_word2vec_format("out/embeddingclustering/%s_H.emb" % timestamp)  # save the embedding in file embedding.emb
    X = np.loadtxt("out/embeddingclustering/%s_H.emb" % timestamp, skiprows=1)  # load the embedding of the nodes of the graph
    # sort the embedding based on node index in the first column in X
    X = X[X[:, 0].argsort()];
    np.savetxt('out/embeddingclustering/%s_X' % timestamp, X, delimiter=' ')
    Z = X[0:X.shape[0], 1:X.shape[1]];  # remove the node index from X and save in Z

    kmeans = KMeans(n_clusters=k, random_state=0).fit(Z)  # apply kmeans on Z
    labels = kmeans.labels_  # get the cluster labels of the nodes.
    np.savetxt('out/embeddingclustering/%s_labels' % timestamp, labels, delimiter=' ')

    with open("out/embeddingclustering/%s_X" % timestamp, "r") as xf, open("out/embeddingclustering/%s_labels" % timestamp, "r") as lf:
        x = []
        for line in xf.readlines():
            x.append(int(float(line.split(" ")[0])))
        l = []
        for line in lf.readlines():
            l.append(int(float(line)))

        xl = {}  # each node has a label
        for i in range(len(x)):
            xl[x[i]] = l[i]
        lx = []  # each label has a list of nodes
        for i in range(k):
            lx.append([])
        for key in list(xl):
            lx[xl[key]].append(str(key))

        pos = nx.spring_layout(G, seed=seed)
        options = {"node_size": 4}
        for i in range(len(lx)):
            nx.draw_networkx_nodes(G, pos, nodelist=lx[i], node_color=colors[i], **options) # colors defined above
        nx.draw_networkx_edges(G, pos, width=0.1)
        plt.savefig("out/embeddingclustering/%s_%sclusters" % (timestamp, k))
        with open("out/embeddingclustering/%s_x_labels.txt" % timestamp, "x") as f:
            for i in range(k):
                f.write("Cluster " + str(i) + ": ")
                f.write(str(lx[i]) + "\n")


def remove_clusters_save_draw(G, clusters, seed_figure):
    timestamp = time.strftime("%d-%m_%H:%M")
    print("Current timestamp: " + timestamp)
    for cluster in clusters:
        G.remove_nodes_from(cluster)
    nx.write_adjlist(G, "out/intermediategraphs/%s_H" % timestamp)
    pos = nx.spring_layout(G, seed=seed_figure)
    options = {"node_size": 4, "width": 0.1}
    nx.draw(G, pos=pos, **options)
    plt.savefig("out/intermediategraphs/%s_H" % timestamp)


def reduce_graph_to_cluster(G, cluster):
    H = copy.deepcopy(G)
    H_nodes = list(H.nodes)
    for node in H_nodes:
        if node not in cluster:
            H.remove_node(node)
    return H


def compute_different_kcores(G, start_k, end_k, seed_figure):
    timestamp = time.strftime("%d-%m_%H:%M")
    print("Current timestamp: " + timestamp)
    for k in range(start_k, end_k + 1):
        a = time.time()
        H = k_core1(G, k=k)
        nx.write_adjlist(H, "out/kcoregraphs/%s_%s-core" % (timestamp, k))
        b = time.time()
        print("Compute & save %s-core: " % k + str(b - a))
        # # Only allow the following if your graph is small, MAXIMUM like 10000 nodes. Otherwise the pos calculation before or during nx.draw takes way too long.
        # options = {"node_size": 4, "width": 0.1}
        # pos = nx.spring_layout(G, seed=seed_figure)
        # nx.draw(H, **options)
        # c = time.time()
        # print("nx.draw() complete: " + str(c - b))
        # plt.savefig("out/kcoregraphs/%s_%s-core" % (timestamp, k))
        # d= time.time()
        # print("Draw & save figure %s-core: " + str(d - c))
        # plt.clf()


def redraw_clusterfigures_differentseed(desired_timestamp, start_k, end_k, new_seed):
    new_timestamp = time.strftime("%d-%m_%H:%M")
    print("Current timestamp: " + new_timestamp)
    for k in range(start_k, end_k + 1):
        G = nx.read_adjlist("out/kcoregraphs/%s_%s-core" % (desired_timestamp, k))
        pos = nx.spring_layout(G, seed=new_seed)
        options = {"node_size": 4, "width": 0.1}
        a = time.time()
        nx.draw(G, pos,  **options)
        plt.savefig("out/kcoregraphs/%s_%s-core" % (new_timestamp, k))
        b = time.time()
        print("Draw & save figure: " + str(b - a))


def plot_graph_using_embedding(filepath):
    G = nx.read_adjlist(filepath)
    # Generate walks
    node2vec = Node2Vec(G, dimensions=2, walk_length=60, num_walks=60, workers=4)
    # Learn embeddings
    model = node2vec.fit(window=10, min_count=1)
    # model.wv.most_similar('1')
    model.wv.save_word2vec_format("%s.emb" % filepath)  # save the embedding in file embedding.emb
    X = np.loadtxt("%s.emb" % filepath, skiprows=1)  # load the embedding of the nodes of the graph

    plt.title(filepath)
    x = [X[i][1] for i in range(len(X))]
    y = [X[i][2] for i in range(len(X))]
    plt.scatter(x, y, color ="blue", marker=".", s=1)
    plt.savefig(filepath + "embeddingdrawing")


def color_clusters_in_graph(G_filepath, clusters, seed):
    G = nx.read_adjlist(G_filepath)
    pos = nx.spring_layout(G, seed=seed)
    options = {"node_size": 4}
    for i in range(len(clusters)):
        nx.draw_networkx_nodes(G, pos, nodelist=clusters[i], node_color=colors[i], **options) # colors defined above
    nx.draw_networkx_edges(G, pos, width=0.1)
    plt.savefig(G_filepath + "_postclustering")


def custom_graph_to_rdf(g, destination):
    with open(destination, 'x') as f:
        for triple in g.triples:
            subject_iri = g.index_to_iri[triple[0]]
            predicate_iri = g.index_to_iri_predicates[triple[1]]
            object_iri = g.index_to_iri[triple[2]]
            f.write(subject_iri + "\t" + predicate_iri + "\t" + object_iri + "\t.\n")
