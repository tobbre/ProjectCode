import numpy as np
import graph
import algorithms
import copy
import time
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 700
import scipy
import networkx as nx

a = time.time()
g = graph.read_graph(filepath="/Users/tobiasbreuer/Desktop/E&OR/Building and Mining Knowledge Graphs/Project/yago_short100k.nt")
k = 5
c = algorithms.k_core1(g, k)
b = time.time()
timestamp = time.strftime("%d-%m-%Y_%H:%M")
nx.write_adjlist(c, "out/kcores%s_" % k + timestamp)
print(b - a)
nx.draw(c, node_size=5, width=0.2)
plt.show()
