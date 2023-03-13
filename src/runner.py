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
g = graph.read_graph(filepath="/Users/tobiasbreuer/Desktop/E&OR/Building and Mining Knowledge Graphs/Project/yago_short10m.nt")
c = algorithms.k_core1(g, 17)
b = time.time()
print(b - a)
nx.draw(c, node_size=5, width=0.2)
plt.show()
