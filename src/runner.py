import numpy as np
import graph
import algorithms
import copy
import time
import scipy
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 700
import networkx as nx


timestamp = time.strftime("%d-%m-%Y_%H:%M")


def read_compute_print_kcores(fi="1m", k=9):
	a = time.time()
	g = graph.read_graph(filepath="/Users/tobiasbreuer/Desktop/E&OR/Building and Mining Knowledge Graphs/Project/yago_short%s.nt" % fi)
	G = algorithms.custom_to_networkX_graph(g)
	cores = algorithms.k_core1(G, k)
	b = time.time()
	print("calculating k-core: ", str(b - a))
	nx.write_adjlist(cores, "out/kcores%s%s_" % (fi, k) + timestamp)
	c = time.time()
	print("writing adjacency list: ", str(c - b))
	nx.draw(cores, node_size=5, width=0.2)
	plt.savefig("out/kcores%s%s_" % (fi, k) + timestamp)
	plt.clf()
	d = time.time()
	print("drawing figure: ", str(d - c))
	# reachable, non_reachable = algorithms.min_cut(c)


def minnodecut_draw(G):
	H = copy.deepcopy(G)
	cutset = nx.minimum_node_cut(H)
	pos = nx.spring_layout(H, seed=3113794652)
	H.remove_nodes_from(cutset)
	comps, largest = algorithms.con_comps_and_largest(H)
	options = {"node_size": 5}
	nx.draw_networkx_nodes(H, pos, nodelist=comps[0], node_color="tab:blue", **options)
	nx.draw_networkx_nodes(H, pos, nodelist=comps[1], node_color="tab:green", **options)
	for i in range(2, 4):
		if len(comps) > i:
			nx.draw_networkx_nodes(H, pos, nodelist=comps[i], node_color="tab:green", **options)
	nx.draw_networkx_nodes(H, pos, nodelist=cutset, node_color="tab:red", **options)
	# edges
	nx.draw_networkx_edges(H, pos, width=0.2)
	with open(file="out/comps.txt", mode="a") as f:
		print("\n"*2)
		print("largest component is index " + str(largest) + ":")
		for i in range(len(comps)):
			print(comps[i])

	return comps, largest



cores = nx.read_adjlist("out/kcores10m9_13-03-2023_18:40")
nx.draw(cores, node_size=4, width=0.1)
plt.savefig("out/kcores10m9_13-03-2023_18:40")
plt.clf()
for k in range(10, 16):
	a = time.time()
	cores = nx.read_adjlist("out/kcores10m%s_15-03-2023_19:09" % k)
	b = time.time()
	print("Read graph: " + str(b - a))
	nx.draw(cores, node_size=4, width=0.1)
	plt.savefig("out/kcores10m%s_15-03-2023_19:09" % k)
	c = time.time()
	print("Draw & Save figure: " + str(c - b))
	plt.clf()


# G = nx.read_adjlist("out/H")
# comps, largest = minnodecut_draw(G)
# H = algorithms.remove_smaller_connected_components(G, comps, largest)
# nx.write_adjlist(H, "out/H%s" % timestamp)









