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
	a = time.time()
	H = copy.deepcopy(G)
	b = time.time()
	print("Copying G: ", str(b - a))
	cutset = nx.minimum_node_cut(H)
	c = time.time()
	print("Fidning mininmum node cut: ", str(c - b))
	pos = nx.spring_layout(H, seed=3113794652)
	H.remove_nodes_from(cutset)
	comps, largest = algorithms.con_comps_and_largest(H)
	d = time.time()
	print("Find connected components: ", str(d - c))
	options = {"node_size": 4}
	nx.draw_networkx_nodes(H, pos, nodelist=comps[largest], node_color="tab:blue", **options)
	for i in range(4):
		if len(comps) > i and i != largest:
			nx.draw_networkx_nodes(H, pos, nodelist=comps[i], node_color="tab:green", **options)
	nx.draw_networkx_nodes(H, pos, nodelist=cutset, node_color="tab:red", **options)
	# edges
	nx.draw_networkx_edges(H, pos, width=0.1)
	plt.savefig("out/10m11cut%s" % timestamp)
	e = time.time()
	print("drawing figure: ", str(e - d))

	plt.clf()
	with open(file="out/comps.txt", mode="a") as f:
		f.write("\n" * 2)
		print("\n")
		f.write("largest component at %s is index " % timestamp + str(largest) + ". All components:\n")
		print("largest component at %s is index " % timestamp + str(largest) + ". All components:")
		for i in range(len(comps)):
			f.write(str(comps[i]) + "\n")
			print(str(comps[i]))

	return H, comps, largest


# reads graph files and draws them
# cores = nx.read_adjlist("out/kcores10m9_13-03-2023_18:40")
# nx.draw(cores, node_size=4, width=0.1)
# plt.savefig("out/kcores10m9_13-03-2023_18:40")
# plt.clf()
# for k in range(10, 16):
# 	a = time.time()
# 	cores = nx.read_adjlist("out/kcores10m%s_15-03-2023_19:09" % k)
# 	b = time.time()
# 	print("Read graph: " + str(b - a))
# 	nx.draw(cores, node_size=4, width=0.1)
# 	plt.savefig("out/kcores10m%s_15-03-2023_19:09" % k)
# 	c = time.time()
# 	print("Draw & Save figure: " + str(c - b))
# 	plt.clf()


# reads graph files and finds components, removes all but biggest
G = nx.read_adjlist("out/H10m11_18-03-2023_20:09")
starttime = time.time()
rt = 0
while rt < 3600*6:
	timestamp = time.strftime("%d-%m-%Y_%H:%M")
	G, comps, largest = minnodecut_draw(G)
	G = algorithms.remove_smaller_connected_components(G, comps, largest)
	nx.write_adjlist(G, "out/H10m11_%s" % timestamp)
	rt = time.time() - starttime









