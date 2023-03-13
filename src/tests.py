import unittest
import graph
import algorithms


class TestsILP(unittest.TestCase):
	def test_outer_density1(self):
		g = graph.Graph()
		g.iri_to_index = {"a": 0, "b": 1, "c": 2}
		g.index_to_iri = ["a", "b", "c"]
		g.neighbors = [[1, 2], [0, 2], [0, 1]]
		set_x = [0, 1]
		set_y = [2]

		result = algorithms.outer_density(g, set_x, set_y)
		self.assertEqual(result, 3/2)

	def test_outer_density2(self):
		g = graph.Graph()
		g.iri_to_index = {"a": 0, "b": 1, "c": 2}
		g.index_to_iri = ["a", "b", "c"]
		g.neighbors = [[1, 2], [0, 2], [0, 1]]
		set_x = [0, 1, 2]
		set_y = [2]

		result = algorithms.outer_density(g, set_x, set_y)
		self.assertEqual(result, 3 / 2)

	def test_k_core1(self):
		g = graph.Graph()
		g.neighbors = [[1, 2], [0, 2, 5], [0, 1, 5, 4, 3, 6], [4, 2, 6, 7], [2, 6, 3, 7], [1, 2, 6, 8], [7, 3, 4, 2, 5, 8], [4, 3, 6], [5, 6]]

		result = algorithms.k_core(g, 3)
		self.assertEqual(len(result), 1)

	def test_k_core2(self):
		g = graph.Graph()
		g.neighbors = [[1, 2, 3], [0, 4, 7], [0], [0], [1, 5, 6, 7], [4, 6, 7], [4, 5, 7], [1, 4, 5, 6]]

		result = algorithms.k_core(g, 3)
		print(result[0])
		self.assertEqual(len(result[0]), 4)