import networkx as nx
import collections

#ordered directed graph class so that we can retrieve edges in order
#all edges must be inserted in order
class OrderedGraph(nx.DiGraph):
    adjlist_dict_factory = collections.OrderedDict

"""
g = OrderedGraph()
g.add_edge(1, 3, score=17)
g.add_edge(1, 2, score=42)
g.add_edge(1, 4, score=55)
print(g.neighbors(1))
print(g.neighbors(2))"""