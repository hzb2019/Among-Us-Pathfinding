"""
Name: hzb2019
CSE 331 FS21 (Onsay)
Project 5
"""

import heapq
import itertools
import math
import queue
import random
import time
import csv
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

#============== Modify Vertex Methods Below ==============#

    def degree(self) -> int:
        """
        The outgoing degree of this vertex (the number of values in its adjacency list)
        :return: The degree
        """
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        """
        Gets a set of tuples with this vertex's outgoing edges and their weights
        :return: The outgoing edges and their weights
        """
        edges = set()
        for edge in self.adj:
            edges.add((edge, self.adj[edge]))
        return edges

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Calculates the euclidean distance between this vertex and the given other Vertex
        :param other: The other Vertex to calculate distance between
        :return: The euclidean distance between the vertices
        """
        x = (self.x - other.x) ** 2
        y = (self.y - other.y) ** 2
        return np.sqrt(x + y)

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Calculates the taxicab distance between this vertex and the given other Vertex
        :param other: The other Vertex to calculate distance between
        :return: The taxicab distance between the vertices
        """
        x = abs(self.x - other.x)
        y = abs(self.y - other.y)
        return x + y
        pass


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)


    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        DO NOT MODFIY
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param start_id: unique string id of starting vertex
        :param dest_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(start_id) is None:
            self.vertices[start_id] = Vertex(start_id)
            self.size += 1
        if dest_id is not None:
            if self.vertices.get(dest_id) is None:
                self.vertices[dest_id] = Vertex(dest_id)
                self.size += 1
            self.vertices.get(start_id).adj[dest_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):         # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):         # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

#============== Modify Graph Methods Below ==============#

    def reset_vertices(self) -> None:
        """
        Resets the visited flag for all the vertices in the graph
        """
        for vertex in self.vertices:
            self.vertices[vertex].visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:
        """
        Gets the vertex with the provided ID. Returns None if the vertex is not found.
        :param vertex_id: The ID of the vertex we are looking for
        :return: The vertex with the given ID
        """
        if vertex_id in self.vertices:
            return self.vertices[vertex_id]
        return None

    def get_vertices(self) -> Set[Vertex]:
        """
        Gets a set of all the Vertex objects in the graph.
        :return: Set containing all Vertex objects in this graph
        """
        vertices = set()
        for vertex in self.vertices:
            vertices.add(self.vertices[vertex])
        return vertices

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:
        """
        Gets the weight of an edge given the start and destination vertices
        :param start_id: The ID of the Vertex that is the start of the edge
        :param dest_id: The ID of the Vertex that is the destination
        :return: Tuple containing the start ID, destination ID, and weight value
        """
        if not (start_id in self.vertices or dest_id in self.vertices):
            return None
        if not dest_id in self.vertices[start_id].adj:
            return None
        edge = self.vertices[start_id].adj[dest_id]
        return start_id, dest_id, edge

    def get_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Gets all the edges in the graph.
        :return: A set containing tuples representing edges in the graph.
        Each tuple is composed of a start vertex ID, end vertex ID, and the weight of the edge
        """
        edges = set()
        for vertex in self.vertices:
            for edge in self.vertices[vertex].adj:
                edges.add((vertex, edge, self.vertices[vertex].adj[edge]))
        return edges

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Breadth first search of a Vertex in the graph. Returns the path taken and the
        distance of the path
        :param start_id: The ID of the Vertex where we start searching
        :param target_id: The ID of the Vertex we are searching for
        :return: Tuple containing a list of IDs representing the path taken and the distance
        of the path taken
        """
        visit_queue = queue.SimpleQueue()
        visit_queue.put(([start_id], 0))
        distance = 0

        while not visit_queue.empty():
            current = visit_queue.get()
            if current[0][-1] not in self.vertices:
                continue

            if current[0][-1] == target_id:
                return current

            current_vertex = self.vertices[current[0][-1]]
            current_vertex.visited = True

            for adj in current_vertex.adj:
                if self.vertices[adj].visited is True:
                    continue
                # new distance = current distance + distance between current and adjacent
                new_distance = current[1]
                new_distance += self.get_edge(current[0][-1], adj)[2]

                # pretty sure this copy kills the complexity but I don't know how to get
                # around doing a BFS keeping track of each unique path without doing this
                new_path = current[0].copy()
                # new path = current path with adjacent appended
                new_path.append(adj)
                visit_queue.put((new_path, new_distance))

        return [], 0

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        """
        Depth first search of a Vertex in the graph. Returns the path taken and the
        distance of the path
        :param start_id: The ID of the Vertex where we start searching
        :param target_id: The ID of the Vertex we are searching for
        :return: Tuple containing a list of IDs representing the path taken and the distance
        of the path taken
        """
        visited = set()

        def dfs_inner(current_id: str, target_id: str,
                      path: List[str] = []) -> Tuple[List[str], float]:
            """
            Recursive portion of DFS algorithm, path is treated as a stack with only append
            and pop being called.
            :param current_id: The ID of the Vertex we are currently checking
            :param target_id: The ID of the Vertex we are searching for
            :param path: The current path of the DFS
            :return: A Tuple containing the search path and the distance the DFS took
            """
            # If it isn't in our set of vertices, return nothing
            if current_id not in self.vertices:
                return [], 0

            # If it has already been visited, do nothing
            if current_id in visited:
                return

            # If we find the target, add it to the path and return
            if current_id == target_id:
                path.append(current_id)
                return path, 0

            # Add the current vertex to the path and to the visited set
            path.append(current_id)
            visited.add(current_id)
            for adj in self.vertices[current_id].adj:
                # If the vertex has already been visited, skip it
                if adj in visited:
                    continue
                # Find the distance between current vertex and the adjacent one
                distance = self.get_edge(current_id, adj)[2]
                recurse = dfs_inner(adj, target_id, path)
                # The adjacent vertex will be in the path if it was found because it gets
                # added in its base case. Return the path and current distance + the
                # distances recursively
                if adj in recurse[0]:
                    return recurse[0], distance + recurse[1]

            # Return the path without the vertex if it is not in the subtree
            path.pop()
            return path, 0

        path = []
        return dfs_inner(start_id, target_id, path)

    def detect_cycle(self) -> bool:
        """
        Cycle detection implemented using a recursive DFS.
        :return: True if a cycle is found, False otherwise
        """
        visited = set()

        def detect_inner(current_id: str, path: List[str]) -> bool:
            """
            Recursive portion of cycle DFS algorithm, path is treated as a stack with only append
            and pop being called.
            :param current_id: The ID of the Vertex we are currently checking
            :param path: The current path of the DFS
            :return: A bool indicating whether or not a cycle has been found
            """

            # Add the current vertex to the path and to the visited set
            path.append(current_id)
            visited.add(current_id)

            for adj in self.vertices[current_id].adj:
                # If the vertex is in our current path, we have a cycle
                if adj in path:
                    return True

                # If the vertex has already been checked, we know it doesn't
                # have any cycles so it may be skipped
                if adj in visited:
                    continue

                recurse = detect_inner(adj, path)
                if recurse:
                    return True

            path.pop()
            return False

        path = []

        for vertex in self.vertices:
            detect = detect_inner(vertex, path)
            if detect:
                return True

        return False

    def a_star(self, start_id: str, target_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:
        """
        Performs an A* search to find an optimized path between Vertex objects in a directed
        weighted graph.
        :param start_id: The ID of the node to start at
        :param target_id: The ID of the node to search for
        :param metric: Callable object representing what distance calculations should be used
        :return: Tuple containing the list of visited Vertices from the search and the distance
        """

        if (start_id not in self.vertices) or (target_id not in self.vertices):
            return [], 0

        priority_queue = AStarPriorityQueue()
        path = [start_id]

        total_distances = dict()

        for vertex in self.vertices:
            total_distances[vertex] = float('inf')
            priority_queue.push(float('inf'), self.get_vertex(vertex))

        total_distances[start_id] = 0
        origin_vertices = dict()

        priority_queue.push(0, self.vertices[start_id])

        while not priority_queue.empty():
            vertex = priority_queue.pop()

            if vertex[1].id == target_id:
                break

            for adj in vertex[1].adj:

                new_distance = self.get_edge(adj, vertex[1].id)
                if new_distance is None:
                    continue

                new_distance = new_distance[2]

                if total_distances[adj] > total_distances[vertex[1].id] + new_distance:
                    total_distances[adj] = total_distances[vertex[1].id] + new_distance

                    priority = metric(self.vertices[adj], self.vertices[target_id])
                    priority += total_distances[adj]

                    priority_queue.update(priority, self.vertices[adj])

                    origin_vertices[adj] = vertex[1].id

        path.clear()

        distance = 0
        vertex = target_id
        while vertex != start_id:
            path.append(vertex)
            distance += self.get_edge(origin_vertices[vertex], vertex)[2]
            vertex = origin_vertices[vertex]

        path.append(start_id)

        path.reverse()

        return path, distance

class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
