from collections import defaultdict
import sys


class Graph():
    def __init__(self, size):
        self.edges = defaultdict(list)  # Dictionary of all connected nodes
        self.weights = {}  # Dictionary of edges and weights
        self.size = size
        self.dist = []
        for i in range(size):
            self.dist.append(sys.maxsize)  # Initialize distances to maxsize
        self.previous = []
        for i in range(size):
            self.previous.append(None)  # Initialize previous node as None

    def add_edge(self, from_node, to_node, weight):  # Bidirectional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

    def findSmallestNode(self):
        smallest = self.dist[self.getIndex(self.Q[0])]
        result = self.getIndex(self.Q[0])
        for i in range(len(self.dist)):
            if self.dist[i] < smallest:
                node = self.unpoppedQ[i]
                if node in self.Q:
                    smallest = self.dist[i]
                    result = self.getIndex(node)
        return result

    def getIndex(self, neighbour):
        for i in range(len(self.unpoppedQ)):
            if neighbour == self.unpoppedQ[i]:
                return i

    def getPopPosition(self, uNode):
        result = 0
        for i in range(len(self.Q)):
            if self.Q[i] == uNode:
                return i
        return result

    def getUnvisitedNodes(self, uNode):
        resultList = []
        allNeighbours = self.edges[uNode]
        for neighbour in allNeighbours:
            if neighbour in self.Q:
                resultList.append(neighbour)
        return resultList

    def dijsktra(self, start, end):
        self.Q = list(self.edges.keys())  # Create the list of unvisited nodes
        for i in range(len(self.Q)):
            if self.Q[i] == start:
                self.dist[i] = 0  # Set the start node distance to 0
        self.unpoppedQ = self.Q[0:]

        while self.Q:  # While there are unvisited nodes
            u = self.findSmallestNode()  # Get the unvisited node with the smallest distance
            if self.dist[u] == sys.maxsize:  # If the smallest distance is maxsize, break (no path)
                break
            if self.unpoppedQ[u] == end:  # If we reach the destination node
                break
            uNode = self.unpoppedQ[u]
            self.Q.remove(uNode)  # Pop the node from the unvisited list

            # Update distances for neighbors
            for neighbor in self.edges[uNode]:
                if neighbor in self.Q:  # Only update if the neighbor is still unvisited
                    alt = self.dist[self.getIndex(uNode)] + self.weights[(uNode, neighbor)]
                    if alt < self.dist[self.getIndex(neighbor)]:
                        self.dist[self.getIndex(neighbor)] = alt
                        self.previous[self.getIndex(neighbor)] = uNode  # Set previous node

        # Reconstruct the shortest path
        shortest_path = []
        shortest_path.insert(0, end)
        u = self.getIndex(end)
        while self.previous[u] is not None:
            shortest_path.insert(0, self.previous[u])  # Insert the previous node to the path
            u = self.getIndex(self.previous[u])

        # Return the shortest path along with the total cost
        return shortest_path, self.dist[self.getIndex(end)]


# Example usage
graph = Graph(8)

edges = [
    ('O', 'A', 2),
    ('O', 'B', 5),
    ('O', 'C', 4),
    ('A', 'B', 2),
    ('A', 'D', 7),
    ('A', 'F', 12),
    ('B', 'C', 1),
    ('B', 'D', 4),
    ('B', 'E', 3),
    ('C', 'E', 4),
    ('D', 'E', 1),
    ('D', 'T', 5),
    ('E', 'T', 7),
    ('F', 'T', 3),
]

for edge in edges:
    graph.add_edge(*edge)

path, cost = graph.dijsktra('O', 'T')
print(f"Shortest path: {path}, Cost: {cost}")
