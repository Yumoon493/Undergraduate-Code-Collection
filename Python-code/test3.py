"""
a)	What is a greedy algorithm? Explain, and provide one example.

Answers:
They make the best choice for the current step.
They can’t see the big picture – i.e. the final solution – until they get there.
The idea is that the final solution will be good enough.
But good enough does not necessarily mean optimal – optimal means the best possible solution.
Regardless, greedy algorithms can be optimal: this applies to Dijkstra.

So, the example: Shortest path - Dijkstra’s Algorithm
Choose node with the smallest value and remove it from list.
Identify all unvisited nodes connected to the visited node, and calculate the candidate distance.
Choose node with the smallest value; remove this node from list (becomes visited node).
Identify all unvisited nodes connected to the selected node and calculate the candidate distance.

It asks where the shortest path is without knowing the solution.
It is optimal, but it got there by making decision steps only.
"""

"""
b)	Describe the result of a greedy algorithm from the tree below
where the result should be the maximum sum of the nodes visited.
Describe what your greedy choice is and the result in respect to the optimal solution.

Answers:
Rules: (1) start with root; (2) select left or right connected node with highest value. 
So, 1: 5, 2: 8 in the right which is bigger, 3: 12 in the left
The result = 5 + 8 + 12 = 25 according to the greedy choice.
However, the rules of optimal solution: (1) start with root; (2) calculate all possible sums; (3) select the largest to return
Then the optimal solution is: 5 -> 4 -> 25, the result respect to it is: 5 + 4 + 25 = 34
"""

"""
(c)Using the code provided, complete a greedy algorithm implemented in Python
for the travelling salesman problem, using the graph and edge set shown.
The graph represents a set of cities.
The salesman has to visit every city, starting from and returning to the same city.
At each step, the algorithm should select the lowest cost path to the next city.
The program should output :
a list of the destinations in the order they were visited,
plus a cost, which should be the lowest possible given the algorithm.
"""
edges = 	[	('A','B',20),('A','C',35),('A','D',42),('A','E',19),('A','F',23),('A','G',11),
	 				('B','C',34),('B','D',30),('B','E',62),('B','F',18),('B','G',19),
 	 				('C','D',12),('C','E',20),('C','F',24),('C','G',26),
	 				('D','E',14),('D','F',11),('D','G',50),
	 				('E','F',52),('E','G',81),
	 				('F','G',16),
				]

from collections import defaultdict
import sys

'''
    Graph class for greedy TSP 
'''
class Graph():
    def __init__(self):
        self.edges = defaultdict(list)                          # dictionary of all connected nodes e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights = []                                       # list of edges and weights e.g. [['X', 'A', 7], ['X', 'B', 2], ...]
        self.dist = 0                                           # initialise variable for total distance of solution path
        self.result = []                                        # initialise variable for solution path
        '''
                create dictionary of edges and list of weights
            '''

    '''
            create dictionary of edges and list of weights
        '''

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)  # bidirectional dictionary of edges
        self.edges[to_node].append(from_node)
        self.weights.append([from_node, to_node, weight])  # bidirectional list of weights
        self.weights.append([to_node, from_node, weight])

    '''
        get index of next node with smallest cost
    '''

    def findSmallestNode(self, current_node):
        smallest = sys.maxsize  # initialise variable smallest
        for weight in self.weights:  # iterate through all weights in weight list
            if weight[0] == current_node:  # if first element of weight is current node
                if weight[1] in self.Q:     #if the neighbour is available
                    result_weight = self.get_weight(current_node, weight[1])       # distance from current node to neighbour
                    if result_weight < smallest:           # if distance is less then smallest value
                        smallest = result_weight      # smallest value is distance
                                                       # get index of neighbour
                        result = weight[1]                          # get distance
        self.dist = self.dist + smallest  # add to total distance of solution path
        return result  # return index of neighbour

    '''
        get position of neighbour in list of total set of nodes
    '''

    def getIndex(self, neighbour):
        for i in range(len(self.unpoppedQ)):
            if neighbour == self.unpoppedQ[i]:
                return i

    '''
            get position of current_node in list of available nodes
        '''

    def getPopPosition(self, current_node):
        result = 0
        for i in range(len(self.Q)):
            if self.Q[i] == current_node:
                return i
        return result

    '''
        find and return edge weight from current node to neighbour
    '''

    def get_weight(self, current_node, item):
        for weight in self.weights:
            if weight[0] == current_node and weight[1] == item:
                return weight[2]

    '''
        main function builds solution path and finds cost
    '''

    def greedy_TSP(self, start):
        self.Q = []  # build list of nodes
        for key in self.edges:
            self.Q.append(key)

        self.unpoppedQ = self.Q[0:]  # copy list of nodes
        current_node = start  # current node set to start
        self.result.append(current_node)  # append current node to solution path
        self.Q.pop(self.getPopPosition(start))  # remove from list of available nodes

        while self.Q != []:
            u = self.findSmallestNode(current_node)  # get the index of the node with the smallest distance
            current_node = self.unpoppedQ[u]  # get the node
            self.result.append(current_node)  # add the node to the solution path
            popPosition = self.getPopPosition(current_node)
            self.Q.pop(popPosition)  # remove the node from the set of available nodes

        self.dist = self.dist + self.get_weight(start, current_node)  # get total distance including return to start
        self.result.append(start)  # append start as end in solution path
        print('Path is:', self.result, end="")  # print solution path
        print(' and cost is:', self.dist)  # print total cost

graph = Graph()

edges = [('A', 'B', 20), ('A', 'C', 35), ('A', 'D', 42), ('A', 'E', 19), ('A', 'F', 23), ('A', 'G', 11),
             ('B', 'C', 34), ('B', 'D', 30), ('B', 'E', 62), ('B', 'F', 18), ('B', 'G', 19),
             ('C', 'D', 12), ('C', 'E', 20), ('C', 'F', 24), ('C', 'G', 26),
             ('D', 'E', 14), ('D', 'F', 11), ('D', 'G', 50),
             ('E', 'F', 52), ('E', 'G', 81),
             ('F', 'G', 16),
             ]

for edge in edges:
    graph.add_edge(*edge)

graph.greedy_TSP('B')