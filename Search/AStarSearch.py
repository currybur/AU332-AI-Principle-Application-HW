# -*- coding: utf-8 -*-
from queue import LifoQueue
from queue import Queue
from queue import PriorityQueue
import math

class Graph:
    """
    Defines a graph with edges, each edge is treated as dictionary
    look up. function neighbors pass in an id and returns a list of 
    neighboring node
    
    """
    def __init__(self):
        self.edges = {}
        self.edgeWeights = {}
        self.locations = {}

    def neighbors(self, id):
        if id in self.edges:
            return self.edges[id]
        else:
            print("The node ", id , " is not in the graph")
            return False

    # this function get the g(n) the cost of going from from_node to 
    # the to_node
    def get_cost(self,from_node, to_node):
        #print("get_cost for ", from_node, to_node)
        nodeList = self.edges[from_node]
        #print(nodeList)
        try:
            edgeList = self.edgeWeights[from_node]
            return edgeList[nodeList.index(to_node)]
        except ValueError:
            print("From node ", from_node, " to ", to_node, " does not exist a direct connection")
            return False


def reconstruct_path(came_from, start, goal):
    """
    Given a dictionary of came_from where its key is the node 
    character and its value is the parent node, the start node
    and the goal node, compute the path from start to the end

    Arguments:
    came_from -- a dictionary indicating for each node as the key and 
                 value is its parent node
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    path. -- A list storing the path from start to goal. Please check 
             the order of the path should from the start node to the 
             goal node
    """
    path = []
    ### START CODE HERE ### (≈ 6 line of code)
    current_node = goal
    while current_node != start:
        path.append(current_node)
        if not came_from:
            return None
        current_node = came_from[current_node]
    path.insert(0,start)
    ### END CODE HERE ###
    return path

def heuristic(graph, current_node, goal_node):
    """
    Given a graph, a start node and a next nodee
    returns the heuristic value for going from current node to goal node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list
             of other nodes
    current_node -- A character indicating the current node
    goal_node --  A character indicating the goal node

    Return:
    heuristic_value of going from current node to goal node
    """
    heuristic_value = 0
    ### START CODE HERE ### (≈ 15 line of code)
    x1, y1 = graph.locations[current_node]
    x2, y2 = graph.locations[goal_node]
    heuristic_value = abs(x1-x2) + abs(y1 - y2)
    ### END CODE HERE ###
    return heuristic_value

def A_star_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize A* search algorithm by finding the path from 
    start node to the goal node
    Use early stoping in your code
    This function returns back a dictionary storing the information of each node
    and its corresponding parent node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list 
             of other nodes
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    came_from -- a dictionary indicating for each node as the key and 
                value is its parent node
    """
    
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    #step = 1
    ### START CODE HERE ### (≈ 15 line of code)
    flag = [0]*2
    for e in graph.edges.values():
        if start in e or start in graph.edges.keys():
            flag[0] = 1
        if goal in e or goal in graph.edges.keys():
            flag[1] = 1
    if 0 in flag:
        if flag.count(0) == 2:
            print("Start and goal not included in graph, please check your setting.")
        else:
            print("%s not included in graph, please check your setting."%["Start","Goal"][flag.index(0)])
        return None
    my_queue = PriorityQueue()
    for son in graph.neighbors(start):
        my_queue.put((graph.get_cost(start,son)+heuristic(graph, son, goal),start,son))
    while not my_queue.empty():
        cost, parent, son = my_queue.get()
        #fringe.remove((cost, parent, son))
        if son not in came_from.keys():
            if son == goal:
                cost_so_far[son] = cost_so_far[parent] + graph.get_cost(parent,son)
                came_from[son] = parent
                return came_from, cost_so_far
            cost_so_far[son] = cost_so_far[parent] + graph.get_cost(parent,son)
            came_from[son] = parent
            for s in graph.neighbors(son):
                if s==parent:
                    continue
                my_queue.put((graph.get_cost(son,s)+cost_so_far[son]+heuristic(graph, s, goal), son, s))
    ### END CODE HERE ###
    return came_from, cost_so_far

def consistency_verification(graph, heuristic, goal):
    """
    For extra credits 2. Check consistency.
    :param graph:
    :param heuristic:
    :param goal:
    :return: whether consistent.
    """
    for current_node in graph.edges:
        for kid in graph.neighbors(current_node):
            if heuristic(graph, current_node, goal) > graph.get_cost(current_node,kid) + heuristic(graph, kid, goal):
                print("Not consistent ! ")
                return False
    return True

def t3graph():
    """
    To generate the graph in question 3.

    :return:
    """
    def on_edge_or_corner(x,y):
        pos = []
        if x==0:
            pos.append("l")
        if x==6:
            pos.append("r")
        if y==0:
            pos.append("t")
        if y==4:
            pos.append("b")
        return pos

    new_graph = Graph()
    for x in range(7):
        for y in range(5):
            if (x,y)==(3,1) or (x,y)==(3,2) or (x,y)==(3,3):
                continue
            node = (x,y)
            edges = []
            pos = on_edge_or_corner(x,y)
            if "l" not in pos:
                edges.append((x-1,y))
            if "r"not in pos:
                edges.append((x+1,y))
            if "t" not in pos:
                edges.append((x,y-1))
            if "b" not in pos:
                edges.append((x,y+1))
            for obstc in [(3,1),(3,2),(3,3)]:
                if obstc in edges:
                    edges.remove(obstc)
            new_graph.edges[node] = edges
            new_graph.edgeWeights[node] = [1]*len(edges)
            new_graph.locations[node] = list(node)
    #print(new_graph.edges)
    return new_graph

# The main function will first create the graph, then use A* search
# which will return the came_from dictionary 
# then use the reconstruct path function to rebuild the path.
if __name__=="__main__":

    small_graph = Graph()
    small_graph.edges = {
        'A': ['B','D'],
        'B': ['A', 'C', 'D'],
        'C': ['A'],
        'D': ['E', 'A'],
        'E': ['B']
    }
    small_graph.edgeWeights={
        'A': [2,4],
        'B': [2, 3, 4],
        'C': [2],
        'D': [3, 4],
        'E': [5]
    }
    small_graph.locations={
        'A': [4,4],
        'B': [2,4],
        'C': [0,0],
        'D': [6,2],
        'E': [8,0]
    }

    large_graph = Graph()
    large_graph.edges = {
        'S': ['A','B','C'],
        'A': ['S','B','D'],
        'B': ['S', 'A', 'D','H'],
        'C': ['S','L'],
        'D': ['A', 'B','F'],
        'E': ['G','K'],
        'F': ['H','D'],
        'G': ['H','E'],
        'H': ['B','F','G'],
        'I': ['L','J','K'],
        'J': ['L','I','K'],
        'K': ['I','J','E'],
        'L': ['C','I','J']
    }
    large_graph.edgeWeights = {
        'S': [7, 2, 3],
        'A': [7, 3, 4],
        'B': [2, 3, 4, 1],
        'C': [3, 2],
        'D': [4, 4, 5],
        'E': [2, 5],
        'F': [3, 5],
        'G': [2, 2],
        'H': [1, 3, 2],
        'I': [4, 6, 4],
        'J': [4, 6, 4],
        'K': [4, 4, 5],
        'L': [2, 4, 4]
    }

    large_graph.locations = {
        'S': [0, 0],
        'A': [-2,-2],
        'B': [1,-2],
        'C': [6,0],
        'D': [0,-4],
        'E': [6,-8],
        'F': [1,-7],
        'G': [3,-7],
        'H': [2,-5],
        'I': [4,-4],
        'J': [8,-4],
        'K': [6,-7],
        'L': [7,-3]
    }
    print("Small Graph")
    start = 'A'
    goal = 'E'
    came_from_Astar, cost_so_far = A_star_search(small_graph, start, goal)
    print("came from Astar " , came_from_Astar)
    print("cost form Astar ", cost_so_far)
    pathAstar = reconstruct_path(came_from_Astar, start, goal)
    print("path from Astar ", pathAstar)


    print("Large Graph")
    start = 'S'
    goal = 'E'
    came_from_Astar, cost_so_far = A_star_search(large_graph, start, goal)
    print("came from Astar " , came_from_Astar)
    print("cost form Astar ", cost_so_far)
    pathAstar = reconstruct_path(came_from_Astar, start, goal)
    print("path from Astar ", pathAstar)

    #consistency_verification(small_graph, heuristic, 'E')
    """"
    t3graph = t3graph()
    print("T3 Graph")
    start = (1,2)
    goal = (5,2)
    came_from_Astar, cost_so_far = A_star_search(t3graph, start, goal)
    print("came from Astar " , came_from_Astar.keys())
    print("cost form Astar ", cost_so_far)
    pathAstar = reconstruct_path(came_from_Astar, start, goal)
    print("path from Astar ", pathAstar)
    """
