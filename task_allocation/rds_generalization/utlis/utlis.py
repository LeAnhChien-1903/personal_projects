import math
import numpy as np
from types import NoneType
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import matplotlib.figure as figure
import matplotlib.colors as mcolors
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import heapq, random
from matplotlib.animation import FuncAnimation
import os
import torch
from copy import deepcopy
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# Seed
seed_value = 50
np.random.seed(seed= seed_value)
random.seed(seed_value)
torch.manual_seed(seed= seed_value)
# Line type
NONE_LINE = -1
DOUBLE_HORIZONTAL = 0
SINGLE_HORIZONTAL = 1
DOUBLE_VERTICAL = 2
SINGLE_VERTICAL = 3
MIX_LINE = 5
# Zone type
WORKING_ZONE = 0
STORAGE_ZONE = 1
WAITING_ZONE = 2
CHARGING_ZONE = 3
LINE_ZONE = 4
# Global graph vertex type
LINE_VERTEX = 0
WORKING_VERTEX = 1
STORAGE_VERTEX = 2
WAITING_VERTEX = 3
CHARGING_VERTEX = 4
ROBOT_VERTEX = 5
# Local graph vertex type
CENTER_GRAPH = 0
GOAL_NEIGHBOR = 1
FRONT_NEIGHBOR = 2
BACK_NEIGHBOR = 3
LEFT_NEIGHBOR = 4 
RIGHT_NEIGHBOR = 5
# Robot state
state_color = ['lime', 'olive', 'gold', 'chocolate', 'navy', 'cyan', 'orange', 'red', 'blue', 'purple', 'brown', 'black']
FREE = 0
ON_WAY_TO_START = 1
ON_WAY_TO_TARGET = 2
ON_WAY_TO_WAITING = 3
ON_WAY_TO_CHARGING = 4
PICKING_UP = 5
BUSY = 6
LOW_BATTERY = 7
CHARGING = 8
AVOIDANCE = 9
FOLLOWING = 10
DIE = 11
# FSM state for control
GO_TO_NEXT_POINT = 0
GO_TO_WAIT_POINT_JUNCTION = 1
GO_TO_WAIT_POINT_CLOSEST = 2
GOAL_OCCUPIED = 3
STOP = 4
STOP_BY_GOAL_OCCUPIED = 5
# Max distance between same point
MAX_SAME_DIST = 0.01
# Direction of a edge
UNDIRECTED = 0
POSITIVE_DIRECTED = 1
NEGATIVE_DIRECTED = -1
# Route type
NONE_ROUTE = -1
TO_START = 0
TO_TARGET = 1
TO_WAITING = 2
TO_CHARGING = 3
# Algorithm mode
TRAIN_MODE = True
TEST_MODE = False

def EuclidDistance(p1: np.ndarray, p2: np.ndarray):
    return round(math.hypot(p1[0] - p2[0], p1[1] - p2[1]), 2)

def isSamePoint(p1: np.ndarray, p2: np.ndarray):
    if EuclidDistance(p1, p2) < MAX_SAME_DIST:
        return True
    return False

def calculatePathCost(path: np.ndarray):
    cost = 0.0
    if path.shape[0] == 1:
        return cost
    for i in range(path.shape[0] - 1):
        cost += EuclidDistance(path[i], path[i+1])
    
    return round(cost, 2)

def ManhattanDistance(p1: np.ndarray, p2: np.ndarray):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) 

def normalizeAngle(angle: float):
    return round(math.atan2(math.sin(angle), math.cos(angle)), 2)

def angleByTwoPoint(p1: np.ndarray, p2: np.ndarray):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def calculateDifferenceOrientation(angle1: float, angle2: float):
    angle1 = normalizeAngle(angle1)
    angle2 = normalizeAngle(angle2)
    
    if 0 <= angle1 <= math.pi and 0 <= angle2 <= math.pi:
        return angle2 - angle1
    elif -math.pi < angle1 < 0 and -math.pi < angle2 < 0:
        return angle2 - angle1
    elif 0 <= angle1 <= math.pi and -math.pi < angle2 < 0:
        turn = angle2 - angle1
        if turn < -math.pi:
            turn += 2 * math.pi
        return turn
    elif -math.pi < angle1 < 0 and 0 <= angle2 <= math.pi:
        turn = angle2 - angle1
        if turn > math.pi:
            turn -= 2 * math.pi
        return turn
    
    return angle2 - angle1

def are_points_collinear(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray):
    # Calculate slopes
    slope1 = (point2[1] - point1[1]) * (point3[0] - point2[0])
    slope2 = (point3[1] - point2[1]) * (point2[0] - point1[0])
    
    # Check if slopes are equal (or close enough, considering floating-point arithmetic)
    return abs(slope1 - slope2) < 1e-10

def distanceBetweenPointAndLine(start_point: np.ndarray, end_point: np.ndarray, point: np.ndarray):
    px = end_point[0] - start_point[0]
    py = end_point[1] - start_point[1]

    norm = px*px + py*py
    if norm == 0.0: return EuclidDistance(start_point, point)

    u =  ((point[0] - start_point[0]) * px + (point[1] - start_point[1]) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = start_point[0] + u * px
    y = start_point[1] + u * py

    dx = x - point[0]
    dy = y - point[1]

    dist = (dx*dx + dy*dy)**.5

    return dist

def pointIsBetweenALine(start: np.ndarray, end: np.ndarray, point: np.ndarray): 
    if isSamePoint(point, end): return True
    distance_to_start = EuclidDistance(point, start)
    distance_to_end = EuclidDistance(point, end)
    segment_length = EuclidDistance(start, end)
    
    return  segment_length - 0.01 <= distance_to_start + distance_to_end <= segment_length + 0.01 and distance_to_start > 0.01

def find_intersection_point(line1_start: np.ndarray, line1_end: np.ndarray, 
                            line2_start: np.ndarray, line2_end: np.ndarray):
    """
    Finds the intersection point of two lines defined by two points each.
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    # Calculate the intersection point
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel
    else:
        intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return intersect_x, intersect_y

def check_line_segments_intersection_2d(line1_start: np.ndarray, line1_end: np.ndarray, 
                                        line2_start: np.ndarray, line2_end: np.ndarray):
    # http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    line1 = line1_end - line1_start
    line2 = line2_end - line2_start
    
    if isSamePoint(line1_end, line2_end) and isSamePoint(line1_start, line2_start) == False:
        return True, line1_end
    if isSamePoint(line1_end, line2_start) and isSamePoint(line1_start, line2_end) == False:
        return True, line1_end
    if isSamePoint(line1_start, line2_start) and isSamePoint(line1_end, line2_end) == False:
        return True, line1_start
    if isSamePoint(line1_start, line2_end) and isSamePoint(line1_end, line2_start) == False:
        return True, line1_start
    if are_points_collinear(line1_start, line1_end, line2_start) and  pointIsBetweenALine(line1_start, line1_end, line2_start):
        return True, line2_start
    if are_points_collinear(line1_start, line1_end, line2_end) and  pointIsBetweenALine(line1_start, line1_end, line2_end):
        return True, line2_end
    if are_points_collinear(line2_start, line2_end, line1_start) and  pointIsBetweenALine(line2_start, line2_end, line1_start):
        return True, line1_start
    if are_points_collinear(line2_start, line2_end, line1_end) and  pointIsBetweenALine(line2_start, line2_end, line1_end):
        return True, line1_end

    denom = line1[0] * line2[1] - line2[0] * line1[1]
    if denom == 0: 
        return False, np.zeros(2)  # Collinear
    denom_positive = denom > 0
    
    aux = line1_start - line2_start
    
    s_numer = line1[0] * aux[1] - line1[1] * aux[0]
    if (s_numer < 0) == denom_positive: 
        return False, np.zeros(2)  # No collision
    
    t_numer = line2[0] * aux[1] - line2[1] * aux[0]
    if (t_numer < 0) == denom_positive:  
        return False, np.zeros(2)  # No collision
    
    if ((s_numer > denom) == denom_positive) or ((t_numer > denom) == denom_positive): 
        return False, np.zeros(2)  # No collision
    
    # Otherwise collision detected
    t = t_numer / denom
    
    return True, line1_start + t * line1

def calculateRectangleCoordinate(center_x: float, center_y: float, angle: float, length: float, width: float):
    halfLength = length/2
    halfWidth = width/2
    sinAngle = math.sin(angle)
    cosAngle = math.cos(angle)
    agent_shape = []
    # Bottom left
    agent_shape.append([center_x + (cosAngle * -halfLength) - (sinAngle * halfWidth), 
                        center_y + (sinAngle * -halfLength) + (cosAngle * halfWidth)])
    # Top left corner
    agent_shape.append([center_x + (cosAngle * -halfLength) - (sinAngle * -halfWidth),
                        center_y + (sinAngle * -halfLength) + (cosAngle * -halfWidth)])
    # Top right 
    agent_shape.append([center_x + (cosAngle * halfLength) - (sinAngle * -halfWidth),
                        center_y + (sinAngle * halfLength) + (cosAngle * -halfWidth)])
    # Bottom right
    agent_shape.append([center_x + (cosAngle * halfLength) - (sinAngle * halfWidth), 
                        center_y + (sinAngle * halfLength) + (cosAngle * halfWidth)])
    # Bottom left
    agent_shape.append([center_x + (cosAngle * -halfLength) - (sinAngle * halfWidth), 
                        center_y + (sinAngle * -halfLength) + (cosAngle * halfWidth)])
    return agent_shape

class Line:
    def __init__(self, type_: int, center: np.ndarray, length: float, width: float):
        self.center: np.ndarray = center.copy()
        self.length: float = length
        self.width: float = width
        self.type_: int = type_
        if type_ == DOUBLE_HORIZONTAL or type_ == SINGLE_HORIZONTAL:
            self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], 0.0, length, width))
        else:
            self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], math.pi/2, length, width))
    def getCenterX(self):
        return self.center[0]
    def getCenterY(self):
        return self.center[1]
    def getCenter(self):
        return self.center.copy()
    def getLength(self):
        return self.length
    def getWidth(self):
        return self.width
    def getLineType(self):
        return self.type_

class Zone:
    def __init__(self, type_: int, center: np.ndarray, length: float, width: float):
        self.center: np.ndarray = center.copy()
        self.length: float = length
        self.width: float = width
        self.type_: int = type_
        self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], 0.0, length, width))
    def getCenterX(self):
        return self.center[0]
    def getCenterY(self):
        return self.center[1]
    def getCenter(self):
        return self.center.copy()
    def getLength(self):
        return self.length
    def getWidth(self):
        return self.width
    def getZoneType(self):
        return self.type_

class Point:
    def __init__(self, point_type: int, line_type: int, center: np.ndarray, length: float, width: float):
        self.center: np.ndarray = center.copy()
        self.length: float = length    
        self.width: float = width
        self.point_type: int = point_type
        self.line_type: int = line_type
        if line_type == DOUBLE_HORIZONTAL or line_type == SINGLE_HORIZONTAL:
            self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], math.pi/2, length, width))
        else:
            self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], 0.0, length, width))
            
    def copy(self):
        return Point(self.point_type, self.line_type, self.center.copy(), self.length, self.width)
    
    def getCenterX(self):
        return self.center[0]
    def getCenterY(self):
        return self.center[1]
    def getCenter(self):
        return self.center.copy()
    def getLength(self):
        return self.length
    def getWidth(self):
        return self.width
    def getPointType(self):
        return self.point_type
    def getLineType(self):
        return self.line_type

class Vertex:
    def __init__(self, id: int, type_: int, center: np.ndarray, h_direct: int = UNDIRECTED, 
                v_direct: int = UNDIRECTED, line_type: int = NONE_LINE):
        self.id: int = id
        self.type_: int = type_
        self.center: np.ndarray = center
        self.line_type: int = line_type
        self.h_direct: int = h_direct
        self.v_direct: int = v_direct
        self.neighbors: List[int] = []
    def copy(self):
        vertex = Vertex(self.id, self.type_, self.center.copy(), self.h_direct, self.v_direct, self.line_type)
        vertex.neighbors = self.neighbors.copy()
        return vertex
    def getID(self) -> int:
        return self.id
    def getCenterX(self) -> float:
        return self.center[0]
    def getCenterY(self) -> float:
        return self.center[1]
    def getCenter(self) -> np.ndarray:
        return self.center.copy()
    def getNeighbors(self) -> List[int]:
        return self.neighbors.copy()
    def addNeighbor(self, neighbor_id: int):
        self.neighbors.append(neighbor_id)
    def removeNeighbor(self, neighbor_id: int):
        self.neighbors.remove(neighbor_id)
    def getLineType(self) -> int:
        return self.line_type
    def getType(self) -> int:
        return self.type_
    def getHorizontalDirect(self) -> int: return self.h_direct
    def getVerticalDirect(self) -> int: return self.v_direct

class Edge:
    def __init__(self, id: int, start: Vertex, end: Vertex, edge_vel: float):
        self.id = id
        self.start: Vertex = start
        self.end: Vertex = end
        self.edge_vel: float = edge_vel
        self.edge_distance: float = math.hypot(self.end.center[0] - self.start.center[0], 
                                                self.end.center[1] - self.start.center[1])
    def copy(self):
        return Edge(self.id, self.start.copy(), self.end.copy(), self.edge_vel)

class GraphZone:
    def __init__(self, row_id: int, col_id: int, id: int, center: np.ndarray, length: float, width: float):
        self.row_id = row_id
        self.col_id = col_id
        self.id: int = id
        self.center: np.ndarray = center.copy()
        self.length: float = length
        self.width: float = width
        self.coords: np.ndarray = np.array(calculateRectangleCoordinate(center[0], center[1], 0.0, length, width))
        self.vertices: List[int] = []
    
    def copy(self):
        zone = GraphZone(self.row_id, self.col_id, self.id, self.center, self.length, self.width)
        zone.vertices = self.vertices.copy()
        return zone
    def addVertex(self, vertex: Vertex):
        condition1 = self.getCenterX() - self.length/2 <vertex.getCenterX() < self.getCenterX() + self.length/2 
        condition2 = self.getCenterY() - self.width/2 <vertex.getCenterY() < self.getCenterY() + self.width/2 
        if condition1 and condition2:
            self.vertices.append(vertex.getID())
            return True
        return False
    
    def getID(self): return self.id
    def getRowID(self): return self.row_id
    def getColID(self): return self.col_id
    def getCenterX(self): return self.center[0]
    def getCenterY(self): return self.center[1]
    def getCenter(self): return self.center.copy()
    def getCoords(self): return self.coords.copy()
    def getLength(self): return self.length
    def getWidth(self): return self.width
    def getVertices(self): return self.vertices.copy()
    

class Graph:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.zones: List[GraphZone] = []
        self.num_col: int = -1
        self.num_row: int = -1
    
    def copy(self):
        graph = Graph()
        graph.vertices = self.vertices.copy()
        graph.edges = self.edges.copy()
        graph.zones = self.zones.copy()
        graph.num_row = self.num_row
        graph.num_col = self.num_col
        return graph

    def createZone(self, map_center_x: float, map_center_y: float, map_length: float, 
                    map_width: float, num_in_cols: int, num_in_rows: int):
        self.num_col: int = num_in_cols
        self.num_row: int = num_in_rows
        zone_length = map_length / num_in_cols
        zone_width = map_width / num_in_rows  
        start_x = map_center_x - map_length/2 + zone_length/2
        start_y = map_center_y - map_width/2 + zone_width/2
        
        for col in range(num_in_cols):
            center_x = start_x + col * zone_length
            for row in range(num_in_rows):
                center_y = start_y + row * zone_width
                self.zones.append(GraphZone(row, col, row + col*num_in_rows, np.array([center_x, center_y]), zone_length, zone_width))
        
        for vertex in self.vertices:
            for zone in self.zones:
                if zone.addVertex(vertex):
                    break
    
    def getVertex(self, id: int):
        return self.vertices[id]

    def getNeighbor(self, id: int):
        return self.vertices[id].neighbors
    
    def getVertices(self):
        return self.vertices.copy()
    
    def getEdges(self):
        return self.edges.copy()   
    
    def getZones(self): 
        return self.zones.copy()

    def getZone(self, id):
        return self.zones[id]
    
    def getNumRow(self): return self.num_row
    
    def getNumCol(self): return self.num_col
    
    def addVertex(self, type_: int, center: np.ndarray, h_direct: int = UNDIRECTED, 
                    v_direct: int = UNDIRECTED, line_type: int = NONE_LINE):
        if len(self.vertices) == 0:
            self.vertices.append(Vertex(0, type_, center, h_direct, v_direct, line_type))
        else:
            self.vertices.append(Vertex(self.vertices[-1].id + 1, type_, center, h_direct, v_direct, line_type))

    def addEdge(self, v1_id: int, v2_id: int, edge_vel: float = 1.0):
        if len(self.edges) == 0:
            self.edges.append(Edge(id= 0, start = self.getVertex(v1_id), end= self.getVertex(v2_id), edge_vel = edge_vel))
            self.vertices[v1_id].neighbors.append(v2_id)
        else:
            self.edges.append(Edge(id= self.edges[-1].id, start = self.getVertex(v1_id), end= self.getVertex(v2_id), edge_vel = edge_vel))
            self.vertices[v1_id].neighbors.append(v2_id)
    
    def getNeighborAStar(self, id: int, target_id: int):
        neighbors = []
        for neighbor in self.vertices[id].neighbors:
            if neighbor != target_id and self.getVertex(neighbor).getType() == WORKING_VERTEX:
                continue
            neighbors.append((neighbor, EuclidDistance(self.vertices[id].getCenter(), self.vertices[neighbor].getCenter())))
        
        return neighbors
class Route:
    def __init__(self, route_type: int = None, route_vertices: List[int] = None, route_coords: np.ndarray = None): #type: ignore
        if type(route_vertices) == NoneType:
            self.is_route: bool = False
            self.type: int = NONE_ROUTE
            self.vertices = route_vertices
        else:
            self.type: int = route_type
            self.is_route: bool = True
            self.vertices: List[int] = route_vertices.copy()
        if type(route_coords) == NoneType:
            self.coords = route_coords
            self.cost: float = 0.0
        else:
            self.coords: np.ndarray = route_coords.copy()
            self.cost: float = calculatePathCost(route_coords)
            
    def copy(self):
        return Route(self.type, self.vertices, self.coords)
    
    def update(self, next_vertex: int, next_coord: np.ndarray):
        self.vertices.append(next_vertex)
        self.coords = np.append(self.coords, next_coord.reshape(1, -1), axis=0)

    def clearRoute(self): 
        self.is_route = False
        self.type = NONE_ROUTE
        
    def isRoute(self): return self.is_route
    def getType(self): return self.type
    def getLength(self): return self.coords.shape[0]
    def getVertices(self): return self.vertices.copy()
    def getRouteCost(self): return self.cost
    def getVertex(self, id: int): return self.vertices[id]
    def getCoords(self): return self.coords.copy()
    def getCoord(self, id: int): return self.coords[id]
    def getCoordXList(self): return self.coords[:, 0]
    def getCoordYList(self): return self.coords[:, 1]
    def getCoordX(self, id: int): return self.coords[id, 0]
    def getCoordY(self, id: int): return self.coords[id, 1]
class Task:
    def __init__(self, start: Vertex = None, target: Vertex = None, type_: int = None, priority: int = None, mass: float = None): #type: ignore
        if type(start) == NoneType:
            self.is_task: bool = False
            self.start = start
        else:
            self.is_task: bool = True
            self.start = start.copy()
        if type(target) == NoneType:
            self.target = target
        else:
            self.target = target.copy()
        self.type_: int = type_
        self.priority: int = priority
        self.mass: float = mass
        self.route: Route = Route()

    def copy(self): 
        task = Task(self.start, self.target, self.type_, self.priority, self.mass)
        task.setRoute(self.route)
        return task
    
    def isTask(self): return self.is_task
    def clearTask(self): self.is_task = False
    
    def getID(self) -> int: return self.id
    def setID(self, id: int): self.id = id
    def getStartID(self) -> int : return self.start.getID()
    def getTargetID(self) -> int: return self.target.getID()
    
    def getStartX(self) -> float: return self.start.getCenterX()
    def getStartY(self) -> float: return self.start.getCenterY()
    def getTargetX(self) -> float: return self.target.getCenterX()
    def getTargetY(self) -> float: return self.target.getCenterY()
    def getStartCenter(self) -> np.ndarray: return self.start.getCenter() 
    def getTargetCenter(self) -> np.ndarray: return self.target.getCenter()
    def getStart(self) -> Vertex: return self.start.copy()
    def getTarget(self) -> Vertex: return self.target.copy()
    
    def getType(self) -> int: return self.type_ 
    def getPriority(self) -> int: return self.priority
    def getMass(self) -> float: return self.mass
    
    def getRoute(self) -> Route: return self.route.copy()
    def getRouteCoords(self): return self.getRouteCoords()
    def getRouteCost(self)-> float: return self.route.getRouteCost()
    
    def setStart(self, start: Vertex): 
        self.is_task = True 
        self.start = start.copy()
    def setTarget(self, target: Vertex): self.target = target.copy()
    
    def setType(self, type_: int): self.type_ = type_
    def setPriority(self, priority: int): self.priority = priority
    def setMass(self, mass: float): self.mass = mass
    
    def setRoute(self, route: Route): self.route = route.copy()

class AllocationState:
    def __init__(self, selected_data: torch.Tensor = None, task_data: torch.Tensor = None, robot_data: torch.Tensor = None): #type: ignore
        if isinstance(selected_data, torch.Tensor):
            self.selected_data: torch.Tensor = selected_data.detach().clone()
        else: 
            self.selected_data = selected_data
        if isinstance(task_data, torch.Tensor):
            self.task_data: torch.Tensor = task_data.detach().clone()
        else: 
            self.task_data = task_data
        if isinstance(robot_data, torch.Tensor):
            self.robot_data: torch.Tensor = robot_data.detach().clone()
        else: 
            self.robot_data = robot_data
        
    def setData(self, selected_data: torch.Tensor, task_data: torch.Tensor, robot_data: torch.Tensor):
        self.selected_data = selected_data.detach().clone()
        self.task_data = task_data.detach().clone()
        self.robot_data = robot_data.detach().clone()
    
    def getSelectedData(self): return self.selected_data
    def getTaskData(self): return self.task_data
    def getRobotData(self): return self.robot_data
class PriorityQueue:
    """
        Implements a priority queue data structure. Each inserted item
        has a priority associated with it and the client is usually interested
        in quick retrieval of the lowest-priority item in the queue. This
        data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

def AStarPlanning(graph: Graph, type_: int, start_id: int, target_id: int) -> Route:
    prior_queue = PriorityQueue()
    prior_queue.push((start_id, [start_id], 0), 0)
    visited = []
    while not prior_queue.isEmpty():
        current_state, actions, cost = prior_queue.pop()
        if current_state not in visited:
            visited.append(current_state)
            if current_state == target_id:
                route = [[graph.getVertex(start_id).getCenterX(), graph.getVertex(start_id).getCenterY()]]
                vertices = [start_id]
                for i in range(1, len(actions) - 1):
                    angle1 = angleByTwoPoint(graph.getVertex(actions[i-1]).getCenter(), graph.getVertex(actions[i]).getCenter())
                    angle2 = angleByTwoPoint(graph.getVertex(actions[i]).getCenter(), graph.getVertex(actions[i+1]).getCenter())
                    if angle1 != angle2:    
                        vertices.append(graph.getVertex(actions[i]).getID())
                        route.append([graph.getVertex(actions[i]).getCenterX(), graph.getVertex(actions[i]).getCenterY()])
                vertices.append(target_id)
                route.append([graph.getVertex(target_id).getCenterX(), graph.getVertex(target_id).getCenterY()])
                
                return Route(type_, vertices, np.array(route))
            else:
                children = graph.getNeighborAStar(current_state, target_id)
                for child in children:
                    heuristic_value = ManhattanDistance(graph.getVertex(child[0]).getCenter(), graph.getVertex(target_id).getCenter())
                    prior_queue.update((child[0], actions + [child[0]], cost + child[1]), child[1] + cost + heuristic_value)
    return Route(NONE_ROUTE, [-1], np.array([None]))

def AStarPlanningCost(graph: Graph, start_id: int, target_id: int) -> float:
    prior_queue = PriorityQueue()
    prior_queue.push((start_id, [start_id], 0), 0)
    visited = []
    while not prior_queue.isEmpty():
        current_state, actions, cost = prior_queue.pop()
        if current_state not in visited:
            visited.append(current_state)
            if current_state == target_id:
                cost = 0.0
                for i in range(len(actions) - 1):
                    cost += EuclidDistance(graph.getVertex(actions[i]).getCenter(), graph.getVertex(actions[i+1]).getCenter())
                return cost
            else:
                children = graph.getNeighborAStar(current_state, target_id)
                for child in children:
                    heuristic_value = ManhattanDistance(graph.getVertex(child[0]).getCenter(), graph.getVertex(target_id).getCenter())
                    prior_queue.update((child[0], actions + [child[0]], cost + child[1]), child[1] + cost + heuristic_value)
    return -1.0

if __name__ == "__main__":
    pass
    