from utlis.utlis import *

class FactoryMap:
    def __init__(self, data_folder: str, single_line_width: float = 0.9, double_line_width: float = 1.8, point_line_length: float = 1.2):
        self.single_line_width = single_line_width
        self.double_line_width = double_line_width
        self.point_line_length = point_line_length
        self.data_folder = data_folder
        self.readMapData()
    
    def readMapData(self):
        self.lines: List[Line] = []
        self.zones: List[Zone] = []
        self.zone_points: List[Point] = []
        self.double_horizontal_lines: List[Line] = []
        self.single_horizontal_lines: List[Line] = []
        self.double_vertical_lines: List[Line] = []
        self.single_vertical_lines: List[Line] = []
        self.working_points: List[Point] = []   
        self.storage_points: List[Point] = []   
        self.waiting_points: List[Point] = []
        self.charging_points: List[Point] = []  
        
        zone_point_centers: List[List] = []

        map_data = np.loadtxt(self.data_folder + "/map.txt")
        self.map_center = map_data[0:2]
        self.map_length = map_data[2]
        self.map_width = map_data[3]
        self.map_coords = np.array(calculateRectangleCoordinate(map_data[0], map_data[1], 0.0, map_data[2], map_data[3]))
        for line in np.loadtxt(self.data_folder + "/line_data.txt"):
            self.addLine(int(line[0]), line[1:3], line[-1])
        # for zone in np.loadtxt(self.data_folder + "/zone_data.txt"):
        #     self.zones.append(Zone(int(zone[0]), zone[1:3], zone[3], zone[4])) 
        for point in np.loadtxt(self.data_folder + "/point_data.txt"):
            self.addZonePoint(int(point[0]), int(point[1]), point[2:4], point[4], point[5])
            zone_point_centers.append([self.zone_points[-1].getCenterX(), self.zone_points[-1].getCenterY()])
        
        self.zone_point_centers = np.array(zone_point_centers)
        self.graph_point: List[List[float]] = []
        self.calculateWaitingZoneBridge()
        self.createFactoryGraphMap()
        
    def createFactoryGraphMap(self):
        self.graph = Graph()
        self.double_vertical_vertices: List[List[Vertex]] = [[] for _ in range(len(self.double_vertical_lines))]
        self.single_vertical_vertices: List[List[Vertex]] = [[] for _ in range(len(self.single_vertical_lines))]
        self.double_horizontal_vertices: List[List[Vertex]] = [[] for _ in range(len(self.double_horizontal_lines))]
        self.single_horizontal_vertices: List[List[Vertex]] = [[] for _ in range(len(self.single_horizontal_lines))]
        self.zone_vertices: List[Vertex] = []
        
        self.getGraphVertices()
        self.getGraphEdges()
        
    def getGraphVertices(self):
        self.getGraphVerticesFromZonePoints()
        self.getVertexInDoubleHorizontalLine()
        self.getVertexInSingleHorizontalLine() 
        self.getVertexInDoubleVerticalLine()
        self.getVertexInSingleVerticalLine()
        self.getVertexInIntersectionLine()

    def getGraphEdges(self):
        self.getGraphEdgesBetweenZoneLines()
        self.getGraphEdgeInDoubleHorizontalLine()
        self.getGraphEdgeInDoubleVerticalLine()
        self.getGraphEdgeInSingleHorizontalLine()
        self.getGraphEdgeInSingleVerticalLine()
    
    def getGraphVerticesFromZonePoints(self):
        self.waiting_vertices: List[List[Vertex]] = [[], []]
        for point in self.working_points:
            self.graph.addVertex(WORKING_VERTEX, point.center)
            self.zone_vertices.append(self.graph.getVertex(-1))
            self.graph_point.append([point.center[0], point.center[1]])
        for point in self.storage_points:
            self.graph.addVertex(STORAGE_VERTEX, point.center)
            self.zone_vertices.append(self.graph.getVertex(-1))
            self.graph_point.append([point.center[0], point.center[1]])
        for point in self.waiting_points:
            self.graph.addVertex(WAITING_VERTEX, point.center)
            self.zone_vertices.append(self.graph.getVertex(-1))
            self.graph_point.append([point.center[0], point.center[1]])
            if point.getCenterY() < 0: 
                self.waiting_vertices[0].append(self.graph.getVertex(-1))
            else:
                self.waiting_vertices[1].append(self.graph.getVertex(-1))
        for point in self.charging_points:
            self.graph.addVertex(CHARGING_VERTEX, point.center)
            self.zone_vertices.append(self.graph.getVertex(-1))
            self.graph_point.append([point.center[0], point.center[1]])
            
    def getVertexInDoubleHorizontalLine(self):
        for i in range(len(self.double_horizontal_lines)):
            line = self.double_horizontal_lines[i]
            origin_y1 = line.getCenterY() - self.double_line_width/4
            origin_y2 = line.getCenterY() + self.double_line_width/4
            point_y1 = round(line.getCenterY() + self.double_line_width/2 + self.working_points[0].getLength() * 0.5, 2)
            point_y2 = round(line.getCenterY() - self.double_line_width/2 - self.working_points[0].getLength() * 0.5, 2)
            y1 = np.where(np.round(self.zone_point_centers[:, 1], 2) == point_y1)
            y2 = np.where(np.round(self.zone_point_centers[:, 1], 2) == point_y2)   
            for idx in y1[0]:
                if line.getCenterX() - line.getLength()/2 <= self.zone_points[idx].getCenterX() <= line.getCenterX() + line.getLength()/2:
                    if [self.zone_points[idx].getCenterX(), origin_y1] not in self.graph_point:
                        self.graph_point.append([self.zone_points[idx].getCenterX(), origin_y1])
                        self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), origin_y1]), 
                                            POSITIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL)
                        self.double_horizontal_vertices[i].append(self.graph.getVertex(-1))
                        self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), origin_y2]), 
                                            NEGATIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL)
                        self.double_horizontal_vertices[i].append(self.graph.getVertex(-1))
            for idx in y2[0]:
                if line.getCenterX() - line.getLength()/2 <= self.zone_points[idx].getCenterX() <= line.getCenterX() + line.getLength()/2:
                    if [self.zone_points[idx].getCenterX(), origin_y1] not in self.graph_point:
                        self.graph_point.append([self.zone_points[idx].getCenterX(), origin_y1])
                        self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), origin_y1]), 
                                            POSITIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL)
                        self.double_horizontal_vertices[i].append(self.graph.getVertex(-1))
                        self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), origin_y2]), 
                                            NEGATIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL)
                        self.double_horizontal_vertices[i].append(self.graph.getVertex(-1))

    def getVertexInSingleHorizontalLine(self):
        for i in range(len(self.single_horizontal_lines)):
            line = self.single_horizontal_lines[i]
            point_y1 = round(line.getCenterY() + self.single_line_width/2 + self.working_points[0].getLength()/2, 2)
            point_y2 = round(line.getCenterY() - self.single_line_width/2 - self.working_points[0].getLength()/2, 2)
            y1 = np.where(np.round(self.zone_point_centers[:, 1], 2) == point_y1)
            y2 = np.where(np.round(self.zone_point_centers[:, 1], 2) == point_y2)
            for idx in y1[0]:
                if line.getCenterX() - line.getLength()/2 <= self.zone_point_centers[idx, 0] <= line.getCenterX() + line.getLength()/2:
                    if [self.zone_points[idx].getCenterX(), line.getCenterY()] not in self.graph_point:
                        self.graph_point.append([self.zone_points[idx].getCenterX(), line.getCenterY()])
                    self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), line.getCenterY()]), 
                                        UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL)
                    self.single_horizontal_vertices[i].append(self.graph.getVertex(-1))
            for idx in y2[0]:
                if line.getCenterX() - line.getLength()/2 <= self.zone_point_centers[idx, 0] <= line.getCenterX() + line.getLength()/2:
                    if [self.zone_points[idx].getCenterX(), line.getCenterY()] not in self.graph_point:
                        self.graph_point.append([self.zone_points[idx].getCenterX(), line.getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([self.zone_points[idx].getCenterX(), line.getCenterY()]), 
                                            UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL)
                        self.single_horizontal_vertices[i].append(self.graph.getVertex(-1))
    
    def getVertexInDoubleVerticalLine(self):
        for i in range(len(self.double_vertical_lines)):
            line = self.double_vertical_lines[i]
            origin_x1 = line.getCenterX() - self.double_line_width/4
            origin_x2 = line.getCenterX() + self.double_line_width/4 
            point_x1 = round(line.getCenterX() + self.double_line_width/2 + self.working_points[0].getLength() * 0.5, 2)
            point_x2 = round(line.getCenterX() - self.double_line_width/2 - self.working_points[0].getLength() * 0.5, 2)
            x1 = np.where(np.round(self.zone_point_centers[:, 0], 2) == point_x1)
            x2 = np.where(np.round(self.zone_point_centers[:, 0], 2) == point_x2)
            for idx in x1[0]:
                if line.getCenterY() - line.getLength()/2 <= self.zone_points[idx].getCenterY() <= line.getCenterY() + line.getLength()/2:
                    if [origin_x1, self.zone_points[idx].getCenterY()] not in self.graph_point:
                        self.graph_point.append([origin_x1, self.zone_points[idx].getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([origin_x1, self.zone_points[idx].getCenterY()]), 
                                            UNDIRECTED, POSITIVE_DIRECTED, DOUBLE_VERTICAL)
                        self.double_vertical_vertices[i].append(self.graph.getVertex(-1))
                        self.graph.addVertex(LINE_VERTEX, np.array([origin_x2, self.zone_points[idx].getCenterY()]), 
                                            UNDIRECTED, NEGATIVE_DIRECTED, DOUBLE_VERTICAL)
                        self.double_vertical_vertices[i].append(self.graph.getVertex(-1))
            for idx in x2[0]:
                if line.getCenterY() - line.getLength()/2 <= self.zone_points[idx].getCenterY() <= line.getCenterY() + line.getLength()/2:
                    if [origin_x1, self.zone_points[idx].getCenterY()] not in self.graph_point:
                        self.graph_point.append([origin_x1, self.zone_points[idx].getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([origin_x1, self.zone_points[idx].getCenterY()]), 
                                            UNDIRECTED, POSITIVE_DIRECTED, DOUBLE_VERTICAL)
                        self.double_vertical_vertices[i].append(self.graph.getVertex(-1))
                        self.graph.addVertex(LINE_VERTEX, np.array([origin_x2, self.zone_points[idx].getCenterY()]), 
                                            UNDIRECTED, NEGATIVE_DIRECTED, DOUBLE_VERTICAL)
                        self.double_vertical_vertices[i].append(self.graph.getVertex(-1))
            
    def getVertexInSingleVerticalLine(self):
        for i in range(len(self.single_vertical_lines)):
            line = self.single_vertical_lines[i]
            point_x1 = round(line.getCenterX() + self.single_line_width/2 + self.working_points[0].getLength() /2, 2)
            point_x2 = round(line.getCenterX() - self.single_line_width/2 - self.working_points[0].getLength() /2, 2)
            x1 = np.where(np.round(self.zone_point_centers[:, 0], 2) == point_x1)
            x2 = np.where(np.round(self.zone_point_centers[:, 0], 2) == point_x2)
            for idx in x1[0]:
                if line.getCenterY() - line.getLength()/2 <= self.zone_point_centers[idx, 1] <= line.getCenterY() + line.getLength()/2:
                    if [line.getCenterX(), self.zone_points[idx].getCenterY()] not in self.graph_point:
                        self.graph_point.append([line.getCenterX(), self.zone_points[idx].getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([line.getCenterX(), self.zone_points[idx].getCenterY()]),
                                            UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL)
                        self.single_vertical_vertices[i].append(self.graph.getVertex(-1))
            for idx in x2[0]:
                if line.getCenterY() - line.getLength()/2 <= self.zone_point_centers[idx, 1] <= line.getCenterY() + line.getLength()/2:
                    if [line.getCenterX(), self.zone_points[idx].getCenterY()] not in self.graph_point:    
                        self.graph_point.append([line.getCenterX(), self.zone_points[idx].getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([line.getCenterX(), self.zone_points[idx].getCenterY()]),
                                            UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL)
                        self.single_vertical_vertices[i].append(self.graph.getVertex(-1))
    
    def getVertexInIntersectionLine(self):
        for i in range(len(self.double_vertical_lines)):
            vertical_line = self.double_vertical_lines[i]
            x1 = vertical_line.getCenterX() - self.double_line_width/4
            x2 = vertical_line.getCenterX() + self.double_line_width/4
            for j in range(len(self.single_horizontal_lines)):
                single_line = self.single_horizontal_lines[j]
                if single_line.getCenterX() - single_line.getLength()/2 - self.double_line_width <= x1 <= single_line.getCenterX() + single_line.getLength()/2 + self.double_line_width:
                    if [x1, single_line.getCenterY()] not in self.graph_point:
                        self.graph_point.append([x1, single_line.getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([x1, single_line.getCenterY()]),
                                            UNDIRECTED, POSITIVE_DIRECTED, MIX_LINE)
                        self.single_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        UNDIRECTED, POSITIVE_DIRECTED, SINGLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        UNDIRECTED, POSITIVE_DIRECTED, DOUBLE_VERTICAL))
                if single_line.getCenterX() - single_line.getLength()/2 - self.double_line_width <= x2 <= single_line.getCenterX() + single_line.getLength()/2 + self.double_line_width:
                    if [x2, single_line.getCenterY()] not in self.graph_point:
                        self.graph_point.append([x2, single_line.getCenterY()])
                        self.graph.addVertex(LINE_VERTEX, np.array([x2, single_line.getCenterY()]), 
                                            UNDIRECTED, NEGATIVE_DIRECTED, MIX_LINE)
                        self.single_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        UNDIRECTED, NEGATIVE_DIRECTED, SINGLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        UNDIRECTED, NEGATIVE_DIRECTED, DOUBLE_VERTICAL))
            for j in range(len(self.double_horizontal_lines)):
                double_line = self.double_horizontal_lines[j]
                if double_line.getCenterX() - double_line.getLength()/2 - self.double_line_width <= x1 <= double_line.getCenterX() + double_line.getLength()/2 + self.double_line_width:
                    if [x1, double_line.getCenterY() - self.double_line_width/4] not in self.graph_point:
                        self.graph_point.append([x1, double_line.getCenterY() - self.double_line_width/4])
                        self.graph.addVertex(LINE_VERTEX, np.array([x1, double_line.getCenterY() - self.double_line_width/4]), 
                                            POSITIVE_DIRECTED, POSITIVE_DIRECTED, MIX_LINE)
                        self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(),
                                                                        POSITIVE_DIRECTED, POSITIVE_DIRECTED, DOUBLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        POSITIVE_DIRECTED, POSITIVE_DIRECTED, DOUBLE_VERTICAL))
                    
                    if [x1, double_line.getCenterY() + self.double_line_width/4] not in self.graph_point:
                        self.graph_point.append([x1, double_line.getCenterY() + self.double_line_width/4])
                        self.graph.addVertex(LINE_VERTEX, np.array([x1, double_line.getCenterY() + self.double_line_width/4]), 
                                            NEGATIVE_DIRECTED, UNDIRECTED, MIX_LINE)
                        self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        NEGATIVE_DIRECTED, POSITIVE_DIRECTED, DOUBLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(),
                                                                        NEGATIVE_DIRECTED, POSITIVE_DIRECTED, DOUBLE_VERTICAL))
                
                if double_line.getCenterX() - double_line.getLength()/2 - self.double_line_width <= x2 <= double_line.getCenterX() + double_line.getLength()/2 + self.double_line_width:
                    if [x2, double_line.getCenterY() - self.double_line_width/4] not in self.graph_point:
                        self.graph_point.append([x2, double_line.getCenterY() - self.double_line_width/4])
                        self.graph.addVertex(LINE_VERTEX, np.array([x2, double_line.getCenterY() - self.double_line_width/4]), 
                                            POSITIVE_DIRECTED, NEGATIVE_DIRECTED, MIX_LINE)
                        self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(),
                                                                        POSITIVE_DIRECTED, NEGATIVE_DIRECTED, DOUBLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        POSITIVE_DIRECTED, NEGATIVE_DIRECTED, DOUBLE_VERTICAL))
                    
                    if [x2, double_line.getCenterY() + self.double_line_width/4] not in self.graph_point:
                        self.graph_point.append([x2, double_line.getCenterY() + self.double_line_width/4])
                        self.graph.addVertex(LINE_VERTEX, np.array([x2, double_line.getCenterY() + self.double_line_width/4]), 
                                            NEGATIVE_DIRECTED, NEGATIVE_DIRECTED, MIX_LINE)
                        self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        NEGATIVE_DIRECTED, NEGATIVE_DIRECTED, DOUBLE_HORIZONTAL))
                        self.double_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                        self.graph.getVertex(-1).getCenter(), 
                                                                        NEGATIVE_DIRECTED, NEGATIVE_DIRECTED, DOUBLE_VERTICAL))
        
        for i in range(len(self.single_vertical_lines)):
            vertical_line = self.single_vertical_lines[i]
            for j in range(len(self.single_horizontal_lines)):
                single_line = self.single_horizontal_lines[j]
                if vertical_line.getCenterY() - vertical_line.getLength()/2 - self.single_line_width <= single_line.getCenterY() <= vertical_line.getCenterY() + vertical_line.getLength()/2 + self.single_line_width:
                    if single_line.getCenterX() - single_line.getLength()/2 - self.double_line_width <= vertical_line.getCenterX() <= single_line.getCenterX() + single_line.getLength()/2 + self.double_line_width:
                        if [vertical_line.getCenterX(), single_line.getCenterY()] not in self.graph_point:
                            self.graph_point.append([vertical_line.getCenterX(), single_line.getCenterY()])
                            self.graph.addVertex(LINE_VERTEX, np.array([vertical_line.getCenterX(), single_line.getCenterY()]), 
                                                UNDIRECTED, UNDIRECTED, MIX_LINE)
                            self.single_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            UNDIRECTED, UNDIRECTED, SINGLE_HORIZONTAL))
                            self.single_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            UNDIRECTED, UNDIRECTED, SINGLE_VERTICAL))
                        
            for j in range(len(self.double_horizontal_lines)):
                double_line = self.double_horizontal_lines[j]
                if  vertical_line.getCenterY() - vertical_line.getLength()/2 - self.double_line_width <= double_line.getCenterY() - self.double_line_width/4 <=  vertical_line.getCenterY() + vertical_line.getLength()/2 + self.double_line_width:
                    if double_line.getCenterX() - double_line.getLength()/2 - self.double_line_width <= vertical_line.getCenterX() <= double_line.getCenterX() + double_line.getLength()/2 + self.double_line_width:
                        if [vertical_line.getCenterX(), double_line.getCenterY() - self.double_line_width/4] not in self.graph_point:
                            self.graph_point.append([vertical_line.getCenterX(), double_line.getCenterY() - self.double_line_width/4])
                            self.graph.addVertex(LINE_VERTEX, np.array([vertical_line.getCenterX(), 
                                                                    double_line.getCenterY() - self.double_line_width/4 ]), 
                                                POSITIVE_DIRECTED, UNDIRECTED, MIX_LINE)
                            self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            POSITIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL))
                            self.single_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            POSITIVE_DIRECTED, UNDIRECTED, SINGLE_VERTICAL))
                        
                if  vertical_line.getCenterY() - vertical_line.getLength()/2 - self.double_line_width <= double_line.getCenterY() + self.double_line_width/4 <=  vertical_line.getCenterY() + vertical_line.getLength()/2 + self.double_line_width:
                    if double_line.getCenterX() - double_line.getLength()/2 - self.double_line_width <= vertical_line.getCenterX() <= double_line.getCenterX() + double_line.getLength()/2 + self.double_line_width:
                        if [vertical_line.getCenterX(), double_line.getCenterY() + self.double_line_width/4 ] not in self.graph_point:
                            self.graph_point.append([vertical_line.getCenterX(), double_line.getCenterY() + self.double_line_width/4 ])
                            self.graph.addVertex(LINE_VERTEX, np.array([vertical_line.getCenterX(), 
                                                                    double_line.getCenterY() + self.double_line_width/4 ]), 
                                                NEGATIVE_DIRECTED, UNDIRECTED, MIX_LINE)
                            self.double_horizontal_vertices[j].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            NEGATIVE_DIRECTED, UNDIRECTED, DOUBLE_HORIZONTAL))
                            self.single_vertical_vertices[i].append(Vertex(self.graph.getVertex(-1).getID(), LINE_VERTEX, 
                                                                            self.graph.getVertex(-1).getCenter(), 
                                                                            NEGATIVE_DIRECTED, UNDIRECTED, SINGLE_VERTICAL))
        
    def getGraphEdgesBetweenZoneLines(self):
        self.getGraphEdgesInZone(self.waiting_vertices, WAITING_ZONE)
        self.getGraphEdgesBetweenZoneLine(self.double_horizontal_vertices)
        self.getGraphEdgesBetweenZoneLine(self.single_horizontal_vertices)
        self.getGraphEdgesBetweenZoneLine(self.double_vertical_vertices)
        self.getGraphEdgesBetweenZoneLine(self.single_vertical_vertices)
        
    def getGraphEdgesInZone(self, vertices: List[List[Vertex]], zone_type: int):
        for v_list in vertices:
            for i in range(len(v_list)):
                for j in range(i, len(v_list)):
                    v1 = v_list[i]
                    v2 = v_list[j]
                    if v1.getID() == v2.getID(): continue
                    distance = 0.0
                    if zone_type == STORAGE_ZONE:
                        distance = abs(v1.getCenterY() - v2.getCenterY()) - self.storage_points[0].getLength()
                    elif zone_type == WAITING_ZONE:
                        distance = abs(v1.getCenterY() - v2.getCenterY()) - self.waiting_points[0].getLength()
                    else: 
                        distance = abs(v1.getCenterY() - v2.getCenterY()) - self.charging_points[0].getLength()
                    if v1.getCenterX() == v2.getCenterX() and v1.getCenterY() != v2.getCenterY() and distance < 1.0:
                        self.graph.addEdge(v1.id, v2.id)
                        self.graph.addEdge(v2.id, v1.id)
                    # if v1.getType() == WAITING_VERTEX and v2.getType() == WAITING_VERTEX:
                    #     dist = abs(v1.getCenterX() - v2.getCenterX()) - self.waiting_points[0].getWidth()
                    #     if v1.getCenterY() == v2.getCenterY() and v1.getCenterX() != v2.getCenterX() and dist < 1.0:
                    #         self.graph.addEdge(v1.id, v2.id)
                    #         self.graph.addEdge(v2.id, v1.id)
                        
    def getGraphEdgesBetweenZoneLine(self, vertices: List[List[Vertex]]):
        for v_list in vertices:
            for v in v_list:
                for z_v in self.zone_vertices:
                    if self.isGraphEdgeBetweenZoneLine(v, z_v) == True:
                        self.graph.addEdge(v.id, z_v.id)
                        self.graph.addEdge(z_v.id, v.id)
                
    def getGraphEdgeInDoubleHorizontalLine(self):
        for v_list in self.double_horizontal_vertices:
            first_list: List[Vertex] = [v_list[0]]
            second_list: List[Vertex] = []
            mix_list_x: List[float] = []
            for i in range(1, len(v_list)):
                if self.graph.getVertex(v_list[i].getID()).getLineType() == MIX_LINE and v_list[i].getCenterX() not in mix_list_x:
                    mix_list_x.append(v_list[i].getCenterX())
                if round(v_list[i].getCenterY(), 2) == round(v_list[0].getCenterY(), 2):
                    first_list.append(v_list[i])
                else:
                    second_list.append(v_list[i])
            first_sorted_list = sorted(first_list, key=lambda vertex: vertex.getCenterX())
            second_sorted_list = sorted(second_list, key=lambda vertex: vertex.getCenterX())
            for i in range(len(first_sorted_list) - 1 ):
                self.graph.addEdge(first_sorted_list[i].getID(), first_sorted_list[i+1].getID())
            for i in range(len(second_sorted_list) - 1):
                self.graph.addEdge(second_sorted_list[i+1].getID(), second_sorted_list[i].getID())
            edge_list = []
            counter = 0
            for v1 in first_sorted_list:
                for v2 in second_sorted_list:
                    if round(v1.getCenterX(), 2) == round(v2.getCenterX(), 2):
                        if self.graph.getVertex(v1.id).getLineType() != MIX_LINE and self.graph.getVertex(v2.id).getLineType() != MIX_LINE:
                            if len(edge_list) == 0:
                                flag = True
                                for x in mix_list_x:
                                    if abs(v1.getCenterX() - x) < 1.0:
                                        flag = False
                                if flag == True:
                                    self.graph.addEdge(v1.id, v2.id)
                                    edge_list.append(v1.getCenterX())
                                    counter+= 1
                            else:
                                if abs(v1.getCenterX() - edge_list[-1]) > 1.0:
                                    flag = True
                                    for x in mix_list_x:
                                        if abs(v1.getCenterX() - x) < 1.0:
                                            flag = False
                                    if flag == True:
                                        if counter% 2 == 0:
                                            self.graph.addEdge(v1.id, v2.id)
                                            edge_list.append(v1.getCenterX())
                                            counter+= 1
                                        else:
                                            self.graph.addEdge(v2.id, v1.id)
                                            edge_list.append(v1.getCenterX())
                                            counter+= 1
                                        
    def getGraphEdgeInDoubleVerticalLine(self):
        for v_list in self.double_vertical_vertices:
            first_list: List[Vertex] = [v_list[0]]
            second_list: List[Vertex] = []
            mix_list_y: List[float] = []
            for i in range(1, len(v_list)):
                if self.graph.getVertex(v_list[i].getID()).getLineType() == MIX_LINE and v_list[i].getCenterY() not in mix_list_y:
                    mix_list_y.append(v_list[i].getCenterY())
                if round(v_list[i].getCenterX(), 2) == round(v_list[0].getCenterX(), 2):
                    first_list.append(v_list[i])
                else:
                    second_list.append(v_list[i])
            first_sorted_list = sorted(first_list, key=lambda vertex: vertex.getCenterY(), reverse= True)
            second_sorted_list = sorted(second_list, key=lambda vertex: vertex.getCenterY(), reverse = True)
            for i in range(len(first_sorted_list) - 1 ):
                self.graph.addEdge(first_sorted_list[i].getID(), first_sorted_list[i+1].getID())
            for i in range(len(second_sorted_list) - 1):
                self.graph.addEdge(second_sorted_list[i+1].getID(), second_sorted_list[i].getID())
            edge_list = []
            counter = 0
            for v1 in first_sorted_list:
                for v2 in second_sorted_list:
                    if round(v1.getCenterY(), 2) == round(v2.getCenterY(), 2):
                        if self.graph.getVertex(v1.id).getLineType() != MIX_LINE and self.graph.getVertex(v2.id).getLineType() != MIX_LINE:
                            if len(edge_list) == 0:
                                flag = True
                                for y in mix_list_y:
                                    if abs(v1.getCenterX() - y) < 1.0:
                                        flag = False
                                if flag == True:
                                    self.graph.addEdge(v1.id, v2.id)
                                    edge_list.append(v1.getCenterY())
                                    counter+= 1
                            else:
                                if abs(v1.getCenterY() - edge_list[-1]) > 1.0:
                                    flag = True
                                    for y in mix_list_y:
                                        if abs(v1.getCenterX() - y) < 1.0:
                                            flag = False
                                    if flag == True:
                                        if counter% 2 == 0:
                                            self.graph.addEdge(v1.id, v2.id)
                                            edge_list.append(v1.getCenterY())
                                            counter+= 1
                                        else:
                                            self.graph.addEdge(v2.id, v1.id)
                                            edge_list.append(v1.getCenterY())
                                            counter+= 1
    
    def getGraphEdgeInSingleHorizontalLine(self):
        for v_list in self.single_horizontal_vertices:
            sorted_list = sorted(v_list, key=lambda vertex: vertex.getCenterX())
            random_value = np.random.rand()
            if random_value > 0.5:
                for i in range(len(sorted_list) - 1 ):
                    if isSamePoint(sorted_list[i].getCenter(), sorted_list[i+1].getCenter()) == False:
                        self.graph.addEdge(sorted_list[i].getID(), sorted_list[i+1].getID())
            else:
                for i in range(len(sorted_list) - 1 ):
                    if isSamePoint(sorted_list[i].getCenter(), sorted_list[i+1].getCenter()) == False:
                        self.graph.addEdge(sorted_list[i+1].getID(), sorted_list[i].getID())
    
    def getGraphEdgeInSingleVerticalLine(self):
        for v_list in self.single_vertical_vertices:
            sorted_list = sorted(v_list, key=lambda vertex: vertex.getCenterY())
            random_value = np.random.rand()
            if random_value > 0.5:
                for i in range(len(sorted_list) - 1 ):
                    if isSamePoint(sorted_list[i].getCenter(), sorted_list[i+1].getCenter()) == False:
                        self.graph.addEdge(sorted_list[i].getID(), sorted_list[i+1].getID())
            else:
                for i in range(len(sorted_list) - 1 ):
                    if isSamePoint(sorted_list[i].getCenter(), sorted_list[i+1].getCenter()) == False:
                        self.graph.addEdge(sorted_list[i+1].getID(), sorted_list[i].getID())    
    def isGraphEdgeBetweenZoneLine(self, vertex1: Vertex, vertex2: Vertex):
        condition1 = round(vertex1.getCenterX(), 2) == round(vertex2.getCenterX(), 2)
        condition2 = round(vertex1.getCenterY(), 2) == round(vertex2.getCenterY(), 2)
        max_point_length = self.working_points[0].getLength()/2
        if len(self.storage_points) != 0:
            max_point_length = max(max_point_length, self.storage_points[0].getLength()/2)
        if len(self.waiting_points) != 0:
            max_point_length = max(max_point_length, self.waiting_points[0].getLength()/2)
        if len(self.charging_points) != 0:
            max_point_length = max(max_point_length, self.charging_points[0].getLength()/2)
        max_dist = self.single_line_width + max_point_length
        # max_dist = self.single_line_width + max(self.working_points[0].getLength()/2, self.storage_points[0].getLength()/2,
        #                                         self.waiting_points[0].getLength()/2, self.charging_points[0].getLength()/2)
        
        if condition1 or condition2:
            dist = math.hypot(vertex1.getCenterY() - vertex2.getCenterY(), vertex1.getCenterX() - vertex2.getCenterX())
            if dist < max_dist:                
                return True
        return False
    def addLine(self, line_type: int, center:  np.ndarray, line_length: float):
        if line_type == DOUBLE_HORIZONTAL:
            self.double_horizontal_lines.append(Line(line_type, center, line_length, self.double_line_width))
            self.lines.append(Line(line_type, center, line_length, self.double_line_width))
        elif line_type == SINGLE_HORIZONTAL:
            self.single_horizontal_lines.append(Line(line_type, center, line_length, self.single_line_width))
            self.lines.append(Line(line_type, center, line_length, self.single_line_width))
        elif line_type == DOUBLE_VERTICAL:
            self.double_vertical_lines.append(Line(line_type, center, line_length, self.double_line_width))
            self.lines.append(Line(line_type, center, line_length, self.double_line_width))
        else:
            self.single_vertical_lines.append(Line(line_type, center, line_length, self.single_line_width))
            self.lines.append(Line(line_type, center, line_length, self.single_line_width))
    
    def addZonePoint(self, point_type: int, line_type: int, center: np.ndarray, length: float, width: float):
        self.zone_points.append(Point(point_type, line_type, center, length, width))
        if point_type == WORKING_VERTEX:
            self.working_points.append(Point(point_type, line_type, center, length, width))
        elif point_type == STORAGE_VERTEX:
            self.storage_points.append(Point(point_type, line_type, center, length, width))
        elif point_type == WAITING_VERTEX:
            self.waiting_points.append(Point(point_type, line_type, center, length, width))
        elif point_type == CHARGING_VERTEX:
            self.charging_points.append(Point(point_type, line_type, center, length, width))
    
    def calculateWaitingZoneBridge(self):
        self.waiting_bridge: List[Point] = []
        for p1 in self.waiting_points:
            for p2 in self.waiting_points:
                if p1.getCenterX() == p2.getCenterX() and p1.getCenterY() == p2.getCenterY():
                    continue
                distance = abs(p1.getCenterY() - p2.getCenterY()) - p1.getLength()
                if p1.getCenterX() == p2.getCenterX() and  distance < 1.0:
                    self.waiting_bridge.append(Point(-1, p1.line_type, 
                                                    np.array([p1.getCenterX(), (p1.getCenterY() + p2.getCenterY())/2]),
                                                    distance, 0.7))
                # dist = abs(p1.getCenterX() - p2.getCenterX()) - p1.getWidth()
                # if abs(p1.getCenterY() - p2.getCenterY()) < MAX_SAME_DIST and dist < 1.0:
                #     self.waiting_bridge.append(Point(-1, p1.line_type,
                #                                     np.array([(p1.getCenterX() + p2.getCenterX())/2, p1.getCenterY()]),
                #                                     0.7, distance))
        
    def visualizeMap(self, map_visual: axs.Axes):
        map_visual.fill(self.map_coords[:, 0], self.map_coords[:, 1], c= 'black')
        map_visual.plot(self.map_coords[:, 0], self.map_coords[:, 1], c= 'black')
        
        for line in self.lines:
            map_visual.fill(line.coords[:, 0], line.coords[:, 1], c='white')
        for point in self.zone_points:
            map_visual.fill(point.coords[:, 0], point.coords[:, 1], c = 'white')
            if point.getPointType() == WORKING_VERTEX:
                map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'green', marker='.')
            elif point.getPointType() == STORAGE_VERTEX:
                map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'blue', marker='.')
            elif point.getPointType() == WAITING_VERTEX:
                map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'red', marker='.')
            else:
                map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'orange', marker='.')
        for point in self.waiting_bridge:
            map_visual.fill(point.coords[:, 0], point.coords[:, 1], c = 'white')
        for vertex in self.graph.vertices:
            for id in vertex.getNeighbors():
                x1 = vertex.getCenterX()
                y1 = vertex.getCenterY()
                x2 = self.graph.getVertex(id).getCenterX()
                y2 = self.graph.getVertex(id).getCenterY()
                map_visual.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1, fc='black', ec='darkgray')