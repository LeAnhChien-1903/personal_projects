from utlis.utlis import *
from env.env import Environment
from typing import List

data_folder = "data/60x20"

figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
env = Environment(data_folder, num_of_robot= 10, robot_max_speed= 1.0, robot_max_payload= 200, figure=figure, map_visual= map_visual)

def extractZonePoints(graph:Graph):
    working_vertices: List[Vertex] = []
    storage_vertices: List[Vertex] = []
    for vertex in graph.getVertices():
        if vertex.getType() == WORKING_VERTEX:
            working_vertices.append(vertex)
        elif vertex.getType() == STORAGE_VERTEX:
            storage_vertices.append(vertex)
                
    return working_vertices, storage_vertices

def createTask(num_type: int, num_priority: int, min_load:float, max_load: float, 
                working_vertices: List[Vertex], storage_vertices:List[Vertex]):
    epsilon = random.random()
    type_ = random.randint(0, num_type - 1)
    priority = random.randint(1, num_priority)
    mass = random.uniform(min_load, max_load)
    if epsilon >= 0.75:
        start = working_vertices[random.randint(0, len(working_vertices) - 1)]
        target = storage_vertices[random.randint(0, len(storage_vertices) - 1)]
        while True:
            if EuclidDistance(start.getCenter(),  target.getCenter()) < 10.0:
                start = working_vertices[random.randint(0, len(working_vertices) - 1)]
                target = storage_vertices[random.randint(0, len(storage_vertices) - 1)]
            else:
                break
        return [start.getID(), target.getID()], [start.getID(), target.getID(), type_, priority, mass]
    elif 0.5 <= epsilon < 0.75:
        id_list: np.ndarray = np.arange(0, len(working_vertices))
        start_id = np.random.choice(id_list)
        target_id = np.random.choice(np.delete(id_list, start_id))
        while True:
            if EuclidDistance(working_vertices[start_id].getCenter(),  working_vertices[target_id].getCenter()) < 10.0:
                target_id = np.random.choice(np.delete(id_list, start_id))
            else:
                break
        return [working_vertices[start_id].getID(), working_vertices[target_id].getID()], [working_vertices[start_id].getID(), working_vertices[target_id].getID(), type_, priority, mass]
    elif 0.25 <= epsilon < 0.5:
        id_list: np.ndarray = np.arange(0, len(storage_vertices))
        start_id = np.random.choice(id_list)
        target_id = np.random.choice(np.delete(id_list, start_id))
        while True:
            if EuclidDistance(storage_vertices[start_id].getCenter(),  storage_vertices[target_id].getCenter()) < 10.0:
                target_id = np.random.choice(np.delete(id_list, start_id))
            else:
                break
        return [storage_vertices[start_id].getID(), storage_vertices[target_id].getID()], [storage_vertices[start_id].getID(), storage_vertices[target_id].getID(), type_, priority, mass]
    else:
        start = storage_vertices[random.randint(0, len(storage_vertices) - 1)]
        target = working_vertices[random.randint(0, len(working_vertices) - 1)]
        while True:
            if EuclidDistance(start.getCenter(),  target.getCenter()) < 10.0:
                start = storage_vertices[random.randint(0, len(storage_vertices) - 1)]
                target = working_vertices[random.randint(0, len(working_vertices) - 1)]
            else:
                break
        return [start.getID(), target.getID()], [start.getID(), target.getID(), type_, priority, mass]
    
num_type = 2
num_priority = 5
min_load = 30
max_load = 200
working_vertices, storage_vertices = extractZonePoints(env.graph)
for num_of_task in range(50, 1001, 50):
    id, task = createTask(num_type, num_priority, min_load, max_load, working_vertices, storage_vertices)
    task_list = [task]
    task_id = [id]
    for _ in range(num_of_task):
        while True:
            id, task = createTask(num_type, num_priority, min_load, max_load, working_vertices, storage_vertices)
            if (id in task_id) == False:
                task_list.append(task)
                task_id.append(id)
                break
    task_list = np.array(task_list)
    np.savetxt(os.path.join(data_folder, "task_data/{}_task.txt".format(num_of_task)), task_list, fmt='%.2f')
