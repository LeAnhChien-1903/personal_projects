from utlis.utlis import *
from env.factory_map import FactoryMap
import cv2

def meterToPixel(point: np.ndarray, origin_x: float, origin_y: float, map_height_pixel: int, resolution: float) -> tuple:
    x = int((point[0] - origin_x)/resolution)
    y = map_height_pixel - int((point[1] - origin_y)/resolution)
    
    return (x, y)

figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
resolution = 0.02
data_folder = "data/50x50"
single_line_width = 0.9
double_line_width = 2.0
point_line_length = 1.2
factory_map = FactoryMap(data_folder, single_line_width, double_line_width, point_line_length)
factory_map.visualizeMap(map_visual)
map_width_pixel: int = int(factory_map.map_length/resolution)
map_height_pixel: int = int(factory_map.map_width/resolution)
map = np.zeros((map_height_pixel, map_width_pixel, 3), dtype=np.int8)
white_color: cv2.typing.Scalar = (255, 255, 255)
black_color: cv2.typing.Scalar = (0, 0, 0)
red_color: cv2.typing.Scalar = (0, 0, 255)
green_color: cv2.typing.Scalar = (0, 255, 0)
blue_color: cv2.typing.Scalar = (255, 0, 0)
origin_x: float = 0.0
origin_y: float = (factory_map.double_vertical_lines[0].length - factory_map.map_width) / 2

for line in factory_map.lines:
    pt1 = meterToPixel(line.coords[0], origin_x, origin_y, map_height_pixel, resolution)
    pt2 = meterToPixel(line.coords[2], origin_x, origin_y, map_height_pixel, resolution)
    cv2.rectangle(map, pt1, pt2, color=white_color, thickness=-1)
for point in factory_map.zone_points:
    pt1 = meterToPixel(point.coords[0], origin_x, origin_y, map_height_pixel, resolution)
    pt2 = meterToPixel(point.coords[2], origin_x, origin_y, map_height_pixel, resolution)
    cv2.rectangle(map, pt1, pt2, color=white_color, thickness=-1)
    # if point.getPointType() == WORKING_VERTEX:
    #     map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'green', marker='.')
    # elif point.getPointType() == STORAGE_VERTEX:
    #     map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'blue', marker='.')
    # elif point.getPointType() == WAITING_VERTEX:
    #     map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'red', marker='.')
    # else:
    #     map_visual.plot(point.getCenterX(), point.getCenterY(), c= 'orange', marker='.')
for point in factory_map.waiting_bridge:
    pt1 = meterToPixel(point.coords[0], origin_x, origin_y, map_height_pixel, resolution)
    pt2 = meterToPixel(point.coords[2], origin_x, origin_y, map_height_pixel, resolution)
    cv2.rectangle(map, pt1, pt2, color=white_color, thickness=-1)
cv2.imwrite(os.path.join(data_folder, 'map.png'), map)
# cv2.imshow("Factory map", map)
# cv2.waitKey(0)
# plt.show()