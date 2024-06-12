#include <stdlib.h>
#include "graph_search.h"
#include "optimal_planner.h"
#include "homotopy_class_planner.h"
#include "robot.h"

using namespace teb_local_planner;
using namespace cv;
using namespace std;
RNG rng(12345);
TebConfig config;
int contxy2disc(double x, double cellsize)
{
    if (x >= 0)
    {
        return static_cast<int>(x / cellsize);
    } 
    else 
    {
        return static_cast<int>(x / cellsize) - 1;
    }
}
void visualizeObstacles(cv::Mat &map, ObstContainer *obstacles, double gain_x, double gain_y, int map_height);

int main()
{
	Mat map_original = cv::imread("map/teb_map.pgm", cv::IMREAD_COLOR);
	
	double origin_x = -46.200000;
	double origin_y = -51.000000;
	double x = 0;
	double y = 0;
	int x_pixel = contxy2disc(x-origin_x, 0.05);
	int y_pixel = 1696- contxy2disc(y-origin_y, 0.05);
	std::cout << "x_pixel: " << x_pixel << ", y_pixel: " << y_pixel << std::endl;
	// cv::flip(map_original, map_original, 0);
	cv::circle(map_original, cv::Point(x_pixel, y_pixel), 20, cv::Scalar(0), -1);
	cv::imshow("Map", map_original);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

int main()
{
	cv::Mat map_original = cv::imread("map/map_empty.png", cv::IMREAD_COLOR);
	PoseSE2 start(-6.0, 0.0, 0.0);
	PoseSE2 goal(6.0, 0.0, 0.0);
	Twist start_vel(0, 0);
	Robot *robot = new Robot(start, start_vel, goal, map_original, 0.1);
	cv::Mat map;
	int i = 0;
	while (robot->goalReached() == false)
	{
		map_original.copyTo(map);
		robot->robotControl();
		robot->visualize(map);
		cv::imshow("Visualize", map);
		std::ostringstream graph_path;
		graph_path << "image/visualize" << i+1 << ".png";
		cv::imwrite(graph_path.str(), map);
		if (cv::waitKey(100) && 0xFF == 'q') break;
		i++;
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	delete robot;
	return 0;
}
// int main()
// {
// 	PoseSE2 start(-6.0, 0.0, 0.0);
// 	PoseSE2 goal(6.0, 0.0, 0.0);
// 	ObstContainer *obstacles = new ObstContainer;
// 	// // obstacles->push_back(ObstaclePtr(new CircularObstacle(Eigen::Vector2d(-1.5, 1.5), 0.6)));
// 	// // obstacles->push_back(ObstaclePtr(new CircularObstacle(Eigen::Vector2d(1.5, -1.5), 0.6)));
// 	// // obstacles->push_back(ObstaclePtr(new CircularObstacle(Eigen::Vector2d(-1.5, -1.5), 0.6)));
// 	// // obstacles->push_back(ObstaclePtr(new CircularObstacle(Eigen::Vector2d(1.5, 1.5), 0.6)));
// 	// // obstacles->push_back(ObstaclePtr(new CircularObstacle(Eigen::Vector2d(0.0, 0.0), 0.6)));
// 	// std::vector<Point2D> corners = extractCoorner(Pose2D(0.0, 0.0, 0.0), 1.2, 1.2);
// 	// Point2dContainer vertices;
// 	// for (int i = 0; i < corners.size(); i++)
// 	// {
// 	// 	vertices.push_back(Eigen::Vector2d(corners[i].x, corners[i].y));
// 	// }
// 	// obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
// 	ViaPointContainer via_points;
// 	PlannerInterfacePtr planner = PlannerInterfacePtr(new HomotopyClassPlanner(config, obstacles, &via_points));
// 	cv::Mat map_original = cv::imread("map/map_empty.png", cv::IMREAD_COLOR);
// 	map_original = computeCostMap(map_original, 0.5, BGR(0, 140, 255), false);
// 	double y1 = 2.0;
// 	double y2 = -2.0;
// 	double gain = 0.2;
// 	for (int i = 0; i < 40; i++)
// 	{
// 		obstacles->clear();
// 		std::vector<Point2D> corners1 = extractCoorner(Pose2D(3.0, y1, 0.0), 1.2, 1.2);
// 		std::vector<Point2D> corners2 = extractCoorner(Pose2D(-3.0, y2, 0.0), 1.2, 1.2);
// 		Point2dContainer vertices1, vertices2;
// 		for (int i = 0; i < corners1.size(); i++)
// 		{
// 			vertices1.push_back(Eigen::Vector2d(corners1[i].x, corners1[i].y));
// 			vertices2.push_back(Eigen::Vector2d(corners2[i].x, corners2[i].y));
// 		}
// 		obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices1)));
// 		obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices2)));
// 		planner->updateObstacles(obstacles);
// 		Twist vel(0.5, 0);
// 		planner->plan(start, goal, &vel);
// 		cv::Mat map;
// 		map_original.copyTo(map);
// 		visualizeObstacles(map, obstacles, 7.5, 7.5, 600);
// 		planner->visualize(map, 7.5, 7.5, 600);
// 		std::vector<PoseSE2> best_path;
// 		planner->getBestTrajectory(best_path);
// 		std::vector<cv::Point> path;
// 		for (int i = 0; i < best_path.size(); ++i)
// 		{
// 			Point2DPixel point(best_path[i].x(), best_path[i].y(), 7.5, 7.5, 600);
// 			path.push_back(cv::Point(point.x, point.y));
// 		}
// 		cv::polylines(map, path, false, cv::Scalar(0, 255, 0), int(0.05 * meter_to_pixel));
// 		cv::imshow("Map", map);
// 		// cv::waitKey(0);
// 		std::ostringstream graph_path;
// 		graph_path << "image/visualize" << i+1 << ".png";
// 		cv::imwrite(graph_path.str(), map);
// 		if (cv::waitKey(100) && 0xFF == 'q') break;
// 		if (i%20 == 0)
// 		{
// 			gain = -gain;
// 		}
// 		y1 += gain;
// 		y2 -= gain;
// 	}
// 	cv::destroyAllWindows();
// 	std::string output_file = "video/visualize.avi";

// 	// Define the codec and create VideoWriter object.
// 	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
// 	cv::VideoWriter video(output_file, fourcc, 12, map_original.size());
// 	for (int i = 0; i < 40; i++)
// 	{
// 		std::ostringstream graph_path;
// 		graph_path << "image/visualize" << i+1 << ".png";
// 		 // Read an image and write it to the video file.
// 		cv::Mat image = cv::imread(graph_path.str());
// 		video.write(image);
// 	}
// 	// Close the video file.
// 	video.release();
// 	delete obstacles, planner;
// 	return 0;
// }
void visualizeObstacles(cv::Mat &map, ObstContainer *obstacles, double gain_x, double gain_y, int map_height)
{
	for (int i = 0; i < obstacles->size(); i++)
	{
		obstacles->at(i)->visualize(map, cv::Scalar(255, 0, 0), gain_x, gain_y, map_height);
	}
}
