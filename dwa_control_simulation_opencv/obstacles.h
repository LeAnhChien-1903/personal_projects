#ifndef OBSTACLES_H
#define OBSTACLES_H
#include "ultis.h"

class Obstacle
{
public:
	Point2D position;
	Velocity vel;
	double width, length, radius, predictTime, dt;
	Obstacle(Point2D, Velocity, double, double, double);
	~Obstacle();
	void updatePosition(const cv::Mat, const double, const double, const int, const BGR);
	std::vector<Point2D> predictObstacleState();
	bool checkCollideWithWall(const Point2D, const cv::Mat, const double, const double, const int, const BGR);
};
class ObstacleList
{
private:
	cv::Mat map;
	int height;
	double gain_x, gain_y, predict_time, dt;
	BGR color_map;
public:
	std::vector<Obstacle> obstacles;
	ObstacleList();
	void initialization(cv::Mat, const double, const double, const BGR);
	~ObstacleList();
	std::vector<Obstacle> getObstacleInObservation(const Pose2D, const double);
	void updateObstaclePosition();
	cv::Mat obstacleVisualization();
};

#endif