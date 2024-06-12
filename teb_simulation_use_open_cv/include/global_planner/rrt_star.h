#pragma once
#ifndef RRTSTAR_H
#define RRTSTAR_H

#include "ultis.h"

using namespace Eigen;
class RRTSTAR
{
private:
	cv::Mat map;
	Pose2D startPos, endPos;
	int map_height_pixel, map_width_pixel;
	double half_height_meter, half_width_meter;
	double gain_x, gain_y;
	BGR color_map;
	double robot_radius;
public:
	RRTSTAR();
	void initialize(cv::Mat map, Pose2D start, Pose2D goal, BGR color_map, double radius);
	Node* getRandomNode();
	Node* nearest(Pose2D point);
	void near(Pose2D point, double radius, std::vector<Node*>& out_nodes);
	double distance(Point2D& p, Point2D& q);
	double cost(Node* q);
	double pathCost(Node* qFrom, Node* qTo);
	void add(Node* qNearest, Node* qNew);
	bool reached();
	void setStepSize(double step);
	void setMaxIterations(int iter);
	void deleteNodes(Node* root);
	Vector3d newConfig(Node* q, Node* qNearest);
	void implementAlgorithm();
	bool checkCollision(Point2D start, Point2D end);
	void getSmoothWaypoints();
	std::vector<Node*> nodes;
	std::vector<Node*> path;
	Node* root, * lastNode;
	Waypoints waypoints;
	Waypoints smoothWaypoints;
	int max_iter;
	double step_size;
};
#endif