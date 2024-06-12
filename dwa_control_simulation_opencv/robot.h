#ifndef ROBOT_H
#define ROBOT_H

#include "obstacles.h"
#include "ultis.h"
#include "cubic_spline.h"
#include "rrtstar.h"
class RobotConfiguration
{
public:
	double meter_to_sec, observable_range, max_left_vel, max_right_vel, min_left_vel, min_right_vel;
	double max_accleration, wheel_radius, wheel_base, resolution, predict_time, cost_map;
	double to_goal_cost_gain, speed_cost_gain, obstacle_cost_gain, reference_cost_gain;
	double robot_width, robot_length, robot_radius;
	BGR color_map;
	RobotConfiguration();
	~RobotConfiguration();
};
class Robot: private RobotConfiguration
{
private:
	Pose2D robot_pose, goal_pose, init_pose;
	double dt;
	InputControl vel;
	cv::Mat map;
	ObstacleList obstacles;
	double gain_x, gain_y;
	int map_height, map_width;
	Path robot_path, ref_path_visual;
	RRTSTAR rrt_star;
	ReferencePath ref_path;
	CubicSpline2D cubic; 
public:
	bool goal_reached;
	Robot(Pose2D, Pose2D, InputControl, cv::Mat, double);
	~Robot();
	cv::Mat robotControl();
	InputControl DWA_Original(const ReachableVelocity, std::vector<Obstacle>);
	bool checkRobotCollideObstacle(const Pose2D, const std::vector<Point2D>, const std::vector<double>);
	double calculateMinDistanceBetweenObsatcles(const Pose2D, const std::vector<Point2D> , const std::vector<double>);
	double calculateObsatcleCost(const PredictPath, const std::vector<Obstacle>);
	double calculateGoalCost(const PredictPath);
	double calculateReferencePathCost(const PredictPath, std::pair<Point2D, int>);
	void run(const InputControl);
	DynamicWindow calculateDynamicWindow();
	ReachableVelocity calculateReachableVelocity();
	std::vector<Obstacle> obstacleDetection();
	Pose2D odometry(const Pose2D, const InputControl);
	PredictPath predictTrajectory(const Pose2D, const InputControl, const double);
	cv::Mat visualization(std::vector<Obstacle>);
	std::vector<cv::Point> robotVisualzation(const Pose2D);
	void setGoal(const Pose2D);
	void caculateReferencePath();
	std::pair<Point2D, int> chooseDesiredPoint();
	std::vector<std::vector<Point2D>> predictStateObstacles(const std::vector<Obstacle>, const double);
};

#endif