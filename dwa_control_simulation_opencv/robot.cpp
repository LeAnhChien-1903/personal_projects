#include "robot.h"

RobotConfiguration::RobotConfiguration()
{
	this->meter_to_sec = 1.2; // for cubic interpolation
	this->observable_range = 10.0; // [m] obstacle observation
    this->max_left_vel = 1.0; // [m / s]
    this->max_right_vel = 1.0;  // [m / s]
    this->min_left_vel = -1.0; // [m / s]
    this->min_right_vel = -1.0; // [m / s]
    this->max_accleration= 1.0; // [m / s ^ 2]
    this->wheel_radius = 0.0625; // [m] wheel radius
    this->wheel_base = 0.45; // [m] wheel base
    this->resolution = 11;//
    this->predict_time = 1.0; // [s]
    this->to_goal_cost_gain = 0.15;
    this->speed_cost_gain = 1.0;
    this->obstacle_cost_gain = 2.0;
    this->reference_cost_gain = 1.0;
    this->robot_width = 0.60; // [m] for collision check
    this->robot_length = 0.90; // [m] for collision check
    this->robot_radius = sqrt(pow((this->robot_length / 2), 2) + pow((this->robot_width / 2), 2));// [m] for collision check
    this->cost_map = this->robot_radius; // [m] 
    this->color_map.b = 0;
    this->color_map.g = 140;
    this->color_map.r = 255;
}

RobotConfiguration::~RobotConfiguration(){}

Robot::Robot(Pose2D init_pose, Pose2D goal, InputControl init_control, cv::Mat map_init, double dt)
{
    this->robot_pose = init_pose;
    this->init_pose = init_pose;
    this->goal_pose = goal;
    this->vel = init_control;
    this->dt = dt;
    this->map = computeCostMap(map_init, this->cost_map, this->color_map);
    this->map_height = this->map.cols;
    this->map_width = this->map.rows;
    this->gain_x = double(this->map_width * pixel_to_meter / 2);
    this->gain_y = double(this->map_height * pixel_to_meter / 2);
    this->obstacles.initialization(this->map, this->predict_time, this->dt, this->color_map);
    this->goal_reached = false;
    this->rrt_star.initilaize(this->map, this->robot_pose, this->goal_pose, this->color_map, this->robot_radius);
    this->caculateReferencePath();
}

Robot::~Robot(){}

cv::Mat Robot::robotControl()
{
    cv::Mat visual;
    std::vector<Obstacle> obsatcles = this->obstacleDetection();
    ReachableVelocity RV = this->calculateReachableVelocity();
    InputControl best_vel = this->DWA_Original(RV, obsatcles);
    visual = this->visualization(obsatcles);
    this->run(best_vel);
    double dx = this->goal_pose.position.x - this->robot_pose.position.x;
    double dy = this->goal_pose.position.y - this->robot_pose.position.y;
    if (hypot(dx, dy) < 0.1) this->goal_reached = true;
    return visual;
}

InputControl Robot::DWA_Original(const ReachableVelocity RV, std::vector<Obstacle> obstacles)
{
    double min_cost = INFINITY;
    InputControl best_speed;
    best_speed.left_vel = 0.0;
    best_speed.right_vel = 0.0;
    std::pair<Point2D, int> desiredPoint = this->chooseDesiredPoint();
    std::cout << "Current point: " << this->robot_pose.position.x << " " << this->robot_pose.position.y << std::endl;
    std::cout << "Desired Point: " << desiredPoint.first.x << " " << desiredPoint.first.y << std::endl;
    for (int i = 0; i < RV.size(); i++)
    {
        PredictPath trajectory = this->predictTrajectory(this->robot_pose, RV[i], this->predict_time);
        // Calculate cost
        double obstacle_cost = this->calculateObsatcleCost(trajectory, obstacles) * this->obstacle_cost_gain;
        double speed_cost = (this->max_left_vel - (RV[i].left_vel + RV[i].right_vel)/2) * this->speed_cost_gain;
        double to_goal_cost = this->calculateGoalCost(trajectory) * this->to_goal_cost_gain;
        double dx = this->goal_pose.position.x - this->robot_pose.position.x;
        double dy = this->goal_pose.position.y - this->robot_pose.position.y;
        double reference_path_cost = this->calculateReferencePathCost(trajectory, desiredPoint) * this->reference_cost_gain;
        double total_cost = obstacle_cost + speed_cost + to_goal_cost + reference_path_cost;
        if (total_cost < min_cost)
        {
            min_cost = total_cost;
            best_speed = RV[i];
        }
    }
    std::cout << "Best velocity: " << best_speed.left_vel <<" " << best_speed.right_vel << std::endl;
    return best_speed;
}
cv::Mat Robot::visualization(std::vector<Obstacle> obs)
{
    // Obstacle visualization
    cv::Mat result;
    this->map.copyTo(result);
    for (int i = 0; i < this->obstacles.obstacles.size(); i++)
    {
        Point2DPixel point = convertMeterToPixel(this->obstacles.obstacles[i].position, this->gain_x, this->gain_y, this->map_height);
        cv::Point center(point.x, point.y);
        int radius = int(this->obstacles.obstacles[i].radius * meter_to_pixel);
        cv::Scalar color(255, 0, 0);
        int thickness = -1;
        cv::circle(result, center, radius, color, thickness);
    }
    for (int i = 0; i < obs.size(); i++)
    {
        Point2DPixel point = convertMeterToPixel(obs[i].position, this->gain_x, this->gain_y, this->map_height);
        cv::Point center(point.x, point.y);
        int radius = int(obs[i].radius * meter_to_pixel);
        cv::Scalar color(0, 255, 0);
        int thickness = -1;
        cv::circle(result, center, radius, color, thickness);
    }
    // Init start point and end point
    std::vector<cv::Point> init_points = this->robotVisualzation(this->init_pose);
    std::vector<cv::Point> goal_points = this->robotVisualzation(this->goal_pose);
    cv::fillConvexPoly(result, init_points, cv::Scalar(0, 0, 255));
    cv::fillConvexPoly(result, goal_points, cv::Scalar(0, 255, 0));
    // Visualize reference path
    cv::polylines(result, this->ref_path_visual, false, cv::Scalar(0, 255, 0), 2);
    // Robot visualization
    std::vector<cv::Point> robot_points = this->robotVisualzation(this->robot_pose);
    cv::Scalar color(0, 0, 255);
    cv::fillConvexPoly(result, robot_points, color);
    // Robot path visualization
    Point2DPixel pose = convertMeterToPixel(this->robot_pose.position, this->gain_x, this->gain_y, this->map_height);
    this->robot_path.push_back(cv::Point(pose.x, pose.y));
    cv::polylines(result, this->robot_path, false, cv::Scalar(0, 128, 255), 2);
    return result;
}
void Robot::run(const InputControl vel)
{
    // Update position of robot and obstacle
    this->vel = vel;
    double linear = (vel.right_vel + vel.left_vel) / 2;
    double angular = (vel.right_vel - vel.left_vel) / this->wheel_base;
    this->robot_pose.theta = normalize_angle(this->robot_pose.theta + angular * this->dt);
    this->robot_pose.position.x += linear * cos(this->robot_pose.theta) * this->dt;
    this->robot_pose.position.y += linear * sin(this->robot_pose.theta) * this->dt;
    this->obstacles.updateObstaclePosition();
}
bool Robot::checkRobotCollideObstacle(const Pose2D pose, const std::vector<Point2D> obstacles_position, const std::vector<double> obstacles_radius)
{
    // Check collision with obsatcles
    for (int i = 0; i < obstacles_position.size(); i++)
    {
        double dx = obstacles_position[i].x - pose.position.x;
        double dy = obstacles_position[i].y - pose.position.y;
        if (hypot(dx, dy) <= obstacles_radius[i] + this->robot_radius + 0.1) return true;
    }
    // Check collision with map
    for (double angle = 0; angle <= CV_2PI; angle += CV_PI / 20)
    {
        Point2D point;
        point.x = pose.position.x + this->robot_radius * cos(angle);
        point.y = pose.position.y + this->robot_radius * sin(angle);
        Point2DPixel point_pixel = convertMeterToPixel(point, this->gain_x, this->gain_y, this->map_height);
        BGR color = getColorInPoint(this->map, point_pixel);
        if (color.b == this->color_map.b && color.g == this->color_map.g && color.r == this->color_map.r) return true;
        if (color.b == 0 && color.g == 0 && color.r == 0) return true;
    }
    return false;
}

double Robot::calculateMinDistanceBetweenObsatcles(const Pose2D pose, const std::vector<Point2D> obstacles_position, const std::vector<double> obstacles_radius)
{
    double min_distance = INFINITY;
    for (int i = 0; i < obstacles_position.size(); i++)
    {
        double dx = obstacles_position[i].x - pose.position.x;
        double dy = obstacles_position[i].y - pose.position.y;
        double distance = hypot(dx, dy);
        if (distance < min_distance)
        {
            min_distance = distance;
        }
    }
    return min_distance;
}

double Robot::calculateObsatcleCost(const PredictPath path, const std::vector<Obstacle> obstacles)
{
    double min_distance = INFINITY;
    std::vector<std::vector<Point2D>> obstacles_position = this->predictStateObstacles(obstacles, this->predict_time);
    std::vector<double> obstacle_radius;
    for (int i = 0; i < obstacles.size(); i++)
    {
        obstacle_radius.push_back(obstacles[i].radius);
    }
    for (int i = 0; i < path.size(); i++)
    {
        
        if (this->checkRobotCollideObstacle(path[i], obstacles_position[i], obstacle_radius) == true) return 1000;
        else
        {
            double distance = this->calculateMinDistanceBetweenObsatcles(path[i], obstacles_position[i], obstacle_radius);
            if (distance < min_distance) min_distance = distance;
        }
    }

    return 1 / min_distance;
}
std::vector<Obstacle> Robot::obstacleDetection()
{
    // Find obstacles in observable range of robot
    std::vector<Obstacle> obstacles;
    for (int i = 0; i < this->obstacles.obstacles.size(); i++)
    {
        double dx = this->obstacles.obstacles[i].position.x - this->robot_pose.position.x;
        double dy = this->obstacles.obstacles[i].position.y - this->robot_pose.position.y;
        if (hypot(dx, dy) <= this->observable_range)
        {
            obstacles.push_back(this->obstacles.obstacles[i]);
        }
    }
    return obstacles;
}
Pose2D Robot::odometry(const Pose2D current, const InputControl vel)
{
    double linear = (vel.right_vel + vel.left_vel) / 2;
    double angular = (vel.right_vel - vel.left_vel) / this->wheel_base;
    Pose2D new_pose;
    new_pose.theta = normalize_angle(current.theta + angular * this->dt);
    new_pose.position.x = current.position.x + linear * cos(new_pose.theta) * this->dt;
    new_pose.position.y = current.position.y + linear * sin(new_pose.theta) * this->dt;

    return new_pose;
}

DynamicWindow Robot::calculateDynamicWindow()
{
    DynamicWindow DW;
    DW.left_min_vel = std::max(this->min_left_vel, this->vel.left_vel - this->max_accleration * this->dt);
    DW.left_max_vel = std::min(this->max_left_vel, this->vel.left_vel + this->max_accleration * this->dt);
    DW.right_min_vel = std::max(this->min_right_vel, this->vel.right_vel - this->max_accleration * this->dt);
    DW.right_max_vel = std::min(this->max_right_vel, this->vel.right_vel + this->max_accleration * this->dt);
    return DW;
}
ReachableVelocity Robot::calculateReachableVelocity()
{
    DynamicWindow DW = this->calculateDynamicWindow();
    ReachableVelocity RV;
    double left_resolution = (DW.left_max_vel - DW.left_min_vel) / (this->resolution -  1);
    double right_resolution = (DW.right_max_vel - DW.right_min_vel) / (this->resolution - 1);
    for (double left_vel = DW.left_min_vel; left_vel <= DW.left_max_vel; left_vel += left_resolution)
    {
        for (double right_vel = DW.right_min_vel; right_vel <= DW.right_max_vel; right_vel += right_resolution)
        {
            InputControl vel;
            vel.left_vel = left_vel;
            vel.right_vel = right_vel;
            RV.push_back(vel);
        }
    }

    return RV;
}
double Robot::calculateGoalCost(const PredictPath path)
{
    double dx = this->goal_pose.position.x - path.back().position.x;
    double dy = this->goal_pose.position.y - path.back().position.y;
    double error_angle = atan2(dy, dx);
    double cost_angle = error_angle - path.back().theta;
    double cost = abs(atan2(sin(cost_angle), cos(cost_angle)));
    return cost;
}

double Robot::calculateReferencePathCost(const PredictPath path, std::pair<Point2D, int> desiredPoint)
{
    double cost = 0.0;
    for (int i = 0; i < path.size(); i++)
    {
        if (desiredPoint.second + i > this->ref_path.size()-1) break;
        double dx = this->ref_path[desiredPoint.second + i].x - path[i].position.x;
        double dy = this->ref_path[desiredPoint.second + i].y - path[i].position.y;
        cost += hypot(dx, dy);
    }
    return cost;
}

PredictPath Robot::predictTrajectory(const Pose2D current, const InputControl vel, const double predictTime)
{
    PredictPath trajectory;
    Pose2D pose = current;
    for (double i = 0; i < predictTime; i+= this->dt)
    {
        pose = this->odometry(pose, vel);
        trajectory.push_back(pose);
    }
    return trajectory;
}
std::vector<cv::Point> Robot::robotVisualzation(const Pose2D pose)
{
    std::vector<cv::Point> points;
    std::vector<Point2D> pointList = extracRectangle(pose, this->robot_length, this->robot_width);
    for (int i = 0; i < pointList.size(); i++)
    {
        Point2DPixel point = convertMeterToPixel(pointList[i], this->gain_x, this->gain_y, this->map_height);
        points.push_back(cv::Point(point.x, point.y));
    }

    return points;
}

void Robot::setGoal(const Pose2D goal)
{
    this->goal_pose = goal;
}

void Robot::caculateReferencePath()
{
    this->rrt_star.implementAlgorithm();
    this->cubic.initialization(this->rrt_star.smoothWaypoints, this->dt, this->meter_to_sec);
    this->ref_path = this->cubic.computeCubicPath();
    this->ref_path_visual.clear();
    for (int i = 0; i < this->ref_path.size(); i++)
    {
        Point2DPixel point = convertMeterToPixel(this->ref_path[i], this->gain_x, this->gain_y, this->map_height);
        this->ref_path_visual.push_back(cv::Point(point.x, point.y));
    }
}

std::pair<Point2D, int> Robot::chooseDesiredPoint()
{
    double min_distance = INFINITY;
    std::pair<Point2D, int> desiredPoint;
    double dx = this->goal_pose.position.x - this->robot_pose.position.x;
    double dy = this->goal_pose.position.y - this->robot_pose.position.y;
    if (hypot(dx, dy) < 2.0)
    {
        desiredPoint.first = this->goal_pose.position;
        desiredPoint.second = (int) this->ref_path.size() - 1;
    }
    for (int i = 0; i < this->ref_path.size(); i++)
    {
        double dx = this->ref_path[i].x - this->robot_pose.position.x;
        double dy = this->ref_path[i].y - this->robot_pose.position.y;
        double distance = hypot(dx, dy);
        if (distance < min_distance and distance > 0.01)
        {
            min_distance = distance;
            desiredPoint.first = this->ref_path[i];
            desiredPoint.second = i;
        }
    }

    return desiredPoint;
}

std::vector<std::vector<Point2D>> Robot::predictStateObstacles(const std::vector<Obstacle> obstacleList, const double predictTime)
{
    std::vector<std::vector<Point2D>> obstacleTrajectory;
    std::vector<Point2D> pos;
    for (int i = 0; i < obstacleList.size(); i++)
    {
        Point2D point = obstacleList[i].position;
        pos.push_back(point);
    }
    for (double i = 0; i < predictTime; i += this->dt)
    {
        for (int j = 0; j < pos.size(); j++)
        {
            pos[j].x = pos[j].x + obstacleList[j].vel.x * this->dt;
            pos[j].y = pos[j].y + obstacleList[j].vel.y * this->dt;
        }
        obstacleTrajectory.push_back(pos);
    }
    return obstacleTrajectory;
}
