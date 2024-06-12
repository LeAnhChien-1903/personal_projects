#include "robot.h"

Robot::Robot(PoseSE2 start, WheelVelocity start_vel, PoseSE2 goal, cv::Mat map, double observable_range_, double dt_)
{
    // Initialize the robot configuration
    this->robot_pose = start;
    this->cmd_vel = start_vel;
    this->robot_goal = goal;
    this->map_original = map;
    this->observable_range = observable_range_;
    this->dt = dt_;
    this->map_height = map.rows;
    this->map_width = map.cols;
    this->gain_x = double(this->map_width * pixel_to_meter / 2);
    this->gain_y = double(this->map_height * pixel_to_meter / 2);
    this->wheel_base = 0.45;
    this->max_left_velocity = 1.0;
    this->max_right_velocity = 1.0;
    this->min_left_velocity = -0.5;
    this->min_right_velocity = -0.5;
    this->max_acceleration = 1.0;
    // Initialize obstacles list
    this->obstacle_list.initialize(dt_);
    this->obstacles = new ObstContainer;
    // Initialize homotopy planner
    this->planner = PlannerInterfacePtr(new HomotopyClassPlanner(this->config));
    // Set value for path tracking matrix
    this->A.setZero();
    this->A(0, 0) = this->A(1, 1) = this->A(2, 2) = 1.0;
    this->Q.setZero();
    this->Q(0, 0) = this->Q(1, 1) = 1.0;
    this->Q(2, 2) = 10.0;
    this->R.setZero();
    this->R(0, 0) = this->R(1, 1) = 0.0001;
}

void Robot::getObstacles()
{
    this->obstacles->clear();
    for (int i = 0; i < this->obstacle_list.circle_obs.size(); i++)
    {
        Eigen::Vector2d diff = this->obstacle_list.circle_obs[i].position - this->robot_pose.position();
        if (diff.norm() < this->observable_range)
        {
            Eigen::Vector2d center = this->obstacle_list.circle_obs[i].position;
            double radius = this->obstacle_list.circle_obs[i].radius;
            obstacles->push_back(ObstaclePtr(new CircularObstacle(center, radius)));
            // if (this->obstacle_list.circle_obs[i].velocity != Eigen::Vector2d(0, 0))
            // {
            //     obstacles->back()->setCentroidVelocity(this->obstacle_list.circle_obs[i].velocity);
            // }
        }
    }
    for (int i = 0 ; i < this->obstacle_list.rectangle_obs.size(); i++)
    {
        Eigen::Vector2d diff = this->obstacle_list.rectangle_obs[i].position - this->robot_pose.position();
        if (diff.norm() < this->observable_range)
        {
            Point2dContainer vertices;
            for (int j = 0; j < this->obstacle_list.rectangle_obs[i].vertices.size(); j++)
            {
                vertices.push_back(this->obstacle_list.rectangle_obs[i].vertices[j]);
            }
            this->obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
            // if (this->obstacle_list.rectangle_obs[i].velocity != Eigen::Vector2d(0, 0))
            // {
            //     obstacles->back()->setCentroidVelocity(this->obstacle_list.rectangle_obs[i].velocity);
            // }
        }
    }
    planner->updateObstacles(this->obstacles);
}

void Robot::robotControl()
{
    this->getObstacles();
    Twist vel = this->convertWheelToVelocity(this->cmd_vel);
    this->planner->plan(this->robot_pose, this->robot_goal, &vel);
    this->best_path.clear();
    this->best_time_diff.clear();
    this->planner->getBestTrajectory(this->best_path);
    this->planner->getBestTimeDifference(this->best_time_diff);
    WheelVelocity wheel_vel = this->extractWheelVelocity();
    this->run(wheel_vel);
}
DynamicWindow Robot::calculateDynamicWindow()
{
    DynamicWindow DW;
    DW.left_min_vel = std::max(this->min_left_velocity, this->cmd_vel.left_vel - this->max_acceleration * this->dt);
    DW.left_max_vel = std::min(this->max_left_velocity, this->cmd_vel.left_vel + this->max_acceleration * this->dt);
    DW.right_min_vel = std::max(this->min_right_velocity, this->cmd_vel.right_vel - this->max_acceleration * this->dt);
    DW.right_max_vel = std::min(this->max_right_velocity, this->cmd_vel.right_vel + this->max_acceleration * this->dt);
    return DW;
}

ReachableVelocity Robot::calculateReachableVelocity()
{
    DynamicWindow DW = this->calculateDynamicWindow();
    ReachableVelocity RV;
    double left_resolution = (DW.left_max_vel - DW.left_min_vel) / 10;
    double right_resolution = (DW.right_max_vel - DW.right_min_vel) / 10;
    for (double left_vel = DW.left_min_vel; left_vel <= DW.left_max_vel + 0.005; left_vel += left_resolution)
    {
        for (double right_vel = DW.right_min_vel; right_vel <= DW.right_max_vel + 0.005; right_vel += right_resolution)
        {

            RV.push_back(WheelVelocity(left_vel, right_vel));
        }
    }
    return RV;
}
WheelVelocity Robot::extractWheelVelocity()
{
    WheelVelocity outputVel;
    ReachableVelocity RV = this->calculateReachableVelocity();
    double min_cost = INFINITY;
    for (int i = 0; i < RV.size(); i++)
    {
        double cost = this->calculateCostWithReference(this->robot_pose, RV[i]);
        if (cost < min_cost)
        {
            min_cost = cost;
            outputVel = RV[i];
        }
    }
    return outputVel;
}
double Robot::calculateCostWithReference(const PoseSE2 current_pose, const WheelVelocity current_vel)
{
    double cost = 0.0;
    double x = current_pose.x();
    double y = current_pose.y();
    double theta = current_pose.theta();
    double linear = (current_vel.right_vel + current_vel.left_vel) / 2;
    double angular = (current_vel.right_vel - current_vel.left_vel) / this->wheel_base;
    for (int i = 1 ; i <= 5; i++)
    {
        if (i >= this->best_path.size()) break;
        theta = normalize_angle(theta + angular * this->dt);
        x = x + linear * cos(theta) * this->dt;
        y = y + linear * sin(theta) * this->dt;
        double dx = this->best_path[i].x() - x;
        double dy = this->best_path[i].y() - y;
        double d_theta = this->findDifferenceOrientation(theta, this->best_path[i].theta());
        cost += hypot(dx, dy) +  abs(d_theta);
    }
    return cost;
}
void Robot::run(const WheelVelocity wheel_vel)
{
    Twist vel = this->convertWheelToVelocity(wheel_vel);
    this->robot_pose.theta() = normalize_angle(this->robot_pose.theta() + vel.angular * this->dt);
    this->robot_pose.x() += vel.linear * cos(this->robot_pose.theta()) * this->dt;
    this->robot_pose.y() += vel.linear * sin(this->robot_pose.theta()) * this->dt;
    this->cmd_vel = wheel_vel;
}

bool Robot::goalReached()
{
    Eigen::Vector2d diff = this->robot_goal.position() - this->robot_pose.position();
    if (diff.norm() < 0.05)
    {
        return true;
    }
    return false;
}

std::vector<cv::Point> Robot::extractRobotVisual(PoseSE2 pose)
{
    std::vector<cv::Point> visual;
    std::vector<Point2D> corners = extractCoorner(Pose2D(pose.x(), pose.y(), pose.theta()), this->config.robot.robot_length, this->config.robot.robot_width);
    Point2DPixel point(corners[0], this->gain_x, this->gain_y, this->map_height);
    visual.push_back(cv::Point(point.x, point.y));
    Point2DPixel point1(corners[1], this->gain_x, this->gain_y, this->map_height);
    visual.push_back(cv::Point(point1.x, point1.y));
    Point2DPixel point2(0.5 *(corners[2].x + corners[3].x), 0.5 *(corners[2].y + corners[3].y), this->gain_x, this->gain_y, this->map_height);
    visual.push_back(cv::Point(point2.x, point2.y));
    return visual;
}
PoseSE2 Robot::calculateOdometry(PoseSE2 current_pose, WheelVelocity current_vel)
{
    double linear = (current_vel.right_vel + current_vel.left_vel) / 2;
    double angular = (current_vel.right_vel - current_vel.left_vel) / this->wheel_base;
    PoseSE2 new_pose;
    new_pose.theta() = normalize_angle(current_pose.theta() + angular * this->dt);
    new_pose.x() = current_pose.x() + linear * cos(new_pose.theta()) * this->dt;
    new_pose.y() = current_pose.y() + linear * sin(new_pose.theta()) * this->dt;

    return new_pose;
}
Twist Robot::convertWheelToVelocity(WheelVelocity wheel_vel)
{
    double linear = (wheel_vel.left_vel + wheel_vel.right_vel) / 2;
    double angular = (wheel_vel.right_vel - wheel_vel.left_vel) / this->wheel_base;
    return Twist(linear, angular);
}
void Robot::visualize(cv::Mat &map)
{
    // Visualize the goal
    std::vector<cv::Point> goal_visual = this->extractRobotVisual(this->robot_goal);
    cv::fillConvexPoly(map, goal_visual, cv::Scalar(0, 255, 0));
    // Visualize the obstacles
    this->obstacle_list.visualizeObstaclesList(map, this->gain_x, this->gain_y, this->map_height, cv::Scalar(255, 0, 0));
    // Visualize robot
    std::vector<cv::Point> robot_visual = this->extractRobotVisual(this->robot_pose);
    cv::fillConvexPoly(map, robot_visual, cv::Scalar(0, 0, 255));

    // visualize the planner
    this->planner->visualize(map, this->gain_x, this->gain_y, this->map_height);
    // Visualize the best planner
    std::vector<cv::Point> path;
    for (int i = 0; i < this->best_path.size(); ++i)
    {
        Point2DPixel point(this->best_path[i].x(), this->best_path[i].y(), this->gain_x, this->gain_y, this->map_height);
        path.push_back(cv::Point(point.x, point.y));
    }
    cv::polylines(map, path, false, cv::Scalar(0, 255, 0), int(0.05 * meter_to_pixel));
    this->obstacle_list.updateObstacleList(this->map_original, this->gain_x, this->gain_y, this->map_height);
}

double Robot::findDifferenceOrientation(double angle1, double angle2)
{
    angle1 = g2o::normalize_theta(angle1);
    angle2 = g2o::normalize_theta(angle2);
    
    if (angle1 <= M_PI && angle1 >= 0 && angle2 <= M_PI && angle2 >= 0) return (angle2 - angle1);
    else if ( angle1 > -M_PI && angle1 < 0 && angle2 > -M_PI && angle2 < 0) return (angle2 - angle1);
    else if ( angle1 <= M_PI && angle1 >= 0 && angle2 > -M_PI && angle2 < 0)
    {
        double turn = angle2 - angle1;
        if (turn < -M_PI)
            turn = turn + 2 * M_PI;
        return turn;
    }
    else if ( angle1 > -M_PI && angle1 < 0 && angle2 <= M_PI && angle2 >= 0)
    {
        double turn = angle2 - angle1;
        if (turn > M_PI)
            turn = turn - 2 * M_PI;
        return turn;
    }

    return (angle2 - angle1);
}
