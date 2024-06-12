#include <ros/ros.h>
#include <tf/tf.h>
#include "teb_ver2.h"
#include "common/pose_se2.h"
#include "common/ultis.h"

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <obstacle_detection/ObstacleVertices.h>
#include <std_msgs/Bool.h>

using namespace teb_local_planner;
// Global variables
PoseSE2 current_robot_pose(0.0, 0.0, 0.0), global_goal(0.0, 0.0, 0.0);
Twist current_vel;
std::vector<std::vector<Eigen::Vector2d>> obstacles;
std::vector<PoseSE2> global_path;
bool start_command = false;
// Local path
nav_msgs::Path global_path_msg, local_path, local_path_simplify;
// Functions
std::pair<bool, int> checkGlobalPathCollideObstacle(PoseSE2 robot_pose, double robot_length, double robot_width, int &index_of_robot_in_global, std::vector<PoseSE2> global_trajectories, std::vector<std::vector<Eigen::Vector2d>> obstacles_, double length_check = 3.0);
bool checkPolygonIntersection(std::vector<Eigen::Vector2d> polygon1, std::vector<Eigen::Vector2d> polygon2);
std::pair<PoseSE2, int> findClosestPoint(std::vector<PoseSE2> global_trajectories, PoseSE2 current_pose, int current_index, double length_check = 3.0);
std::vector<PoseSE2> getGlobalPath(PoseSE2 current_pose, PoseSE2 goal);
std::vector<Eigen::Vector2d> extractRobotBoundingBox(PoseSE2 pose, double robot_length, double robot_width);
std::vector<PoseSE2> simplyLocalTrajectories(std::vector<PoseSE2> trajectory);
double findDifferenceOrientation(double angle1, double angle2);
// Get data from robot and environment
void getCurrentRobotPose(const geometry_msgs::PoseWithCovarianceStamped msg);
void getStartCommand(const std_msgs::Bool msg);
void getGlobalGoal(const geometry_msgs::PoseStamped msg);
void getObstacles(const obstacle_detection::ObstacleVertices msg);
void setVelocity(nav_msgs::Odometry msg);
bool goalReached(PoseSE2 current_pose, PoseSE2 global_goal_);

int main(int argc, char **argv)
{
    ros::init(argc, argv, ros::this_node::getName());
    ros::NodeHandle node;
    ros::Subscriber amcl_sub = node.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 10, getCurrentRobotPose);
    ros::Subscriber start_sub = node.subscribe<std_msgs::Bool>("start_command", 10, getStartCommand);
    ros::Subscriber goal_sub = node.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10, getGlobalGoal);
    ros::Subscriber obstacle_sub = node.subscribe<obstacle_detection::ObstacleVertices>("/detected_obstacle", 10, getObstacles);
    ros::Subscriber odom_sub = node.subscribe<nav_msgs::Odometry>("/odometry/filtered", 10, setVelocity);
    ros::Publisher local_path_pub = node.advertise<nav_msgs::Path>("/local_path", 10);
    ros::Publisher local_path_simplify_pub = node.advertise<nav_msgs::Path>("/local_path_simplify", 10);
    ros::Publisher global_path_pub = node.advertise<nav_msgs::Path>("/global_path", 10);
    ros::Publisher cmd_vel_pub = node.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ros::Rate loop_rate(10);
    TEBVersion2 controller;
    int current_index_robot = 0;
    bool local_goal_flag = false; 
    while (ros::ok())
    {
        if (start_command == true)
        {
            global_path_msg.header.frame_id = "map";
            global_path_msg.header.stamp = ros::Time::now();
            global_path_pub.publish(global_path_msg);
            geometry_msgs::Twist set_vel;
            if (goalReached(current_robot_pose, global_goal) == true)
            {
                cmd_vel_pub.publish(set_vel);
            }
            else
            {
                std::pair<bool, int> check_global;
                if (global_path.size() > 0)
                {
                    check_global = checkGlobalPathCollideObstacle(current_robot_pose, 0.89, 0.58, current_index_robot, global_path, obstacles, 3.0);
                    if (local_goal_flag == false)
                    {
                        if (check_global.first == false)
                        {
                            // Go to global path tracking of Ms Trang
                            set_vel.linear.x = 0.1;
                            cmd_vel_pub.publish(set_vel);
                        }
                        else
                        {
                            cmd_vel_pub.publish(set_vel);
                            local_goal_flag = true; // Have local goal
                        }
                    }
                    if (local_goal_flag == true)
                    {
                        std::pair<std::vector<PoseSE2>, int> local_trajectory = controller.runTEB(current_robot_pose, global_goal ,check_global.second, current_vel, global_path, obstacles);
                        if (local_trajectory.second == 1)
                        {
                            local_goal_flag = false;
                        }
                        else if (local_trajectory.second == -1)
                        {
                            // TEB find trajectories fail
                        }
                        else
                        {
                            local_path.header.frame_id = "map";
                            local_path_simplify.header.frame_id = "map";
                            local_path.poses.clear();
                            local_path_simplify.poses.clear();
                            local_path.header.stamp = ros::Time::now();
                            local_path_simplify.header.stamp = ros::Time::now();
                            geometry_msgs::PoseStamped pose;
                            pose.header.frame_id = "map";
                            // Publish smooth trajectory
                            for (int i = 0; i <local_trajectory.first.size(); i++)
                            {
                                pose.header.seq = i;
                                pose.pose.position.x = local_trajectory.first[i].x();
                                pose.pose.position.y = local_trajectory.first[i].y();
                                tf::Quaternion q;
                                q.setRPY(0.0, 0.0, local_trajectory.first[i].theta());
                                pose.pose.orientation.x = q.getX();
                                pose.pose.orientation.y = q.getY();
                                pose.pose.orientation.z = q.getZ();
                                pose.pose.orientation.w = q.getW();
                                local_path.poses.push_back(pose);
                            }
                            // Publish simplify trajectory
                            std::vector<PoseSE2> simplicity_trajectory = simplyLocalTrajectories(local_trajectory.first);
                            for (int i = 0; i < simplicity_trajectory.size(); i++)
                            {
                                pose.header.seq = i;
                                pose.pose.position.x = simplicity_trajectory[i].x();
                                pose.pose.position.y = simplicity_trajectory[i].y();
                                tf::Quaternion q;
                                q.setRPY(0.0, 0.0, simplicity_trajectory[i].theta());
                                pose.pose.orientation.x = q.getX();
                                pose.pose.orientation.y = q.getY();
                                pose.pose.orientation.z = q.getZ();
                                pose.pose.orientation.w = q.getW();
                                local_path_simplify.poses.push_back(pose);
                            }
                            local_path_pub.publish(local_path);
                            local_path_simplify_pub.publish(local_path_simplify);
                        }
                    }
                }
            }
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}

std::pair<bool, int> checkGlobalPathCollideObstacle(PoseSE2 robot_pose, double robot_length, double robot_width, int &index_of_robot_in_global, std::vector<PoseSE2> global_trajectories, std::vector<std::vector<Eigen::Vector2d>> obstacles_, double length_check)
{
    std::pair<bool, int> output;
    if (obstacles_.size() == 0 || global_trajectories.size() == 0) 
    {
        output.first = false;
        output.second = 0;
        return output;
    }
    std::pair<PoseSE2, int> closest_point = findClosestPoint(global_trajectories, robot_pose, index_of_robot_in_global, length_check);
    double total_distance = 0.0;
    
    std::vector<Eigen::Vector2d> bounding_box;

    index_of_robot_in_global = closest_point.second;
    int index = closest_point.second + 1;
    double diff_distance, diff_angle;
    while (total_distance < length_check)
    {
        total_distance += (global_trajectories[index].position() - global_trajectories[index - 1].position()).norm();
        diff_distance = (global_trajectories[index].position() - robot_pose.position()).norm();
        diff_angle = abs(findDifferenceOrientation(robot_pose.theta(), global_trajectories[index].theta()));
        if (diff_angle > 0.1 || diff_distance > robot_length) 
        {
            bounding_box = extractRobotBoundingBox(global_trajectories[index], robot_length, robot_width);
            for (int i = 0; i < obstacles_.size(); ++i)
            {
                if (checkPolygonIntersection(bounding_box, obstacles_[i]))
                {
                    output.first = true;
                    output.second = index;
                    return output;
                }
            }
        }
        if (index >= global_path.size() - 1) 
        {
            output.second = index;
            break;
        }
        index++;
    }

    output.first = false;
    return output;
}

bool checkPolygonIntersection(std::vector<Eigen::Vector2d> polygon1, std::vector<Eigen::Vector2d> polygon2)
{
    for (int i = 0; i < polygon1.size(); i++)
    {
        for (int j = 0; j < polygon2.size(); j++)
        {
            if (i == polygon1.size() - 1 && j != polygon2.size() - 1)
            {
                if (check_line_segments_intersection_2d(polygon1[i], polygon1[0], polygon2[j], polygon2[j+1])) return true;
            }
            else if (i != polygon1.size() - 1 && j == polygon2.size() - 1)
            {
                if (check_line_segments_intersection_2d(polygon1[i], polygon1[i+1], polygon2[j], polygon2[0])) return true;
            }
            else if (i == polygon1.size() - 1 && j == polygon2.size() - 1)
            {
                if (check_line_segments_intersection_2d(polygon1[i], polygon1[0], polygon2[j], polygon2[0])) return true;
            }
            else
            {   
                if (check_line_segments_intersection_2d(polygon1[i], polygon1[i+1], polygon2[j], polygon2[j+1])) return true;
            }
        }
    }
    return false;
}

std::pair<PoseSE2, int> findClosestPoint(std::vector<PoseSE2> global_trajectories, PoseSE2 current_pose, int current_index, double length_check)
{
    std::pair<PoseSE2, int> closest_point;
    double min_distance = INFINITY;
    double total_distance = 0.0;
    double distance;
    while (total_distance < length_check)
    {
        total_distance += (global_trajectories[current_index + 1].position() - global_trajectories[current_index].position()).norm();
        distance = (global_trajectories[current_index].position() - current_pose.position()).norm();
        if (distance < min_distance) 
        {
            min_distance = distance;
            closest_point.second = current_index;
        }
        current_index++;
        if (current_index >= global_trajectories.size()) break;
    }   
    closest_point.first = global_trajectories[closest_point.second];
    return closest_point;
}

std::vector<PoseSE2> getGlobalPath(PoseSE2 current_pose, PoseSE2 goal)
{
    std::vector<PoseSE2> global_trajectory;
    double sample_distance = 0.1;
    // Get global path
    if (abs(current_pose.x() - goal.x()) <= 0.03)
    {
        if (current_pose.y() < goal.y())
        {
            for (double y = current_pose.y(); y <= goal.y(); y+= sample_distance)
            {
                global_trajectory.push_back(PoseSE2(current_pose.x(), y, 0.0));
            }
        }
        else
        {
            for (double y = current_pose.y(); y >= goal.y(); y-= sample_distance)
            {
                global_trajectory.push_back(PoseSE2(current_pose.x(), y, 0.0));
            }
        }
    }
    else if (abs(current_pose.y() - goal.y()) <= 0.03)
    {
        if (current_pose.x() < goal.x())
        {
            for (double x = current_pose.x(); x <= goal.y(); x += sample_distance)
            {
                global_trajectory.push_back(PoseSE2(x, current_pose.y(), 0.0));
            }
        }
        else
        {
            for (double x = current_pose.x(); x >= goal.x(); x-= sample_distance)
            {
                global_trajectory.push_back(PoseSE2(x, current_pose.y(), 0.0));
            }
        }
    }
    else
    {
        // y = ax + b
        double a = (goal.y() - current_pose.y()) / (goal.x() - current_pose.x());
        double b = current_pose.y() - a * current_pose.x();
        global_trajectory.push_back(current_pose);
        if (current_pose.x() < goal.x())
        {
            for (double x = current_pose.x() + sample_distance; x < goal.x(); x += sample_distance)
            {
                global_trajectory.push_back(PoseSE2(x, a * x + b, 0.0));
            }
        }
        else
        {
            for (double x = current_pose.x() - sample_distance; x > goal.x(); x -= sample_distance)
            {
                global_trajectory.push_back(PoseSE2(x, a * x + b, 0.0));
            }
        }
        global_trajectory.push_back(goal);
    }
    // Estimate the pose of global path
    for (int i = 1; i < global_trajectory.size() - 1; i++)
    {
        global_trajectory[i].theta() = atan2(global_trajectory[i].y() - global_trajectory[i-1].y(), global_trajectory[i].x() - global_trajectory[i-1].x());
    }

    return global_trajectory;
}

std::vector<Eigen::Vector2d> extractRobotBoundingBox(PoseSE2 pose, double robot_length, double robot_width)
{
    std::vector<Eigen::Vector2d> bounding_box;
	double x = pose.x();
	double y = pose.y();
	double halfLength = robot_length / 2;
	double halfWidth = robot_width / 2;
	double sinAngle = sin(pose.theta());
	double cosAngle = cos(pose.theta());

	// Bottom left
	bounding_box.push_back(Eigen::Vector2d(x + (cosAngle * -halfLength) - (sinAngle * halfWidth), y + (sinAngle * -halfLength) + (cosAngle * halfWidth)));

	// Top left corner
	bounding_box.push_back(Eigen::Vector2d(x + (cosAngle * -halfLength) - (sinAngle * -halfWidth), y + (sinAngle * -halfLength) + (cosAngle * -halfWidth)));

	// Top right 
	bounding_box.push_back(Eigen::Vector2d(x + (cosAngle * halfLength) - (sinAngle * -halfWidth), y + (sinAngle * halfLength) + (cosAngle * -halfWidth)));

	// Bottom right
	bounding_box.push_back(Eigen::Vector2d(x + (cosAngle * halfLength) - (sinAngle * halfWidth), y + (sinAngle * halfLength) + (cosAngle * halfWidth)));

	return bounding_box;
}

std::vector<PoseSE2> simplyLocalTrajectories(std::vector<PoseSE2> trajectory)
{
    std::vector<PoseSE2> result_trajectory;
    // Find peak and bottom points
    int peak_point_index, bottom_point_index;
    Eigen::Vector2d vec1, vec2;
    double cross_product;
    double max_peak_distance = 0.0, max_bottom_distance = 0.0;
    bool flag_peak = false, flage_bottom = false;
    double distance = 0.0;
    for (int i = 1; i < trajectory.size() - 1; i++)
    {
        vec1 = trajectory.front().position() - trajectory[i].position();
        vec2 = trajectory.back().position() - trajectory[i].position();
        cross_product = vec1.x() * vec2.y() - vec1.y() * vec2.x();
        distance = calc_distance_point_to_segment(trajectory[i].position(), trajectory.front().position(), trajectory.back().position());
        if (cross_product > 0)
        {
            flag_peak = true;
            if (distance > max_peak_distance)
            {
                max_peak_distance = distance;
                peak_point_index = i;
                
            }
        }
        else
        {
            flage_bottom = true;
            if (distance > max_bottom_distance)
            {
                max_bottom_distance = distance;
                bottom_point_index = i;
            }
        }
    }
    // Simply the trajectory
    result_trajectory.push_back(trajectory.front());
    if (flag_peak == true && flage_bottom == false)
    {
        int middle_index1 = std::ceil(peak_point_index/2);
        int middle_index2 = std::ceil((peak_point_index + trajectory.size() - 1)/2);
        double distance1 = calc_distance_point_to_segment(trajectory[middle_index1].position(), trajectory.front().position(), trajectory[peak_point_index].position());
        double distance2 = calc_distance_point_to_segment(trajectory[middle_index2].position(), trajectory[peak_point_index].position(), trajectory.back().position());

        if (distance1 > 0.1)
        {
            result_trajectory.push_back(trajectory[middle_index1]);
        }
        result_trajectory.push_back(trajectory[peak_point_index]);
        if (distance2 > 0.1)
        {
            result_trajectory.push_back(trajectory[middle_index2]);
        }
        result_trajectory.push_back(trajectory.back());
    }
    else if (flag_peak == false && flage_bottom == true)
    {
        int middle_index1 = std::ceil(bottom_point_index/2);
        int middle_index2 = std::ceil((bottom_point_index + trajectory.size() - 1)/2);
        double distance1 = calc_distance_point_to_segment(trajectory[middle_index1].position(), trajectory.front().position(), trajectory[bottom_point_index].position());
        double distance2 = calc_distance_point_to_segment(trajectory[middle_index2].position(), trajectory[bottom_point_index].position(), trajectory.back().position());

        if (distance1 > 0.1)
        {
            result_trajectory.push_back(trajectory[middle_index1]);
        }
        result_trajectory.push_back(trajectory[bottom_point_index]);
        if (distance2 > 0.1)
        {
            result_trajectory.push_back(trajectory[middle_index2]);
        }
        result_trajectory.push_back(trajectory.back());
    }
    else
    {
        if (peak_point_index < bottom_point_index)
        {
            int middle_index1 = std::ceil(peak_point_index/2);
            int middle_index2 = std::ceil((peak_point_index + bottom_point_index)/2);
            int middle_index3 = std::ceil((bottom_point_index + trajectory.size() - 1)/2);
            double distance1 = calc_distance_point_to_segment(trajectory[middle_index1].position(), trajectory.front().position(), trajectory[peak_point_index].position());
            double distance2 = calc_distance_point_to_segment(trajectory[middle_index2].position(), trajectory[peak_point_index].position(), trajectory[bottom_point_index].position());
            double distance3 = calc_distance_point_to_segment(trajectory[middle_index3].position(), trajectory[bottom_point_index].position(), trajectory.back().position());
            if (distance1 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index1]);
            }
            result_trajectory.push_back(trajectory[peak_point_index]);
            if (distance2 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index2]);
            }
            result_trajectory.push_back(trajectory[bottom_point_index]);
            if (distance3 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index3]);
            }
            result_trajectory.push_back(trajectory.back());
        }
        else
        {
            int middle_index1 = std::ceil(bottom_point_index/2);
            int middle_index2 = std::ceil((peak_point_index + bottom_point_index)/2);
            int middle_index3 = std::ceil((peak_point_index + trajectory.size() - 1)/2);
            double distance1 = calc_distance_point_to_segment(trajectory[middle_index1].position(), trajectory.front().position(), trajectory[bottom_point_index].position());
            double distance2 = calc_distance_point_to_segment(trajectory[middle_index2].position(), trajectory[bottom_point_index].position(), trajectory[peak_point_index].position());
            double distance3 = calc_distance_point_to_segment(trajectory[middle_index3].position(), trajectory[peak_point_index].position(), trajectory.back().position());
            if (distance1 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index1]);
            }
            result_trajectory.push_back(trajectory[bottom_point_index]);
            if (distance2 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index2]);
            }
            result_trajectory.push_back(trajectory[peak_point_index]);
            if (distance3 > 0.1)
            {
                result_trajectory.push_back(trajectory[middle_index3]);
            }
            result_trajectory.push_back(trajectory.back());
        }
    }
    return result_trajectory;
}

double findDifferenceOrientation(double angle1, double angle2)
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

void getCurrentRobotPose(const geometry_msgs::PoseWithCovarianceStamped msg)
{
    current_robot_pose.x() = msg.pose.pose.position.x;
    current_robot_pose.y() = msg.pose.pose.position.y;
    // Convert quaternion to RPY 
    tf::Quaternion q(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w);
    tf::Matrix3x3 m(q);

    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    current_robot_pose.theta() = yaw;
}

void getStartCommand(const std_msgs::Bool msg)
{
    start_command = msg.data;
}

void getGlobalGoal(const geometry_msgs::PoseStamped msg)
{
    global_goal.x() = msg.pose.position.x;
    global_goal.y() = msg.pose.position.y;
    // Convert quaternion to RPY 
    tf::Quaternion q(
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w);
    tf::Matrix3x3 m(q);

    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    global_goal.theta() = yaw;
    global_path = getGlobalPath(current_robot_pose, global_goal);
    // Publish goal path
    global_path_msg.poses.clear();
    global_path_msg.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "map";
    for (int i = 0; i < global_path.size(); i++)
    {
        pose.header.seq = i;
        pose.pose.position.x = global_path[i].x();
        pose.pose.position.y = global_path[i].y();
        tf::Quaternion q;
        q.setRPY(0.0, 0.0, global_path[i].theta());
        pose.pose.orientation.x = q.getX();
        pose.pose.orientation.y = q.getY();
        pose.pose.orientation.z = q.getZ();
        pose.pose.orientation.w = q.getW();
        global_path_msg.poses.push_back(pose);
    }
}

void getObstacles(const obstacle_detection::ObstacleVertices msg)
{
    obstacles.clear();
    std::vector<Eigen::Vector2d> obstacle;
    for (int i = 0; i < msg.vertices_list.size(); i++)
    {
        obstacle.clear();
        for (int k = 0; k < msg.vertices_list[i].poses.size(); k++)
        {
            Eigen::Vector2d vertex(msg.vertices_list[i].poses[k].position.x, msg.vertices_list[i].poses[k].position.y);
            obstacle.push_back(vertex);
        }
        obstacles.push_back(obstacle);
    }
}

void setVelocity(nav_msgs::Odometry msg)
{
    current_vel.linear = msg.twist.twist.linear.x;
    current_vel.angular = msg.twist.twist.angular.z;
}
bool goalReached(PoseSE2 current_pose, PoseSE2 global_goal_)
{
    Eigen::Vector2d diff = global_goal_.position() - current_pose.position();
    if (diff.norm() < 0.05)
    {
        return true;
    }
    return false;
}