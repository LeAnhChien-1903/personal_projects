#pragma once
#ifndef TEB_CONTROLLER_H
#define TEB_CONTROLLER_H

#include "homotopy_class_planner/graph_search.h"
#include "homotopy_class_planner/homotopy_class_planner.h"
#include "common/ultis.h"
#include <string>

using namespace teb_local_planner;

class TEBVersion2
{
private:
    // Constant of environments
    double sample_time; // Time step in seconds
    // Constant of robot constraints
    double robot_length; // length of robot
    double robot_width; // width of robot
    double robot_radius; // radius of robot
    double max_linear_velocity;
    double max_angular_velocity;
    double min_linear_velocity;
    double min_angular_velocity;
    double max_linear_acceleration;
    double max_angular_acceleration;
    double goal_tolerance; // the goal tolerance (m)
    // Robot
    PoseSE2 robot_pose; // Store the robot pose [x, y, theta]
    PoseSE2 local_goal; // Store the robot local pose [x, y, theta]
    int local_index; // Index of local goal in global path
    Twist current_vel; // The wheel velocities of robot [v, omega]
    // Obstacles
    ObstContainer *obstacles;
    // Planner
    TebConfig config; // Store the configuration
    PlannerInterfacePtr planner; // Store the planner interface
    ViaPointContainer* via_points; // Store the via point container for planner
    std::vector<PoseSE2> best_path; // Store the best trajectory
    // Start command
    bool startCommand = false;
public:
    /**
     * @brief Construct a new teb version 2 object
     * 
     */
    TEBVersion2();
    /**
     * @brief Construct a new teb version 2 object
     * 
     */
    ~TEBVersion2(){};
    /**
     * @brief Run teb controller 
     * 
     * @param current_pose 
     * @param index_of_collision 
     * @param current_vel 
     * @param global_trajectories 
     */
    std::pair<std::vector<PoseSE2>, int> runTEB(PoseSE2 current_pose, PoseSE2 global_goal, int index_of_collision, Twist current_vel, std::vector<PoseSE2> global_trajectories, std::vector<std::vector<Eigen::Vector2d>> obstacles_);
    /**
     * @brief update obstacles to planner
     * 
     * @param obstacles 
     */
    void updateObstacles(std::vector<std::vector<Eigen::Vector2d>> obstacles_);
    /**
     * @brief Check if local goal safety or not
     * 
     * @return true if safety
     * @return false if no safety
     */
    bool checkSafetyOfLocalGoal(std::vector<std::vector<Eigen::Vector2d>> obstacles_);
    /**
     * @brief Find local goal from global path, current pose and obstacles
     * 
     */
    void findLocalGoal(std::vector<std::vector<Eigen::Vector2d>> obstacles_, int index_of_collision, std::vector<PoseSE2> global_trajectories);
    /**
     * @brief Check if the a circle intersect with a segment represented by two points
     * 
     * @param center 
     * @param radius 
     * @param line_start 
     * @param line_end 
     */
    bool checkCircleIntersectSegment(Eigen::Vector2d center, double radius, Eigen::Vector2d line_start, Eigen::Vector2d line_end);
    /**
     * @brief Check if the a circle intersect with a polygon
     * 
     * @param center 
     * @param radius 
     * @param polygon 
     * @return true 
     * @return false 
     */
    bool checkCircleIntersectPolygon(Eigen::Vector2d center, double radius, std::vector<Eigen::Vector2d> polygon);
    /**
     * @brief check the local goal reached
     * 
     * @return true 
     * @return false 
     */
    bool localGoalReached();
    /**
     * @brief Find minimum difference between two angles
     * 
     * @param angle1 
     * @param angle2 
     * @return double 
     */
    double findDifferenceOrientation(double angle1, double angle2);
    /**
     * @brief Convert meter to pixel position
     * 
     * @param x 
     * @param cellsize 
     * @return int 
     */
    int contxy2disc(double x, double cellsize);
    /**
     * @brief Extract bounding box of robot from pose, length and width of robot
     * 
     * @param pose 
     * @return std::vector<Eigen::Vector2d> 
     */
    std::vector<Eigen::Vector2d> extractRobotBoundingBox(PoseSE2 pose);
};

#endif