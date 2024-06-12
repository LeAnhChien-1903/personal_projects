#ifndef ROBOT_H
#define ROBOT_H

#include "graph_search.h"
#include "homotopy_class_planner.h"
#include "rrt_star.h"
#include "cubic_spline.h"
#include <Eigen/Dense>
#include "obstacles_ver2.h"
using namespace teb_local_planner;

// Structure for store DynamicWindow of robot
struct DynamicWindow
{
    DynamicWindow(double min_left, double max_left, double min_right, double max_right)
    {
        left_min_vel = min_left;
        left_max_vel = max_left;
        right_min_vel = min_right;
        right_max_vel = max_right;
    }
    DynamicWindow()
    {
        left_min_vel = right_min_vel = left_max_vel = right_min_vel = 0.0;
    }
	double left_min_vel , left_max_vel;
	double right_min_vel, right_max_vel;
};
// Structure for store wheel velocities
struct WheelVelocity
{
    double left_vel , right_vel;
    WheelVelocity(double left_vel_, double right_vel_)
    {
        left_vel = left_vel_;
        right_vel = right_vel_;
    }
    WheelVelocity()
    {
        left_vel = 0.0;
        right_vel = 0.0;
    }
};
// The reachable velocity typedef
typedef std::vector<WheelVelocity> ReachableVelocity;
/**
 * @brief Robot class for visualize teb_local_planner
 * @class Robot: Path tracking and create the trajectory
 */
class Robot
{
private:
    // Robot and obstacles
    PoseSE2 robot_pose; // Store the robot pose [x, y, theta]
    PoseSE2 robot_goal; // Store the robot goal [x, y, theta]
    TebConfig config; // Store the configuration
    ObstacleList obstacle_list; // The obstacle list
    double observable_range; // The observable range of robot
    WheelVelocity cmd_vel; // Store the command velocity
    ObstContainer *obstacles;
    double wheel_base; // The distance between two wheels [m]
    // Robot left and right wheel velocity constraints
    double max_left_velocity; // The maximum left wheel velocity (m/s)
    double max_right_velocity; // The maximum right wheel velocity (m/s)
    double min_left_velocity; // The minimum left wheel velocity (m/s)
    double min_right_velocity; // The minimum right wheel velocity (m/s)
    double max_acceleration; // The maximum acceleration (m/s^2)
    // Planner
    PlannerInterfacePtr planner; // Store the planner interface
    ViaPointContainer via_points; // Store the via point container for planner
    std::vector<PoseSE2> best_path; // Store the best trajectory 
    std::vector<double> best_time_diff; // Store the best time difference between consecutive pose 
    // Path tracking use LQR
    Eigen::Matrix<double, 3, 3> A; // Store the A matrix for tracking control
    Eigen::Matrix<double, 3, 3> Q; // Store the Q matrix for penalize pose error
    Eigen::Matrix<double, 2, 2> R; // Store the R matrix for penalize velocity effort
    Eigen::Matrix<double, 3, 2> B; // Store the B matrix for compute velocity
    double dt; // Store the sample time
    // Visualize
    cv::Mat map_original; // Map original
    double gain_x; // gain to add in x-coordinate
    double gain_y; // gain to add in y-coordinate
    double map_height; // Height of the map
    double map_width; // Width of the map
public:
    /**
     * @brief Default Construct a new Robot object
     * 
     */
    Robot(){}; 
    /**
     * @brief Destroy the Robot object
     * 
     */
    ~Robot(){};
    /**
     * @brief Default construct a new Robot object
     * @param start Start robot pose
     * @param goal goal robot pose
     * @param map map original
     * @param observable_range_ the observable range of robot
     * @param dt_ the sample time
     */
    Robot(PoseSE2 start, WheelVelocity start_vel, PoseSE2 goal, cv::Mat map, double observable_range_ = 10,double dt_ = 0.1);
    /**
     * @brief Get the Obstacles object
     * 
     * @return ObstContainer 
     */
    void getObstacles();
    /**
     * @brief Control robot with path tracking and path planner
     */
    void robotControl();
    /**
     * @brief Extract wheel velocity from best trajectory and reachable velocity
     * @param desired_pose desired pose of robot
     * @return WheelVelocity 
     */
    WheelVelocity extractWheelVelocity();
    /**
     * @brief Calculate the cost with best trajectory
     * 
     * @param current_pose current pose of robot
     * @param current_vel wheel velocity of robot
     * @return double cost with best trajectory
     */
    double calculateCostWithReference(const PoseSE2 current_pose, const WheelVelocity current_vel);
    /**
     * @brief Calculate the dynamic window from current velocity
     * 
     * @return DynamicWindow 
     */
    DynamicWindow calculateDynamicWindow();
    /**
     * @brief Calculate the reachable velocity from current velocity and dynamic window
     * 
     * @return ReachableVelocity 
     */
    ReachableVelocity calculateReachableVelocity();
    /**
     * @brief Run robot with specify velocity 
     * @param wheel_vel the left and right velocity
     */
    void run(const WheelVelocity wheel_vel);
    /**
     * @brief Check the goal reached 
     * @return true if goal reached
     */
    bool goalReached();
    /**
     * @brief Find minimum of difference between two angles
     * 
     * @param angle1 (radians)
     * @param angle2 (radians)
     * @return double (radians)
     */
    double findDifferenceOrientation(double angle1, double angle2);
    /**
     * @brief Extract the vertex of robot at point
     * @param pose the pose of the robot
     */
    std::vector<cv::Point> extractRobotVisual(PoseSE2 pose);
    /**
     * @brief Calculate the next pose from current pose and wheel velocity
     * 
     * @param current_pose current pose of robot
     * @param current_vel current velocity of robot
     * @return PoseSE2 
     */
    PoseSE2 calculateOdometry(PoseSE2 current_pose, WheelVelocity current_vel);
    /**
     * @brief Convert left and right velocity to linear and angular velocity
     * 
     * @param wheel_vel input wheel velocities
     * @return Twist 
     */
    Twist convertWheelToVelocity(WheelVelocity wheel_vel);
    /**
     * @brief Visualize path tracking, obstacles, robot, create teb path
     * @param map map for visualization
     */
    void visualize(cv::Mat &map);
};
#endif