/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016,
 *  TU Dortmund - Institute of Control Theory and Systems Engineering.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the institute nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Christoph Rösmann
 *********************************************************************/

#ifndef PLANNER_INTERFACE_H_
#define PLANNER_INTERFACE_H_

// boost
#include <boost/shared_ptr.hpp>

// this package
#include "common/pose_se2.h"
#include "robot/robot_footprint_model.h"


namespace teb_local_planner
{
  //! Typedef for a container storing via-points
  typedef std::vector< Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > ViaPointContainer;
/**
 * @class PlannerInterface
 * @brief This abstract class defines an interface for local planners
 */  
class PlannerInterface
{
public:

  /**
   * @brief Default constructor
   */
  PlannerInterface()
  {
  }  
  /**
   * @brief Virtual destructor.
   */
  virtual ~PlannerInterface()
  {
  }
    
  
  /** @name Plan a trajectory */
  //@{
  
  /**
   * @brief Plan a trajectory based on an initial reference plan.
   * 
   * Provide this method to create and optimize a trajectory that is initialized
   * according to an initial reference plan (given as a container of poses).
   * @param initial_plan vector of geometry_msgs::PoseStamped
   * @param start_vel Current start velocity (e.g. the velocity of the robot, only linear.x and angular.z are used)
   * @param free_goal_vel if \c true, a nonzero final velocity at the goal pose is allowed,
   *        otherwise the final velocity will be zero (default: false)
   * @return \c true if planning was successful, \c false otherwise
   */
  virtual bool plan(const std::vector<Pose2D>& initial_plan, const Twist* start_vel = NULL, bool free_goal_vel=false) = 0;
  /**
   * @brief Plan a trajectory between a given start and goal pose.
   * 
   * Provide this method to create and optimize a trajectory that is initialized between a given start and goal pose.
   * @param start PoseSE2 containing the start pose of the trajectory
   * @param goal PoseSE2 containing the goal pose of the trajectory
   * @param start_vel Initial velocity at the start pose (twist msg containing the translational and angular velocity).
   * @param free_goal_vel if \c true, a nonzero final velocity at the goal pose is allowed,
   *        otherwise the final velocity will be zero (default: false)
   * @return \c true if planning was successful, \c false otherwise
   */
  virtual bool plan(const PoseSE2& start, const PoseSE2& goal, const Twist* start_vel = NULL, bool free_goal_vel=false) = 0;
  
  /**
   * @brief Get the velocity command from a previously optimized plan to control the robot at the current sampling interval.
   * @warning Call plan() first and check if the generated plan is feasible.
   * @param[out] velocity translational and rotational velocity [m/s]
   * @param[in] look_ahead_poses index of the final pose used to compute the velocity command.
   * @return \c true if command is valid, \c false otherwise
   */
  virtual bool getVelocityCommand(Twist &velocity,  int look_ahead_poses) const = 0;
  
  //@}
  
  
  /**
   * @brief Reset the planner.
   */
  virtual void clearPlanner() = 0;
  
  /**
   * @brief Prefer a desired initial turning direction (by penalizing the opposing one)
   * 
   * A desired (initial) turning direction might be specified in case the planned trajectory oscillates between two 
   * solutions (in the same equivalence class!) with similar cost. Check the parameters in order to adjust the weight of the penalty.
   * Initial means that the penalty is applied only to the first few poses of the trajectory.
   * @param dir This parameter might be RotType::left (prefer left), RotType::right (prefer right) or RotType::none (prefer none)
   */
  virtual void setPreferredTurningDir(RotType dir) {std::cout << "setPreferredTurningDir() not implemented for this planner."<< std::endl;}
  virtual void updateRobotModel(RobotFootprintModelPtr robot_model)
  {
  }

  // /**
  //  * @brief Check whether the planned trajectory is feasible or not.
  //  * 
  //  * This method currently checks only that the trajectory, or a part of the trajectory is collision free.
  //  * Obstacles are here represented as costmap instead of the internal ObstacleContainer.
  //  * @param costmap_model Pointer to the costmap model
  //  * @param footprint_spec The specification of the footprint of the robot in world coordinates
  //  * @param inscribed_radius The radius of the inscribed circle of the robot
  //  * @param circumscribed_radius The radius of the circumscribed circle of the robot
  //  * @param look_ahead_idx Number of poses along the trajectory that should be verified, if -1, the complete trajectory will be checked.
  //  * @return \c true, if the robot footprint along the first part of the trajectory intersects with 
  //  *         any obstacle in the costmap, \c false otherwise.
  //  */
  // virtual bool isTrajectoryFeasible(cv::Mat* costmap_model, const std::vector<Point2D>& footprint_spec,
  //       double inscribed_radius = 0.0, double circumscribed_radius=0.0, int look_ahead_idx=-1, double feasibility_check_lookahead_distance=-1.0) = 0;
    
  /**
   * Compute and return the cost of the current optimization graph (supports multiple trajectories)
   * @param[out] cost current cost value for each trajectory
   *                  [for a planner with just a single trajectory: size=1, vector will not be cleared]
   * @param obst_cost_scale Specify extra scaling for obstacle costs
   * @param alternative_time_cost Replace the cost for the time optimal objective by the actual (weighted) transition time
   */
  virtual void computeCurrentCost(std::vector<double>& cost, double obst_cost_scale=1.0, bool alternative_time_cost=false)
  {
  }

  /**
   * @brief Returns true if the planner has diverged.
   */
  virtual bool hasDiverged() const = 0;
  /**
   * @brief Visualize planner specific stuff.
   * @param map original map
   * @param gain_x gain to add in x-coordinate
   * @param gain_y gain to add in y-coordinate
   * @param map_height height of map
   */ 
  virtual void visualize(cv::Mat map, double gain_x, double gain_y, double map_height){}
  /**
   * @brief Update the obstacles for planner
   * @param obstacles the new set of obstacle
   * 
   */
  virtual void updateObstacles(ObstContainer* obstacles){};
  /**
   * @brief Update the vias points for planner
   * 
   * @param points 
   */
  virtual void updateViaPoints(ViaPointContainer* points){};
  /**
   * @brief Get the best trajectory
   * @param trajectory the best trajectory
   */
  virtual void getBestTrajectory(std::vector<PoseSE2>& trajectory){};
  /**
   * @brief Get the best time difference between consecutive pose of the best trajectory
   * @param best_time_diff the best time difference
   */
  virtual void getBestTimeDifference(std::vector<double>& best_time_diff){};
};

//! Abbrev. for shared instances of PlannerInterface or it's subclasses 
typedef boost::shared_ptr<PlannerInterface> PlannerInterfacePtr;


} // namespace teb_local_planner

#endif /* PLANNER_INTERFACE_H__ */
