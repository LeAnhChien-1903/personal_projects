/******************************************************************************
 * Copyright (c) 2023, NKU Mobile & Flying Robotics Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Jian Wen (nkuwenjian@gmail.com)
 *****************************************************************************/

#include "astar_planner_ros/astar_planner.h"

#include <memory>
#include <mutex>  // NOLINT

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <glog/logging.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <string>
namespace astar_planner_ros
{

  namespace
  {
  constexpr uint8_t kOccupied = 100;
  constexpr uint8_t kFree = 0;
  }  // namespace

  class AStarGlobalPlaner 
  {
    public:
      AStarGlobalPlaner();
      virtual ~AStarGlobalPlaner() = default;
    private:
      void SetMap(const nav_msgs::OccupancyGrid::ConstPtr& map);
      void SetStart(const geometry_msgs::PoseWithCovarianceStamped start);
      void SetGoal(const geometry_msgs::PoseStamped goal);
      void MakePlan();
      void PublishGlobalPlan(const std::vector<geometry_msgs::PoseStamped>& plan) const;
      void timerCallback(const ros::TimerEvent & event);
    private:
      // Ros node handle, publishers and subscribers
      ros::NodeHandle node;
      ros::Subscriber map_sub;
      ros::Subscriber start_sub;
      ros::Subscriber goal_sub;
      ros::Timer global_timer;
      // Topic
      std::string map_topic;
      std::string amcl_topic;
      std::string goal_topic;
      std::string global_path_topic;
      std::vector<double> original_position;
      // Variables
      double frequency; // Frequency of publish global
      uint32_t last_size_x = 0;
      uint32_t last_size_y = 0;
      uint32_t size_x = 0;
      uint32_t size_y = 0;
      double origin_x = 0.0;
      double origin_y = 0.0;
      float resolution = 0.0;
      std::vector<uint8_t> map;
      std::mutex map_mutex;

      int start_x = 0;
      int start_y = 0;
      int goal_x = 0;
      int goal_y = 0;
      bool start_received = false;
      bool goal_received = false;
      std::string global_frame;
      ros::Publisher plan_pub;

      std::unique_ptr<GridSearch> planner = nullptr;
  };

}
