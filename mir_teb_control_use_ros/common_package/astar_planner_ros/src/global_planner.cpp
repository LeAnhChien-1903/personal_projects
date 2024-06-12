#include "astar_planner_ros/global_planner.h"

namespace astar_planner_ros
{
	AStarGlobalPlaner::AStarGlobalPlaner()
	{
    this->node.getParam("/astar_global_planner/map_topic",this->map_topic);
    this->node.getParam("/astar_global_planner/amcl_topic",this->amcl_topic);
    this->node.getParam("/astar_global_planner/goal_topic", this->goal_topic);
    this->node.getParam("/astar_global_planner/global_path_topic", this->global_path_topic);
    this->node.getParam("/astar_global_planner/origin", this->original_position);
    this->node.getParam("/astar_global_planner/resolution", this->resolution);
    this->node.getParam("/astar_global_planner/global_frequency", this->frequency);

    this->origin_x = this->original_position[0];
    this->origin_y = this->original_position[1];
		this->map_sub = this->node.subscribe<nav_msgs::OccupancyGrid>(this->map_topic, 10, &AStarGlobalPlaner::SetMap, this);
		this->start_sub = this->node.subscribe<geometry_msgs::PoseWithCovarianceStamped>(this->amcl_topic, 10, &AStarGlobalPlaner::SetStart, this);
		this->goal_sub = this->node.subscribe(this->goal_topic, 10, &AStarGlobalPlaner::SetGoal, this);
		this->plan_pub = this->node.advertise<nav_msgs::Path>(this->global_path_topic, 10);
    this->global_timer = this->node.createTimer(ros::Duration(double(1/this->frequency)), &AStarGlobalPlaner::timerCallback, this);
  }

	void AStarGlobalPlaner::SetMap(const nav_msgs::OccupancyGrid::ConstPtr &map_)
	{
		std::lock_guard<std::mutex> lock(this->map_mutex);
		this->size_x = map_->info.width;
		this->size_y = map_->info.height;
		// this->resolution = map_->info.resolution;
		// this->origin_x = map_->info.origin.position.x;
		// this->origin_y = map_->info.origin.position.y;

		this->map.clear();
		this->map.resize(map_->data.size());
		// http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
		// The map data, in row-major order, starting with (0,0). Occupancy
		// probabilities are in the range[0, 100]. Unknown is - 1.
		// Here, we treat the unknown state as the occupied state.
		for (size_t i = 0; i < map_->data.size(); ++i)
    {
        if (map_->data[i] == kFree) this->map[i] = kFree;
        else this->map[i] = kOccupied;
		}
	}

	void AStarGlobalPlaner::SetStart(const geometry_msgs::PoseWithCovarianceStamped start)
	{
    if (start.pose.pose.position.x > -20000)
    {
      int start_x_ = CONTXY2DISC(start.pose.pose.position.x - this->origin_x, this->resolution);
      int start_y_ = CONTXY2DISC(start.pose.pose.position.y - this->origin_y, this->resolution);

      if (this->start_x == start_x_ && this->start_y == start_y_)return;

      // Update start position
      this->start_x = start_x_;
      this->start_y = start_y_;
      this->start_received = true;

      // global_frame_ = "map";
      if (this->global_frame.empty()) this->global_frame = start.header.frame_id;
      else CHECK_EQ(this->global_frame, start.header.frame_id);

      // LOG(INFO) << "A new start (" << this->start_x << ", " << this->start_y << ") is received.";
    }

	}

	void AStarGlobalPlaner::SetGoal(const geometry_msgs::PoseStamped goal)
	{
		// retrieving goal position
		int goal_x_ = CONTXY2DISC(goal.pose.position.x - this->origin_x, this->resolution);
		int goal_y_ = CONTXY2DISC(goal.pose.position.y - this->origin_y, this->resolution);

    // Check new goal position
		if (this->goal_x == goal_x_ && this->goal_y == goal_y_) return;

    // Update new goal position
    this->goal_x = goal_x_;
		this->goal_y = goal_y_;
		this->goal_received = true;

		// global_frame_ = "map";
		if (this->global_frame.empty()) this->global_frame = goal.header.frame_id;
		else CHECK_EQ(this->global_frame, goal.header.frame_id);
		// LOG(INFO) << "A new goal (" << this->goal_x << ", " << this->goal_y << ") is received.";
	}
    void AStarGlobalPlaner::MakePlan()
    {
        // if a start as well as goal are defined go ahead and plan
      if (!this->start_received || !this->goal_received) {
          return;
      }

      std::lock_guard<std::mutex> lock(this->map_mutex);
      if (this->planner == nullptr || this->last_size_x != this->size_x || this->last_size_y != this->size_y)
      {
          this->planner.reset();
          this->planner = std::make_unique<GridSearch>(this->size_x, this->size_y, this->resolution);
          this->last_size_x = this->size_x;
          this->last_size_y = this->size_y;
      }

      GridAStarResult result;
      if (!this->planner->GenerateGridPath(this->start_x, this->start_y, this->goal_x, this->goal_y,
                                          this->map.data(), kOccupied, SearchType::A_STAR, &result))
      {
          LOG(INFO) << "A-star search failed.";
          return;
      }
      // LOG(INFO) << "A-star search successfully.";

      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = this->global_frame;
      pose.header.stamp = ros::Time::now();
      pose.pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

      CHECK_EQ(result.x.size(), result.y.size());
      std::vector<geometry_msgs::PoseStamped> plan;
      for (size_t i = 0; i < result.x.size(); i++) {
          pose.pose.position.x = DISCXY2CONT(result.x[i], this->resolution) + this->origin_x;
          pose.pose.position.y = DISCXY2CONT(result.y[i], this->resolution) + this->origin_y;
          plan.push_back(pose);
      }
      PublishGlobalPlan(plan);
    }
    void AStarGlobalPlaner::PublishGlobalPlan(const std::vector<geometry_msgs::PoseStamped>& plan) const
    {
        CHECK(!plan.empty());
        nav_msgs::Path gui_path;
        gui_path.header.frame_id = plan.front().header.frame_id;
        gui_path.header.stamp = ros::Time::now();

        gui_path.poses = plan;
        this->plan_pub.publish(gui_path);
    }

    void AStarGlobalPlaner::timerCallback(const ros::TimerEvent &event)
    {
      this->MakePlan();
    }
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "astar_global_planer");
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  astar_planner_ros::AStarGlobalPlaner astar_planner_test;
  ros::spin();

  return 0;
}
