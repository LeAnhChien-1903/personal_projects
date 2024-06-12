#include "robot/teb_controller.h"

TEBControllerWithMap::TEBControllerWithMap()
{
    this->node.getParam("/teb_control/sample_time", this->sample_time);
    this->node.getParam("/teb_control/wheel_base", this->wheel_base);
    this->node.getParam("/teb_control/wheel_radius", this->wheel_radius);
    this->node.getParam("/teb_control/robot_length", this->robot_length);
    this->node.getParam("/teb_control/robot_width", this->robot_width);

    this->node.getParam("/teb_control/max_left_velocity", this->max_left_velocity);
    this->node.getParam("/teb_control/max_right_velocity", this->max_right_velocity);
    this->node.getParam("/teb_control/min_left_velocity", this->min_left_velocity);
    this->node.getParam("/teb_control/min_right_velocity", this->min_right_velocity);
    this->node.getParam("/teb_control/max_acceleration", this->max_acceleration);
    this->node.getParam("/teb_control/goal_tolerance", this->goal_tolerance);

    this->node.getParam("/teb_control/global_planner_topic", this->global_planner_topic);
    this->node.getParam("/teb_control/start_topic", this->start_topic);
    this->node.getParam("/teb_control/obstacle_topic", this->obstacle_topic);
    this->node.getParam("/teb_control/odometry_topic", this->odometry_topic);
    this->node.getParam("/teb_control/amcl_topic", this->amcl_topic);
    this->node.getParam("/teb_control/vel_topic", this->vel_topic);
    this->node.getParam("/teb_control/goal_topic", this->goal_topic);
    this->node.getParam("/teb_control/global_map_topic", this->global_costmap_topic);
    this->node.getParam("/teb_control/local_path_topic", this->local_path_topic);

    this->node.getParam("/obstacle_detector/height", this->local_costmap_height);
    this->node.getParam("/obstacle_detector/width", this->local_costmap_width);
    
    this->robot_radius = sqrt(pow(this->robot_length/2, 2) + pow(this->robot_width/2, 2));
    this->local_path.header.frame_id = "map";
    // this->best_path_visual.header.frame_id = "map";

    this->config.obstacles.min_obstacle_dist = this->robot_radius;
    this->config.obstacles.inflation_dist = this->robot_radius;
    this->config.obstacles.dynamic_obstacle_inflation_dist = this->robot_radius + 0.1;
    this->config.robot.max_vel_x = this->max_left_velocity;
    this->config.robot.max_vel_x_backwards = -0.5 * this->max_left_velocity;
    this->config.robot.max_vel_theta = 1.0;
    this->config.robot.acc_lim_x = this->max_acceleration;
    this->config.robot.acc_lim_theta = this->max_acceleration;
    this->config.obstacles.inflation_dist = 0.3;
    this->config.obstacles.dynamic_obstacle_inflation_dist = 0.3;

    this->robot_goal = PoseSE2(INFINITY, 0, 0);

    this->obstacles = new ObstContainer;
    this->via_points = new ViaPointContainer;
    this->planner = PlannerInterfacePtr(new HomotopyClassPlanner(this->config, this->obstacles, this->via_points));
    
    // Initialize goal and current pose
    this->robot_goal = PoseSE2(10000.0, 10000.0, 0.0);
    this->init_robot_pose = Eigen::Vector3d(0.0, 0.0, 0.0);
    this->global2local = Eigen::Rotation2D<double>(0.0).inverse();
    this->initSubAndPub();
}

void TEBControllerWithMap::timerCallback(const ros::TimerEvent &msg)
{
    // this->transform.setOrigin(tf::Vector3(this->init_robot_pose.x(), this->init_robot_pose.y(), 0.0));
    // tf::Quaternion q;
    // q.setRPY(0, 0, this->init_robot_pose.z());
    // this->transform.setRotation(q);
    // this->br.sendTransform(tf::StampedTransform(this->transform, ros::Time::now(), "map", "init_robot_pose"));
    if (this->startCommand == true && this->global_path.size() > 1)
    {
        // Get reference path
        std::cout << this->global_path.size() << std::endl;
        this->pruneGlobalPlan();
        this->transformGlobalPlan(3);
        this->via_points->clear();
        for (std::size_t i=1; i < this->transformed_plan.size(); i+=5 ) // skip first one, since we do not need any point before the first min_separation [m]
        {                
            // add via-point
            this->via_points->push_back( Eigen::Vector2d( transformed_plan[i].pose.position.x, transformed_plan[i].pose.position.y ) );
        }
        this->planner->updateViaPoints(this->via_points);
        // Get sub goal
        double yaw = this->estimateLocalGoalOrientation();
        PoseSE2 local_goal(this->transformed_plan.back().pose.position.x, this->transformed_plan.back().pose.position.y, yaw);
        Twist vel(0, 0);
        // Plan TEB
        int numberPlaner = 0;
        while(true)
        {   
            if (local_goal.x() == this->robot_goal.x() && local_goal.y() == this->robot_goal.y())
            {
                this->planner->plan(this->robot_pose, local_goal, &vel, false);
            }
            else
            {
                this->planner->plan(this->robot_pose, local_goal, &vel, true);
            }
            
            this->best_path.clear();
            this->planner->getBestTrajectory(this->best_path);
            if (this->isFeasiblePath() == true or numberPlaner > 2)
            {
                break;
            }
            numberPlaner++;
        }
        // Visual by OpenCV
        cv::Mat open_cv_visual = cv::imread("/home/leanhchien/catkin_ws/src/robot_controller/real_robot/robot_slam/map/real_map.pgm", cv::IMREAD_COLOR);
        // cv::imshow("Image", open_cv_visual);
        // cv::waitKey();
        // std::cout << "Open CV" << std::endl;
        // Publish local path 
        this->local_path.poses.clear();
        this->local_path.header.stamp = ros::Time::now();
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        for (int i = 0; i < this->best_path.size(); i++)
        {
            pose.header.seq = i;
            pose.pose.position.x = this->best_path[i].x();
            pose.pose.position.y = this->best_path[i].y();
            tf::Quaternion q;
            q.setRPY(0.0, 0.0, this->best_path[i].theta());
            pose.pose.orientation.x = q.getX();
            pose.pose.orientation.y = q.getY();
            pose.pose.orientation.z = q.getZ();
            pose.pose.orientation.w = q.getW();
            this->local_path.poses.push_back(pose);
        }

        this->local_path_pub.publish(this->local_path);
        // // Transform the best path to init pose coordinates
        // for (int i = 0; i < this->best_path.size();  i++)
        // {
        //     double diff_x = this->best_path[i].x() - this->init_robot_pose.x();
        //     double diff_y = this->best_path[i].y() - this->init_robot_pose.y();
        //     this->best_path[i].x() = diff_x * this->global2local(0, 0) + diff_y * this->global2local(0, 1);
        //     this->best_path[i].y() = diff_x * this->global2local(1, 0) + diff_y * this->global2local(1, 1);
        // }
        // // Cubic spline interpolation
        // Waypoints best_path_cubic;
        // for (int i = 0; i < this->best_path.size(); i++)
        // {
        //     best_path_cubic.push_back(Point2D(this->best_path[i].x(), this->best_path[i].y()));
        // }
        // CubicSpline2D cubic_spline;
        // cubic_spline.initialization(best_path_cubic, 0.1, 2.0);
        // ReferencePath path = cubic_spline.computeCubicPath();

        // this->local_path.poses.clear();
        // this->local_path.header.stamp = ros::Time::now();
        // pose.header.frame_id = "init_robot_pose";
        // for (int i = 0; i < path.size(); i++)
        // {
        //     pose.header.seq = i;
        //     pose.pose.position.x = path[i].x;
        //     pose.pose.position.y = path[i].y;
        //     this->local_path.poses.push_back(pose);
        // }

        // this->local_path_pub.publish(this->local_path);
        
        // this->visualization();
    }
}

void TEBControllerWithMap::visualization()
{
    visualization_msgs::MarkerArray best_path_visual = this->bestPathVisualization();
    this->visualization_pub.publish(best_path_visual);
}

bool TEBControllerWithMap::pruneGlobalPlan(double dist_behind_robot)
{
    if (this->global_path.empty()) return true;
    
    double dist_thresh_sq = dist_behind_robot*dist_behind_robot;
    
    // iterate plan until a pose close the robot is found
    std::vector<geometry_msgs::PoseStamped>::iterator it = this->global_path.begin();
    std::vector<geometry_msgs::PoseStamped>::iterator erase_end = it;
    while (it != this->global_path.end())
    {
        double dx = this->robot_pose.x() - it->pose.position.x;
        double dy = this->robot_pose.y() - it->pose.position.y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq < dist_thresh_sq)
        {
            erase_end = it;
            break;
        }
        ++it;
    }
    if (erase_end == this->global_path.end())
    return false;

    if (erase_end != this->global_path.begin())
    this->global_path.erase(this->global_path.begin(), erase_end);
    return true;
}

bool TEBControllerWithMap::transformGlobalPlan(double max_plan_length)
{
    const geometry_msgs::PoseStamped& plan_pose = this->global_path[0];

    this->transformed_plan.clear();
    if (this->global_path.empty())
    {
        this->sub_goal_index = 0;
        return false;
    }
    //we'll discard points on the plan that are outside the local costmap
    double dist_threshold = std::max(double(this->local_costmap_width/2) , double(this->local_costmap_height/2));
    dist_threshold *= 0.85; // just consider 85% of the costmap size to better incorporate point obstacle that are
                           // located on the border of the local costmap
    int i = 0;
    double sq_dist_threshold = dist_threshold * dist_threshold;
    double sq_dist = 1e10;
    
    //we need to loop to a point on the plan that is within a certain distance of the robot
    bool robot_reached = false;
    for(int j=0; j < (int)this->global_path.size(); ++j)
    {
        double x_diff = this->robot_pose.x() - this->global_path[j].pose.position.x;
        double y_diff = this->robot_pose.y() - this->global_path[j].pose.position.y;
        double new_sq_dist = x_diff * x_diff + y_diff * y_diff;

        if (robot_reached && new_sq_dist > sq_dist)
            break;

        if (new_sq_dist < sq_dist) // find closest distance
        {
            sq_dist = new_sq_dist;
            i = j;
            if (sq_dist < 0.05)      // 2.5 cm to the robot; take the immediate local minima; if it's not the global
            robot_reached = true;  // minima, probably means that there's a loop in the path, and so we prefer this
        }
    }
    
    double plan_length = 0; // check cumulative Euclidean distance along the plan
    
    //now we'll transform until points are outside of our distance threshold
    while(i < (int)this->global_path.size() && sq_dist <= sq_dist_threshold && (max_plan_length<=0 || plan_length <= max_plan_length))
    {
        transformed_plan.push_back(this->global_path[i]);
        double x_diff = this->robot_pose.x() - this->global_path[i].pose.position.x;
        double y_diff = this->robot_pose.y() - this->global_path[i].pose.position.y;
        sq_dist = x_diff * x_diff + y_diff * y_diff;
        
        // calculate distance to previous pose
        if (i>0 && max_plan_length > 0)
            plan_length += distance_points2d(this->global_path[i-1].pose.position, this->global_path[i].pose.position);
        ++i;
    }
    this->sub_goal_index = i-1; // subtract 1, since i was increased once before
    return true;
}

double TEBControllerWithMap::estimateLocalGoalOrientation(int moving_average_length)
{
    int n = (int)this->global_path.size();
    
    // check if we are near the global goal already
    if (this->sub_goal_index > n - moving_average_length - 2)
    {
        return this->robot_goal.theta();    
    }
    
    // reduce number of poses taken into account if the desired number of poses is not available
    moving_average_length = std::min(moving_average_length, n - this->sub_goal_index - 1 ); // maybe redundant, since we have checked the vicinity of the goal before
    
    std::vector<double> candidates;
    geometry_msgs::PoseStamped tf_pose_k = this->transformed_plan.back();
    geometry_msgs::PoseStamped tf_pose_kp1;
    
    int range_end = this->sub_goal_index + moving_average_length;
    for (int i = this->sub_goal_index; i < range_end; ++i)
    {
        // calculate yaw angle  
        candidates.push_back( std::atan2(this->global_path[i+1].pose.position.y - this->transformed_plan.back().pose.position.y,
            this->global_path[i+1].pose.position.x - this->transformed_plan.back().pose.position.x));
        
        if (i < range_end - 1) 
        tf_pose_k = this->global_path[i+1];
    }
    return average_angles(candidates);
}
visualization_msgs::MarkerArray TEBControllerWithMap::bestPathVisualization()
{
    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker init_pose_marker;
    init_pose_marker.header.stamp = ros::Time::now();
    init_pose_marker.header.frame_id = "map";
    init_pose_marker.ns = "goal";
    init_pose_marker.id = 0;
    init_pose_marker.action = visualization_msgs::Marker::ADD;
    init_pose_marker.type = visualization_msgs::Marker::CUBE;
    
    init_pose_marker.pose.position.x = this->init_robot_pose.x();
    init_pose_marker.pose.position.y = this->init_robot_pose.y();
    init_pose_marker.pose.position.z = 0.05;

    init_pose_marker.pose.orientation.x = 0.0;
    init_pose_marker.pose.orientation.y = 0.0;
    init_pose_marker.pose.orientation.z = 0.0;
    init_pose_marker.pose.orientation.w = 1.0;


    init_pose_marker.scale.x = 0.1; // line width
    init_pose_marker.scale.y = 0.1; // line height
    init_pose_marker.scale.z = 0.1; 
    
    init_pose_marker.color.r = 0.0;
    init_pose_marker.color.g = 1.0;
    init_pose_marker.color.b = 0;
    init_pose_marker.color.a = 1.0;
    markers.markers.push_back(init_pose_marker);

    return markers;
}

visualization_msgs::MarkerArray TEBControllerWithMap::visitedPoseVisualization()
{
    visualization_msgs::MarkerArray markers;
    this->robot_visited.push_back(this->robot_pose);

    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = "map";
    marker.ns = "visited_pose";
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1; // line width
    marker.scale.y = 0.1; // line height
    marker.scale.z = 0.1; 
    
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0;
    marker.color.a = 1.0;
    
    geometry_msgs::Point point;
    for (uint16_t index = 0; index <  this->robot_visited.size(); index++)
    {
        point.x = this->robot_visited[index].x();
        point.y = this->robot_visited[index].y();
        marker.points.push_back(point);
    }
    
    markers.markers.push_back(marker);

    
    return markers;
}

void TEBControllerWithMap::initSubAndPub()
{
    this->odometry_sub = this->node.subscribe<nav_msgs::Odometry>(this->odometry_topic, 10, &TEBControllerWithMap::odometryCallback, this);
    this->amcl_sub = this->node.subscribe<geometry_msgs::PoseWithCovarianceStamped>(this->amcl_topic, 10, &TEBControllerWithMap::acmlCallback, this);
    this->start_sub = this->node.subscribe<std_msgs::Bool>(this->start_topic, 10, &TEBControllerWithMap::startCallback, this);
    this->goal_sub = this->node.subscribe<geometry_msgs::PoseStamped>(this->goal_topic, 10, &TEBControllerWithMap::goalCallback, this);
    this->obstacle_sub = this->node.subscribe<obstacle_detection::ObstacleVertices>(this->obstacle_topic, 10, &TEBControllerWithMap::obstaclesCallback, this);
    this->global_path_sub = this->node.subscribe<nav_msgs::Path>(this->global_planner_topic, 10, &TEBControllerWithMap::globalPathCallback, this);
    this->global_costmap_sub = this->node.subscribe<nav_msgs::OccupancyGrid>(this->global_costmap_topic, 10, &TEBControllerWithMap::globalCostMapCallback, this);
    this->local_path_pub = this->node.advertise<nav_msgs::Path>(this->local_path_topic, 10);
    // this->best_path_visual_pub = this->node.advertise<nav_msgs::Path>("/best_path_visual", 10);
    this->visualization_pub = this->node.advertise<visualization_msgs::MarkerArray>("/visualization", 10);
    // this->robot_local_pose_pub = this->node.advertise<geometry_msgs::Pose2D>("/robot_local_pose", 10);
    this->timer_ = this->node.createTimer(ros::Duration(0.05), &TEBControllerWithMap::timerCallback, this);
}

void TEBControllerWithMap::odometryCallback(const nav_msgs::Odometry msg)
{
    if (msg.pose.pose.position.x != 0)
    {   
        double linear = msg.twist.twist.linear.x;
        double angular = msg.twist.twist.angular.z;
        this->cmd_vel = this->convertVelocityToWheelVelocity(linear, angular);
    }
}

void TEBControllerWithMap::acmlCallback(const geometry_msgs::PoseWithCovarianceStamped msg)
{
    if (msg.pose.pose.position.x != 0)
    {
        this->robot_pose.x() = msg.pose.pose.position.x;
        this->robot_pose.y() = msg.pose.pose.position.y;
        // Convert quaternion to RPY 
        tf::Quaternion q(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w);
        tf::Matrix3x3 m(q);

        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        this->robot_pose.theta() = yaw;
        // double diff_x = msg.pose.pose.position.x - this->init_robot_pose.x();
        // double diff_y = msg.pose.pose.position.y - this->init_robot_pose.y();
        // geometry_msgs::Pose2D pose;
        // pose.x = diff_x * this->global2local(0, 0) + diff_y * this->global2local(0, 1);
        // pose.y = diff_x * this->global2local(1, 0) + diff_y * this->global2local(1, 1);
        // pose.theta = normalize_angle(yaw - this->init_robot_pose.z());
        // this->robot_local_pose_pub.publish(pose);
    }
}

void TEBControllerWithMap::startCallback(const std_msgs::Bool msg)
{
    this->startCommand = msg.data;
}

void TEBControllerWithMap::goalCallback(const geometry_msgs::PoseStamped msg)
{
    if (msg.pose.position.x != 0)
    {
        if (msg.pose.position.x != this->robot_goal.x() && msg.pose.position.y != this->robot_goal.y())
        {
            this->init_robot_pose.x() = this->robot_pose.x();
            this->init_robot_pose.y() = this->robot_pose.y();
            this->init_robot_pose.z() = this->robot_pose.theta();
            this->global2local = Eigen::Rotation2D<double>(this->robot_pose.theta()).inverse();
        }
        this->robot_goal.x() = msg.pose.position.x;
        this->robot_goal.y() = msg.pose.position.y;
        // Convert quaternion to RPY 
        tf::Quaternion q(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w);
        tf::Matrix3x3 m(q);

        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        float angle = normalize_angle(yaw);
        this->robot_goal.theta() = yaw;
    }
}

void TEBControllerWithMap::globalPathCallback(const nav_msgs::Path msg)
{
    if (msg.poses.size() > 1)
    {
        double distance = hypot(msg.poses.back().pose.position.x - this->robot_goal.x(), msg.poses.back().pose.position.y - this->robot_goal.y());
        std::cout << "distance: " << distance << std::endl;
        if (distance > 0.2)
        {
            this->global_path.clear();
            this->global_path = msg.poses;
        }
    }
}
void TEBControllerWithMap::obstaclesCallback(const obstacle_detection::ObstacleVertices msg)
{
    this->obstacles->clear();
    for (int i = 0; i < msg.vertices_list.size(); i++)
    {
        if (msg.vertices_list[i].poses.size() == 1)
        {
            Eigen::Vector2d point(msg.vertices_list[i].poses[0].position.x, msg.vertices_list[i].poses[0].position.y);
            this->obstacles->push_back(ObstaclePtr(new CircularObstacle(point, 0.1)));
        }
        else if (msg.vertices_list[i].poses.size() == 2)
        {
            Eigen::Vector2d start_point(msg.vertices_list[i].poses[0].position.x, msg.vertices_list[i].poses[0].position.y);
            Eigen::Vector2d end_point(msg.vertices_list[i].poses[1].position.x, msg.vertices_list[i].poses[1].position.y);
            this->obstacles->push_back(ObstaclePtr(new LineObstacle(start_point, end_point)));
        }
        else
        {
            Point2dContainer vertices;
            for (int k = 0; k < msg.vertices_list[i].poses.size(); k++)
            {
                Eigen::Vector2d vertex(msg.vertices_list[i].poses[k].position.x, msg.vertices_list[i].poses[k].position.y);
                vertices.push_back(vertex);
            }
            this->obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
        }
    }
    this->planner->updateObstacles(this->obstacles);
}

void TEBControllerWithMap::globalCostMapCallback(const nav_msgs::OccupancyGrid msg)
{
    this->global_costmap.clear();
    this->global_costmap_width = msg.info.width;
    this->resolution = msg.info.resolution;
    this->original_position.push_back(msg.info.origin.position.x);
    this->original_position.push_back(msg.info.origin.position.y);
    this->global_costmap.resize(msg.data.size());
    // http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
    // The map data, in row-major order, starting with (0,0). Occupancy
    // probabilities are in the range[0, 100]. Unknown is - 1.
    // Here, we treat the unknown state as the occupied state.
    for (size_t i = 0; i < msg.data.size(); ++i)
    {
        if (msg.data[i] == 0) this->global_costmap[i] = 0;
        else this->global_costmap[i] = 100;
    }
}

double TEBControllerWithMap::findDifferenceOrientation(double angle1, double angle2)
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

Twist TEBControllerWithMap::convertWheelToVelocity(WheelVelocity wheel_vel)
{
    double linear = (wheel_vel.left_vel + wheel_vel.right_vel) / 2;
    double angular = (wheel_vel.right_vel - wheel_vel.left_vel) / this->wheel_base;
    return Twist(linear, angular);
}

WheelVelocity TEBControllerWithMap::convertVelocityToWheelVelocity(double linear, double angular)
{
    double left_vel = linear - angular * 0.5 * this->wheel_base;
    double right_vel = linear + angular * 0.5 * this->wheel_base;
    return WheelVelocity(left_vel, right_vel);
}

bool TEBControllerWithMap::isFeasiblePath()
{
    for (int i = 0; i < this->best_path.size(); i++)
    {
        int x = this->contxy2disc(this->best_path[i].x() - this->original_position[0], this->resolution);
        int y = this->contxy2disc(this->best_path[i].y() - this->original_position[1], this->resolution);
        if (this->global_costmap[x + y * this->global_costmap_width] != 0) return false;
    }
    return true;
}

bool TEBControllerWithMap::goalReached()
{
    Eigen::Vector2d diff = this->robot_goal.position() - this->robot_pose.position();
    if (diff.norm() < this->goal_tolerance)
    {
        return true;
    }
    return false;
}

int TEBControllerWithMap::contxy2disc(double x, double cellsize)
{
    if (x >= 0)
    {
        return static_cast<int>(x / cellsize);
    } 
    else 
    {
        return static_cast<int>(x / cellsize) - 1;
    }
}