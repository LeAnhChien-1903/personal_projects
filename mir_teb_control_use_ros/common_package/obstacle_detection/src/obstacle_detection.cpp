#include "obstacle_detection.h"

bool ObstacleDetectorWithMap::initialize()
{
    if (!this->node.getParam("/obstacle_detector/scan_topic", this->scan_topic))
    {
        ROS_WARN(" Parameter '/scan_topic not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/obstacle_topic", this->obstacle_topic))
    {
        ROS_WARN(" Parameter '/obstacle_topic not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/amcl_topic", this->amcl_topic))
    {
        ROS_WARN(" Parameter '/amcl_topic not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/local_costmap_topic", this->local_costmap_topic))
    {
        ROS_WARN(" Parameter '/local_costmap_topic not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/start_topic", this->start_topic))
    {
        ROS_WARN(" Parameter '/start_topic not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/sample_time", this->sample_time))
    {
        ROS_WARN(" Parameter '/deltaT not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/robot_length", this->robot_length))
    {
        ROS_WARN(" Parameter '/robot_length not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/robot_width", this->robot_width))
    {
        ROS_WARN(" Parameter '/robot_width not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/min_distance_between_points", this->min_distance_between_points))
    {
        ROS_WARN(" Parameter '/min_distance_between_points not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_x", this->lidar_x))
    {
        ROS_WARN(" Parameter '/min_angle not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_y", this->lidar_y))
    {
        ROS_WARN(" Parameter '/lidar_y not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_z", this->lidar_z))
    {
        ROS_WARN(" Parameter '/lidar_z not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_roll", this->lidar_roll))
    {
        ROS_WARN(" Parameter '/lidar_roll not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_pitch", this->lidar_pitch))
    {
        ROS_WARN(" Parameter '/lidar_pitch not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/lidar_yaw", this->lidar_yaw))
    {
        ROS_WARN(" Parameter '/lidar_yaw not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/inflation_radius", this->inflation_radius))
    {
        ROS_WARN(" Parameter '/obstacle_detector/inflation_radius not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/height", this->map_height))
    {
        ROS_WARN(" Parameter '/obstacle_detector/height not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/width", this->map_width))
    {
        ROS_WARN(" Parameter '/obstacle_detector/width not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    if (!this->node.getParam("/obstacle_detector/resolution", this->resolution))
    {
        ROS_WARN(" Parameter '/obstacle_detector/resolution not set on %s node" , ros::this_node::getName().c_str());
        return false;
    }
    
    this->local_map = cv::Mat(int(this->map_height/this->resolution), int(this->map_width/this->resolution), CV_8UC1);
    this->map = cv::Mat(int(this->map_height/this->resolution), int(this->map_width/this->resolution), CV_8UC1);
    this->center = Eigen::Vector2d(int(this->map_width/this->resolution)/2 - 1, int(this->map_height/this->resolution)/2 - 1);
    this->robot_radius = sqrt(pow(this->robot_width/2, 2) + pow(this->robot_length/2, 2));
    
    this->amcl_sub = this->node.subscribe<geometry_msgs::PoseWithCovarianceStamped>(this->amcl_topic, 10, &ObstacleDetectorWithMap::acmlCallback, this);
    this->laser_scan_sub = this->node.subscribe<sensor_msgs::LaserScan>(this->scan_topic, 10, &ObstacleDetectorWithMap::laserScanCallback, this);
    this->start_sub = this->node.subscribe<std_msgs::Bool>(this->start_topic, 10, &ObstacleDetectorWithMap::startCallback, this);
    this->visualization_pub = this->node.advertise<visualization_msgs::MarkerArray>("/obstacle_visualization", 10);
    this->obstacle_pub = this->node.advertise<obstacle_detection::ObstacleVertices>(this->obstacle_topic, 10);
    this->local_costmap_pub = this->node.advertise<nav_msgs::OccupancyGrid>(this->local_costmap_topic, 10);

    dynamic_reconfigure::Server<obstacle_detection::ObstacleDetectionReconfigureConfig>::CallbackType cb = boost::bind(&ObstacleDetectorWithMap::dynamicConfigurationCallback, this, _1, _2);

    this->timer_ = this->node.createTimer(ros::Duration(this->sample_time), &ObstacleDetectorWithMap::timerCallback, this);
    this->dynamic_config_server.setCallback(cb);

    return true;
}

visualization_msgs::MarkerArray ObstacleDetectorWithMap::obstacleVisualization()
{
    visualization_msgs::MarkerArray markerArray;
    for (uint32_t index = 0; index < this->obstacles.size(); index++)
    {
        if (this->obstacles[index].size() < 2) continue;
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "map";
        marker.ns = "obstacles";
        marker.action = visualization_msgs::Marker::ADD;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.id = index;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.1; // line width
        marker.scale.y = 0.1; // line height
        marker.scale.z = 0.1; 
        
        marker.color.r = 1.0;;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        
        geometry_msgs::Point point;
        for (uint16_t index1 = 0; index1 <  this->obstacles[index].size(); index1++)
        {
            point.x = this->obstacles[index][index1].x();
            point.y = this->obstacles[index][index1].y();
            marker.points.push_back(point);
        }
        if (this->obstacles[index].size() > 2)
        {
            point.x = this->obstacles[index][0].x();
            point.y = this->obstacles[index][0].y();
            marker.points.push_back(point);
        }
        marker.lifetime = ros::Duration(0.05);
        markerArray.markers.push_back(marker);
    }

    return markerArray;
}

void ObstacleDetectorWithMap::acmlCallback(const geometry_msgs::PoseWithCovarianceStamped msg)
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
        float angle = normalize_angle(yaw);
        this->robot_pose.z() = yaw;
    }
}

void ObstacleDetectorWithMap::odometryCallback(const nav_msgs::Odometry msg)
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
        float angle = normalize_angle(yaw);
        this->robot_pose.z() = yaw;
    }
}
void ObstacleDetectorWithMap::laserScanCallback(const sensor_msgs::LaserScan msg)
{
    if (msg.ranges.size() > 0)
    {
        Eigen::Vector2d original_point(this->robot_pose.x(), this->robot_pose.y());
        // Clear the laser cluster 
        this->laser_scan.clear();
        Eigen::Vector3d lidar_pose; 
        lidar_pose.x() = this->robot_pose.x() + (this->lidar_x * cos(this->robot_pose.z()) - this->lidar_y * sin(this->robot_pose.z()));
        lidar_pose.y() = this->robot_pose.y() + (this->lidar_x * sin(this->robot_pose.z()) + this->lidar_y * cos(this->robot_pose.z()));
        lidar_pose.z() = normalize_angle(this->robot_pose.z() + this->lidar_roll);
        for (int i = 0; i < msg.ranges.size(); i++)
        {
            if (msg.ranges[i] > msg.range_min && msg.ranges[i] < msg.range_max)
            {
                LaserScanPoint laser_point;
                laser_point.range = msg.ranges[i];
                laser_point.angle = normalize_angle(msg.angle_min + i * msg.angle_increment + lidar_pose.z());
                laser_point.point.x() = lidar_pose.x() + laser_point.range * cos(laser_point.angle);
                laser_point.point.y() = lidar_pose.y() + laser_point.range * sin(laser_point.angle);
                if (abs(laser_point.point.x() - original_point.x()) < this->map_width/2 && abs(laser_point.point.y() - original_point.y()) < this->map_height/2)
                {
                    this->laser_scan.push_back(laser_point);
                }
                
            }
        }
        std::sort(this->laser_scan.begin(), this->laser_scan.end(), comparisonFunction);
    }
}

void ObstacleDetectorWithMap::startCallback(const std_msgs::Bool msg)
{
    this->startCommand = msg.data;
}

void ObstacleDetectorWithMap::timerCallback(const ros::TimerEvent &event)
{
    if (this->startCommand == true)
    {
        this->createCostmapFromLaserScan();
        this->extractObstacleFromLocalCostmap();
        this->extractEquivalentObstacles();
        this->associateObstacle();
        obstacle_detection::ObstacleVertices vertices_list;
        for (int i = 0; i < this->obstacles.size(); i++)
        {
            geometry_msgs::PoseArray vertices; 
            for (int j = 0; j < this->obstacles[i].size(); j++)
            {
                geometry_msgs::Pose pose;
                pose.position.x = this->obstacles[i][j].x();
                pose.position.y = this->obstacles[i][j].y();
                vertices.poses.push_back(pose);
            }
            vertices_list.vertices_list.push_back(vertices);
        }
        this->obstacle_pub.publish(vertices_list);
        visualization_msgs::MarkerArray obstacle_visual = this->obstacleVisualization();
        this->visualization_pub.publish(obstacle_visual);
        this->prev_obstacles = this->obstacles;
    }
}

void ObstacleDetectorWithMap::localCostmapCallback(const nav_msgs::OccupancyGrid local_costmap)
{
    // Iterate over the occupancy grid and copy the data to the cv::Mat.
    for (int y = 0; y < local_costmap.info.height; y++) {
        for (int x = 0; x < local_costmap.info.width; x++) {
            this->local_map.at<uint8_t>(y, x) = local_costmap.data[y * local_costmap.info.width + x];
        }
    }
}

void ObstacleDetectorWithMap::createCostmapFromLaserScan()
{
    // this->clusterPointCloud();
    Eigen::Vector2d original_point(this->robot_pose.x(), this->robot_pose.y());
    for (int i = 0; i < this->map.rows; i++)
    {
        for (int j = 0; j < this->map.cols; j++)
        {
            this->map.at<uint8_t>(i, j) = 0;
        }
    }
    for (int i = 0; i < this->laser_scan.size(); i++)
    {
        cv::Point point = convertMeterToPixel(this->laser_scan[i].point, original_point, this->resolution, this->map_width, this->map_height);
        cv::circle(this->map, point, int(this->inflation_radius/this->resolution), cv::Scalar(100), -1);
        // for (int j = 0; j < this->point_cloud_clusters[i].size(); j++)
        // {
        //     cv::Point point = convertMeterToPixel(this->point_cloud_clusters[i][j].point, original_point, this->resolution, this->map_width, this->map_height);
        //     cv::circle(this->map, point, int(this->inflation_radius/this->resolution), cv::Scalar(100), -1);
        // }
    }
    nav_msgs::OccupancyGrid map_convert;
    map_convert.info.width = int(this->map_width/this->resolution);
    map_convert.info.height = int(this->map_height/this->resolution);
    map_convert.info.resolution = this->resolution;
    map_convert.info.origin.position.x = this->robot_pose.x() - this->map_width/2;
    map_convert.info.origin.position.y = this->robot_pose.y() - this->map_height/2;
    map_convert.header.frame_id = "map";
    cv::flip(this->map, this->map, 0);
    map_convert.data.assign(this->map.data, this->map.data + this->map.total());

    this->local_costmap_pub.publish(map_convert);
}

void ObstacleDetectorWithMap::extractObstacleFromLocalCostmap()
{
    cv::Mat canny_map;
    // Extract edges by canny
    cv::Canny(this->map, canny_map, 100, 200);
    // Find contours from canny map 
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_map, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // Extract polygon from contours
    std::vector<std::vector<cv::Point>> contours_polygon(contours.size());
    for (int i = 0; i < contours.size(); ++i)
    {
        cv::approxPolyDP(contours[i], contours_polygon[i], 3, true);
    }
    // Filter vertices invalid
    std::vector<std::vector<cv::Point>> filtered_polygon;
    for (int i = 0; i < contours_polygon.size(); i++)
    {
        std::vector<cv::Point> polygon;
        for (int j = 0; j < contours_polygon[i].size(); j++)
        {
            if (contours_polygon[i][j].x >= 0 && contours_polygon[i][j].x < this->map.cols && contours_polygon[i][j].y >= 0 && contours_polygon[i][j].y < this->map.rows)
            {
                polygon.push_back(contours_polygon[i][j]);
            }
        }
        filtered_polygon.push_back(polygon);
    }
    // Convert to global coordinates
    this->obstacles.clear();
    for (int i = 0; i < filtered_polygon.size(); i++)
    {
        std::vector<Eigen::Vector2d> polygon;
        for (int j = 0; j < filtered_polygon[i].size(); j++)
        {
            double x = double((filtered_polygon[i][j].x - this->center.x())*this->resolution) + this->robot_pose.x();
            double y = double((filtered_polygon[i][j].y - this->center.y())*this->resolution) + this->robot_pose.y();
            polygon.push_back(Eigen::Vector2d(x,y));
        }
        this->obstacles.push_back(polygon);
    }
}

void ObstacleDetectorWithMap::extractEquivalentObstacles()
{
    std::vector<std::vector<Eigen::Vector2d>> obstacles_temp = this->obstacles;
    this->obstacles.clear();
    std::vector<std::vector<Eigen::Vector2d>> obstacles_class;
    obstacles_class.push_back(obstacles_temp[0]);
    for (int i = 1; i < obstacles_temp.size(); i++)
    {
        if (this->checkTwoEquivalentPolygon(obstacles_class.back(), obstacles_temp[i]) == true)
        {
            obstacles_class.push_back(obstacles_temp[i]);
        }
        else
        {
            this->obstacles.push_back(obstacles_class.front());
            obstacles_class.clear();
            obstacles_class.push_back(obstacles_temp[i]);
        }
    }

    this->obstacles.push_back(obstacles_class.front());
}

bool ObstacleDetectorWithMap::checkTwoEquivalentPolygon(std::vector<Eigen::Vector2d> polygon1, std::vector<Eigen::Vector2d> polygon2)
{
    if (polygon1.size() > polygon2.size() + 2 || polygon1.size() < polygon2.size() - 2) return false;
    std::vector<double> max_distance_list;
    for (int i = 0; i < polygon1.size(); i++)
    {
        double min_distance = INFINITY;
        for (int j = 0; j < polygon2.size(); j++)
        {
            double distance = (polygon1[i] - polygon2[j]).norm();
            if (distance < min_distance)
            {
                min_distance = distance;
            }
        }
        max_distance_list.push_back(min_distance);
    }
    // Calculate the center
    Eigen::Vector2d center1(0,0), center2(0,0);
    for (int i = 0; i < polygon1.size(); i++)
    {
        center1 += polygon1[i];
    }
    center1 /= polygon1.size();
    for (int j = 0; j < polygon2.size(); j++)
    {
        center2 += polygon2[j];
    }
    center2 /= polygon2.size();
    auto max_distance = std::max_element(max_distance_list.begin(), max_distance_list.end());
    if (*max_distance < this->min_distance_between_points && (center1 - center2).norm() < this->min_distance_between_points) return true;
    return false;
}

void ObstacleDetectorWithMap::associateObstacle()
{
    if (this->prev_obstacles.size() == 0) return;
    for (int i = 0; i < this->obstacles.size(); ++i)
    {
        for (int j = 0; j < this->prev_obstacles.size(); ++j)
        {
            if (this->checkTwoEquivalentPolygon(this->obstacles[i], this->prev_obstacles[j]) ==  true)
            {
                this->obstacles[i] = this->prev_obstacles[j];
                break;
            }
        }
    }
}

void ObstacleDetectorWithMap::dynamicConfigurationCallback(const obstacle_detection::ObstacleDetectionReconfigureConfig config, uint32_t level)
{
    if (config.enable_config == true)
    {
        this->inflation_radius = config.inflation_radius;
        this->min_distance_between_points = config.min_distance_between_points;
    }
    
}

void ObstacleDetectorWithMap::visualizeObstacleOpenCV(cv::Mat &map, double origin_x, double origin_y, double resolution, double map_height_pixel)
{
    for (int i = 0; i < this->obstacles.size(); i++)
    {
        if (this->obstacles[i].size() == 2)
        {
            PointPixel start = getPointPixel(this->obstacles[i][0].x(), this->obstacles[i][0].y(), origin_x, origin_y, resolution, map_height_pixel);
            PointPixel end = getPointPixel(this->obstacles[i][1].x(), this->obstacles[i][1].y(), origin_x, origin_y, resolution, map_height_pixel);
            cv::line(map, cv::Point(start.first, start.second), cv::Point(end.first, end.second), cv::Scalar(255,0,0), int(0.1 / resolution));
        }
        else
        {
            std::vector<cv::Point> points;
            for (int k = 0; k < this->obstacles[i].size(); k++)
            {
                PointPixel point = getPointPixel(this->obstacles[i][k].x(), this->obstacles[i][k].y(), origin_x, origin_y, resolution, map_height_pixel);
                points.push_back(cv::Point(point.first, point.second));
            }
            cv::fillConvexPoly(map, points, cv::Scalar(255, 0, 0));
        }
    }
}

double ObstacleDetectorWithMap::calculateAreaOfConvex(std::vector<Eigen::Vector2d> convex)
{
    double area = 0.0;
    // Calculate value of shoelace formula
    int j = convex.size() - 1;
    for (int i = 0; i < convex.size(); i++)
    {
        area += (convex[j].x() + convex[i].x()) * (convex[j].y() - convex[i].y());
        j = i;  // j is previous vertex to i
    }
    // Return absolute value
    return abs(area / 2.0);

}

bool ObstacleDetectorWithMap::isConvexObject(LaserPointCloud cluster)
{
    Eigen::Vector2d center(this->robot_pose.x(), this->robot_pose.y());
    double left_distance = (cluster.front() - center).norm() - extra_distance;
    double right_distance = (cluster.back() - center).norm() - extra_distance;

    return true;
}

std::vector<LineSegment> ObstacleDetectorWithMap::lineExtraction(LaserPointCloud cluster)
{
    std::vector<LineSegment> line_segments;
    if (cluster.size() < MINIMUM_POINTS_CHECK) return line_segments;

    // 1#: we initial a line from start to end
    //-----------------------------------------
    Eigen::Vector2d start = cluster.front();
    Eigen::Vector2d end = cluster.back();
    LineIndex l;
    l.first = 0;
    l.second = cluster.size() - 1;
    std::list<LineIndex> line_list;
    line_list.push_back(l);

    while (!line_list.empty()) 
    {
        // 2#: every time we take the first line in
        //line list to check if every point is on this line
        //-----------------------------------------
        LineIndex& lr = *line_list.begin();

        //
        if (lr.second - lr.first < MINIMUM_INDEX || lr.first == lr.second)
        {
            line_list.pop_front();
            continue;
        }

        // 3#: use two points to generate a line equation
        //-----------------------------------------
        start.x() = cluster[lr.first].x();
        start.y() = cluster[lr.first].y();
        end.x() = cluster[lr.second].x();
        end.y() = cluster[lr.second].y();

        // two points P1(x1, y1), P2(x2,y2) are given, and these two points are not the same
        // we can calculate an equation to model a line these two points are on.
        // A = y2 - y1
        // B = x1 - x2
        // C = x2 * y1 - x1 * y2
        double A = end.y() - start.y();
        double B = start.x() - end.x();
        double C = end.x() * start.y() - start.x() * end.y();

        double max_distance = 0;
        int max_i;
        int gap_i(-1);
        // the kernel code
        for (int i = lr.first + 1; i <= lr.second - 1; i++) 
        {
            // 4#: if two points' distance is too large, it's meaningless to generate a line
            // connects these two points, so we have to filter it.
            //-----------------------------------------
            double point_gap_dist = hypot(cluster[i].x() - cluster[i+1].x(), cluster[i].y() - cluster[i+1].y());
            if (point_gap_dist > MAXIMUM_GAP_DISTANCE) 
            {
                gap_i = i;
                break;
            }

            // 5#: calculate the distance between every point to the line
            //-----------------------------------------
            double dist = fabs(A * cluster[i].x() + B * cluster[i].y() + C) / hypot(A, B);
            if (dist > max_distance) 
            {
                max_distance = dist;
                max_i = i;
            }
        }

        // 6#: if gap is too large or there's a point is far from the line,
        // we have to split this line to two line, then check again.
        //-----------------------------------------
        if (gap_i != -1) 
        {
            int tmp = lr.second;
            lr.second = gap_i;
            LineIndex ll;
            ll.first = gap_i + 1;
            ll.second = tmp;
            line_list.insert(++line_list.begin(), ll);
        }
        else if (max_distance > IN_LINE_THRESHOLD) 
        {
            int tmp = lr.second;
            lr.second = max_i;
            LineIndex ll;
            ll.first = max_i + 1;
            ll.second = tmp;
            line_list.insert(++line_list.begin(), ll);
        } 
        else 
        {
            LineSegment line_;
            line_.push_back(cluster[line_list.front().first]);
            line_.push_back(cluster[line_list.front().second]);
            line_segments.push_back(line_);
            line_list.pop_front();
        }
    }
    return line_segments;
}

Eigen::Vector2d ObstacleDetectorWithMap::lineIntersection(double a1, double b1, double c1, double a2, double b2, double c2)
{
    double determinant = a1*b2 - a2*b1;
    Eigen::Vector2d intersection_point;
    intersection_point.x()  = (b2*c1 - b1*c2)/determinant;
    intersection_point.y() = (a1*c2 - a2*c1)/determinant;

    return intersection_point;
}

double ObstacleDetectorWithMap::findDifference(double angle1, double angle2)
{
    angle1 = normalize_angle(angle1);
    angle2 = normalize_angle(angle2);
    
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