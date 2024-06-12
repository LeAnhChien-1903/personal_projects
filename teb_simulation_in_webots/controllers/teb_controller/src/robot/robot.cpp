#include "robot.h"

TEBControl::TEBControl()
{
    this->timeStep = int(this->getBasicTimeStep());
    this->dt = double(timeStep) / 1000;

    this->initializeMotorAndPositionSensor();
    this->initializeSensor();

    this->wheel_base = 0.445208;
    this->wheel_radius = 0.0625;
    this->robot_length = 0.890;
    this->robot_width = 0.580;
    this->robot_radius = sqrt(pow(this->robot_length/2, 2) + pow(this->robot_width/2, 2));
    this->observable_range = 10; 
    this->max_left_velocity = 1.0;
    this->max_right_velocity = 1.0;
    this->min_left_velocity = -0.5;
    this->min_right_velocity = -0.5;
    this->max_acceleration = 2.0;
    
    this->extra_position << 0.387, 0.2305;
    this->front_transform.angle() = -M_PI_4;
    this->back_transform.angle() = 3 * M_PI_4;
    this->all_transform.setZero();
    this->all_transform(0,0) = 1;
    this->all_transform(1,1) = -1;
    this->min_range = this->lidar->getMinRange();
    this->max_range = 10.0;
    this->min_angle = M_PI/2;
    this->max_angle = -M_PI/2;
    this->resolution = this->lidar->getHorizontalResolution();
    this->angular_resolution = -DEG2RAD(0.5);

    this->prevLeftPosition = 0.0;
    this->prevRightPosition = 0.0;

    this->robot_goal = PoseSE2(6, 0, 0);
    this->map_height_pixel = (int) map_height_meter * meter_to_pixel;
    this->map_width_pixel = (int) map_width_meter * meter_to_pixel;
    this->gain_x = double(this->map_width_pixel * pixel_to_meter / 2);
    this->gain_y = double(this->map_height_pixel * pixel_to_meter / 2);

    this->obstacles = new ObstContainer;
    this->planner = PlannerInterfacePtr(new HomotopyClassPlanner(this->config));
    
}

void TEBControl::run()
{
    while (step(this->timeStep) != -1)
	{
        cv::Mat map_visual(this->map_height_pixel, this->map_width_pixel, CV_8UC3, cv::Scalar(255, 255, 255));
        std::cout << "--------------------------------" << std::endl;
        this->getRobotPose();
        this->getRobotVelocity();
        this->getLaserScanData();
        Twist vel = this->convertWheelToVelocity(this->cmd_vel);
        this->planner->plan(this->robot_pose, this->robot_goal, &vel);
        this->best_path.clear();
        this->planner->getBestTrajectory(this->best_path);
        WheelVelocity wheel_vel = this->extractWheelVelocity();
        this->setRobotVelocity(wheel_vel);
        // Visualize
        this->visualize(map_visual);
        // Display 
        webots::ImageRef *ir =  this->display->imageNew(this->map_width_pixel, this->map_height_pixel, map_visual.data, webots::Display::RGB);
        this->display->imagePaste(ir, 0, 0, false);
        this->display->imageDelete(ir);
	}
}


void TEBControl::visualize(cv::Mat &map)
{
    // Visualize the goal
    std::vector<cv::Point> goal_visual = this->extractRobotVisual(this->robot_goal);
    cv::fillConvexPoly(map, goal_visual, cv::Scalar(0, 255, 0));
    // Visualize robot
    std::vector<cv::Point> robot_visual = this->extractRobotVisual(this->robot_pose);
    cv::fillConvexPoly(map, robot_visual, cv::Scalar(255, 0, 0));
    // Visualize the robot position visited
    cv::polylines(map, this->robot_visited, false, cv::Scalar(255, 0, 0), int(0.05 * meter_to_pixel)); 
    // Visualize obstacles
    for (int i = 0; i < this->rectangles.size(); i++)
    {
        std::vector<cv::Point> points;
        for (int j = 0; j < this->rectangles[i].size(); j++)
        {
            Point2DPixel point(this->rectangles[i][j], this->gain_x, this->gain_y, this->map_height_pixel);
            points.push_back(cv::Point(point.x, point.y));
        }
        int blue = rand() % 255;
        int green = rand() % 255;
        int red = rand() % 255;
        cv::polylines(map, points, true, cv::Scalar(blue, green, red), int(0.05 * meter_to_pixel)); 
    }  
    // Visualize the best planner
    std::vector<cv::Point> path;
    for (int i = 0; i < this->best_path.size(); ++i)
    {
        Point2DPixel point(this->best_path[i].x(), this->best_path[i].y(), this->gain_x, this->gain_y, this->map_height_pixel);
        path.push_back(cv::Point(point.x, point.y));
    }
    cv::polylines(map, path, false, cv::Scalar(0, 255, 0), int(0.05 * meter_to_pixel));
}

void TEBControl::getLaserScanData()
{
    // Clear the laser cluster 
    this->laser_scan.clear();
    this->point_cloud_clusters.clear();

    // Get point cloud from front and left laser
    // const webots::LidarPoint *point_cloud = this->lidar->getPointCloud();
    const float *range_image = this->lidar->getRangeImage();
    for (int i = 0; i < this->resolution; i++)
    {
        if (range_image[i] > this->min_range && range_image[i] < this->max_range)
        {
            LaserScanPoint laser_point;
            laser_point.range = range_image[i];
            laser_point.angle = normalize_angle(this->min_angle + i * this->angular_resolution + this->robot_pose.theta());
            laser_point.point.x() = this->robot_pose.x() + laser_point.range * cos(laser_point.angle);
            laser_point.point.y() = this->robot_pose.y() + laser_point.range * sin(laser_point.angle);
            this->laser_scan.push_back(laser_point);
        }
    }
    std::sort(this->laser_scan.begin(), this->laser_scan.end(), comparisonFunction);
    this->clusterPointCloud();
    this->rectangleFitting();
}

WheelVelocity TEBControl::extractWheelVelocity()
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

double TEBControl::calculateCostWithReference(const PoseSE2 current_pose, const WheelVelocity current_vel)
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

DynamicWindow TEBControl::calculateDynamicWindow()
{
    DynamicWindow DW;
    DW.left_min_vel = std::max(this->min_left_velocity, this->cmd_vel.left_vel - this->max_acceleration * this->dt);
    DW.left_max_vel = std::min(this->max_left_velocity, this->cmd_vel.left_vel + this->max_acceleration * this->dt);
    DW.right_min_vel = std::max(this->min_right_velocity, this->cmd_vel.right_vel - this->max_acceleration * this->dt);
    DW.right_max_vel = std::min(this->max_right_velocity, this->cmd_vel.right_vel + this->max_acceleration * this->dt);
    return DW;
}

ReachableVelocity TEBControl::calculateReachableVelocity()
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

void TEBControl::clusterPointCloud()
{
    this->point_cloud_clusters.clear();

    double lambda = M_PI_4; // λ is an acceptable angle for determining the points to be of the same cluster
    const double omega_r = 0.1; // σr is the standard deviation of the noise of the distance measure
    
    LaserPointCloud point_block;
    LaserScanData point_list;
    
    point_list.push_back(this->laser_scan.front());
    point_block.push_back(this->laser_scan.front().point);

    for (unsigned int i = 1; i < this->laser_scan.size(); i++)
    {
        // Distance between two consecutive points
        double distance = (this->laser_scan[i].point - point_list.back().point).norm();
        // Delta theta between consecutive points
        double dTheta = this->laser_scan[i].angle - point_list.back().angle;

        // Calculate the distance threshold 
        double D_thd;
        if (abs(findDifferenceOrientation(this->laser_scan[i].angle,  point_list.back().angle)) >  4 * abs(this->angular_resolution)) D_thd = 0.0;
        else D_thd = std::min(point_list.back().range, this->laser_scan[i].range) * sin(dTheta) / sin(lambda - dTheta) + 3 * omega_r;
        if (distance < D_thd)
        {
            point_list.push_back(this->laser_scan[i]);
            point_block.push_back(this->laser_scan[i].point);
        }
        else
        {
            this->point_cloud_clusters.push_back(point_block);
            point_block.clear();
            point_list.clear();
            point_list.push_back(this->laser_scan[i]);
            point_block.push_back(this->laser_scan[i].point);
        }
    }
    this->point_cloud_clusters.push_back(point_block);
}

void TEBControl::rectangleFitting()
{
    // Clear the obstacles
    this->rectangles.clear();
    this->obstacles->clear();
    for (int i = 0; i < this->point_cloud_clusters.size(); i++)
    {
        if (this->isConvexObject(this->point_cloud_clusters[i]) == true)
        {
            uint16_t n = this->point_cloud_clusters[i].size();
            Eigen::VectorXd e1(2), e2(2);
            Eigen::MatrixXd X(n, 2); 
            
            for (uint16_t j = 0; j < n; j++)
            {
                X(j,0) = this->point_cloud_clusters[i][j].x();
                X(j,1) = this->point_cloud_clusters[i][j].y();
            }
            Eigen::VectorXd C1(n),C2(n);
            double q;
            double theta = 0.0;
            double step = M_PI/(2 * step_of_theta);
            Eigen::ArrayX2d Q(step_of_theta,2);
            for (int k = 0; k < step_of_theta; ++k) 
            {
                e1 << cos(theta), sin(theta);
                e2 <<-sin(theta), cos(theta);
                C1 = X * e1;
                C2 = X * e2;
                q = this->closenessCriterion(C1, C2, 0.0001) + this->areaCriterion(C1, C2);
                Q(k, 0) = theta;
                Q(k, 1) = q;

                theta += step;
            }
            Eigen::ArrayX2d::Index max_index;
            Q.col(1).maxCoeff(&max_index);//find Q with maximum value
            theta = Q(max_index,0);
            e1 << cos(theta), sin(theta);
            e2 <<-sin(theta), cos(theta);
            C1 = X * e1;
            C2 = X * e2;
            
            double a1 = cos(theta);
            double b1 = sin(theta);
            double c1 = C1.minCoeff();
            
            double a2 = -sin(theta);
            double b2 = cos(theta);
            double c2 = C2.minCoeff();

            double a3 = cos(theta);
            double b3 = sin(theta);
            double c3 = C1.maxCoeff();
            
            double a4 = -sin(theta);
            double b4 = cos(theta);
            double c4 = C2.maxCoeff();

            std::vector<Eigen::Vector2d> corners;
            corners.push_back(lineIntersection(a1, b1, c1, a2, b2, c2));
            corners.push_back(lineIntersection(a2, b2, c2, a3, b3, c3));
            corners.push_back(lineIntersection(a3, b3, c3, a4, b4, c4));
            corners.push_back(lineIntersection(a1, b1, c1, a4, b4, c4));   
            double edge_1 = (corners[0] - corners[1]).norm();
            double edge_2 = (corners[1] - corners[2]).norm();
            if (std::max(edge_1, edge_2) / std::min(edge_1, edge_2) > 5)
            {
                std::vector<LineSegment> lineSegments = this->lineExtraction(this->point_cloud_clusters[i]);
                for (int k = 0; k < lineSegments.size(); k++)
                {
                    this->rectangles.push_back(lineSegments[k]);
                    this->obstacles->push_back(ObstaclePtr(new LineObstacle(lineSegments[k][0], lineSegments[k][1])));
                }
            }
            else
            {
                this->rectangles.push_back(corners);
                Point2dContainer vertices;
                for (int k = 0; k < corners.size(); k++)
                {
                    vertices.push_back(corners[k]);
                }
                this->obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
            }
        }
        else
        {
            std::vector<LineSegment> lineSegments = this->lineExtraction(this->point_cloud_clusters[i]);
            for (int k = 0; k < lineSegments.size(); k++)
            {
                this->rectangles.push_back(lineSegments[k]);
                this->obstacles->push_back(ObstaclePtr(new LineObstacle(lineSegments[k][0], lineSegments[k][1])));
            }
        }
    }
    this->planner->updateObstacles(this->obstacles);
}

bool TEBControl::isConvexObject(LaserPointCloud cluster)
{
    double left_distance = (cluster.front() - this->robot_pose.position()).norm() - extra_distance;
    double right_distance = (cluster.back() - this->robot_pose.position()).norm() - extra_distance;
    double middle_distance = 0;
    for (int i = 1; i < cluster.size() - 1; i++)
    {
        middle_distance += (cluster[i] - this->robot_pose.position()).norm();
    }
    middle_distance = double(middle_distance/(cluster.size() - 2));
    if (middle_distance <= right_distance && middle_distance <= left_distance) return true;
    return false;
}
std::vector<LineSegment> TEBControl::lineExtraction(LaserPointCloud cluster)
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
Eigen::Vector2d TEBControl::lineIntersection(double a1, double b1, double c1, double a2, double b2, double c2)
{
    double determinant = a1*b2 - a2*b1;
    Eigen::Vector2d intersection_point;
    intersection_point.x()  = (b2*c1 - b1*c2)/determinant;
    intersection_point.y() = (a1*c2 - a2*c1)/determinant;

    return intersection_point;
}

double TEBControl::areaCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2)
{
    double c1_max = C1.maxCoeff();
    double c1_min = C1.minCoeff();
    double c2_max = C2.maxCoeff();
    double c2_min = C2.minCoeff();

    double alpha = -(c1_max - c1_min) * (c2_max - c2_min);

    return alpha;
}

double TEBControl::closenessCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2, const double &d0)
{
    double c1_max = C1.maxCoeff();
    double c1_min = C1.minCoeff();
    double c2_max = C2.maxCoeff();
    double c2_min = C2.minCoeff();

    Eigen::VectorXd C1_max = c1_max - C1.array(); 
    Eigen::VectorXd C1_min = C1.array() - c1_min;
    Eigen::VectorXd D1, D2;
    if(C1_max.squaredNorm() < C1_min.squaredNorm()){
        D1 = C1_max;
    }
    else{
        D1 = C1_min;
    }
    Eigen::VectorXd C2_max = c2_max - C2.array(); 
    Eigen::VectorXd C2_min = C2.array() - c2_min;
    if(C2_max.squaredNorm() < C2_min.squaredNorm()){
        D2 = C2_max;
    }
    else{
        D2 = C2_min;
    }

    double d, min;
    double b = 0 ;
    for (int i = 0; i < D1.size(); ++i) 
    {
        min = std::min(D1(i),D2(i));
        d = std::max(min, d0);
        b = b + 1/d;
    }
    
    return b; 
}

double TEBControl::varianceCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2)
{
    double c1_max = C1.maxCoeff();
    double c1_min = C1.minCoeff();
    double c2_max = C2.maxCoeff();
    double c2_min = C2.minCoeff();

    Eigen::VectorXd C1_max = c1_max - C1.array(); 
    Eigen::VectorXd C1_min = C1.array() - c1_min;
    Eigen::VectorXd D1, D2;
    if(C1_max.squaredNorm() < C1_min.squaredNorm()){
        D1 = C1_max;
    }
    else{
        D1 = C1_min;
    }
    Eigen::VectorXd C2_max = c2_max - C2.array(); 
    Eigen::VectorXd C2_min = C2.array() - c2_min;
    if(C2_max.squaredNorm() < C2_min.squaredNorm()){
        D2 = C2_max;
    }
    else{
        D2 = C2_min;
    }
    Eigen::VectorXd E1(D1.size()), E2(D2.size());
    for (int i = 0; i < E1.size(); i++)
    {
        if (D1(i) < D2(i)) E1(i) = D1(i);
        if (D2(i) < D1(i)) E2(i) = D2(i);
    }
    double gamma = - sqrt((E1.array() - E1.mean()).square().sum() / (E1.size() - 1)) - sqrt((E2.array() - E2.mean()).square().sum() / (E2.size() - 1));
    return 0.0;
}

void TEBControl::setRobotVelocity(WheelVelocity wheel_vel)
{
    this->left_motor->setVelocity(wheel_vel.left_vel/this->wheel_radius);
    this->right_motor->setVelocity(wheel_vel.right_vel/this->wheel_radius);
    this->cmd_vel = wheel_vel;
}

void TEBControl::initializeMotorAndPositionSensor()
{
    this->left_motor = this->getMotor("middle_left_wheel_joint");
    this->right_motor = this->getMotor("middle_right_wheel_joint");
    this->left_sensor = this->getPositionSensor("middle_left_wheel_joint_sensor");
    this->right_sensor = this->getPositionSensor("middle_right_wheel_joint_sensor");

    // Set position and velocity
    this->left_motor->setPosition(INFINITY);
    this->right_motor->setPosition(INFINITY);
    this->left_motor->setVelocity(0.0);
    this->right_motor->setVelocity(0.0);
    // Enable position sensor
    this->left_sensor->enable(this->timeStep);
    this->right_sensor->enable(this->timeStep);
}

void TEBControl::initializeSensor()
{
    // LiDAR sensor
	// this->front_lidar = this->getLidar("front_lidar");
	// this->back_lidar = this->getLidar("back_lidar");
	// this->front_lidar->enable(this->timeStep);
	// this->back_lidar->enable(this->timeStep);
    // this->front_lidar->enablePointCloud();
    // this->back_lidar->enablePointCloud();
    this->lidar = this->getLidar("lidar");
    this->lidar->enable(this->timeStep);
    this->lidar->enablePointCloud();
    // Inertial unit sensor
	this->iu = this->getInertialUnit("iu_sensor");
	this->iu->enable(this->timeStep);
    // Distance sensor
	this->ds0 = this->getDistanceSensor("ds0");
	this->ds1 = this->getDistanceSensor("ds1");
	this->ds2 = this->getDistanceSensor("ds2");
	this->ds3 = this->getDistanceSensor("ds3");
    this->ds0->enable(this->timeStep);
	this->ds1->enable(this->timeStep);
	this->ds2->enable(this->timeStep);
	this->ds3->enable(this->timeStep);
    // GPS sensor
	this->gps = this->getGPS("gps");
	this->gps->enable(this->timeStep);
	// Display
	this->display = this->getDisplay("display");
}

bool TEBControl::goalReached()
{
    Eigen::Vector2d diff = this->robot_goal.position() - this->robot_pose.position();
    if (diff.norm() < 0.05)
    {
        return true;
    }
    return false;
}

double TEBControl::findDifferenceOrientation(double angle1, double angle2)
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

Twist TEBControl::convertWheelToVelocity(WheelVelocity wheel_vel)
{
    double linear = (wheel_vel.left_vel + wheel_vel.right_vel) / 2;
    double angular = (wheel_vel.right_vel - wheel_vel.left_vel) / this->wheel_base;
    return Twist(linear, angular);
}

void TEBControl::getRobotPose()
{
    const double *position = this->gps->getValues();
    this->robot_pose.x() = position[0];
    this->robot_pose.y() = position[1];
    const double *orientation = this->iu->getRollPitchYaw();
    this->robot_pose.theta() = normalize_angle(orientation[2]);
    Point2DPixel visited_point(this->robot_pose.position(), this->gain_x, this->gain_y, this->map_height_pixel);
    this->robot_visited.push_back(cv::Point(visited_point.x, visited_point.y));
}   

void TEBControl::getRobotVelocity()
{
    // Get current wheel position
    double current_left = this->left_sensor->getValue();
	double current_right = this->right_sensor->getValue();

    // Calculate the velocity of the wheels
    this->cmd_vel.left_vel = (current_left - this->prevLeftPosition) * this->wheel_radius / this->dt; // velocity of left wheel (rad/s)
	this->cmd_vel.right_vel = (current_right - this->prevRightPosition) * this->wheel_radius / this->dt; // velocity of right wheel (rad/s)
	
    // Update previous left and right position 
	this->prevLeftPosition = current_left;
	this->prevRightPosition = current_right;
}

std::vector<cv::Point> TEBControl::extractRobotVisual(PoseSE2 pose)
{
    std::vector<cv::Point> visual;
    std::vector<Point2D> corners = extractCoorner(Pose2D(pose.x(), pose.y(), pose.theta()), this->robot_length, this->robot_width);
    Point2DPixel point(corners[0], this->gain_x, this->gain_y, this->map_height_pixel);
    visual.push_back(cv::Point(point.x, point.y));
    Point2DPixel point1(corners[1], this->gain_x, this->gain_y, this->map_height_pixel);
    visual.push_back(cv::Point(point1.x, point1.y));
    Point2DPixel point2(0.5 *(corners[2].x + corners[3].x), 0.5 *(corners[2].y + corners[3].y), this->gain_x, this->gain_y, this->map_height_pixel);
    visual.push_back(cv::Point(point2.x, point2.y));
    return visual;
}