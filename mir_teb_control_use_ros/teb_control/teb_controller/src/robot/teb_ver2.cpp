#include "teb_ver2.h"

TEBVersion2::TEBVersion2()
{
    this->sample_time = 0.05;
    this->robot_length = 0.89;
    this->robot_width = 0.58;
    this->max_linear_velocity = 1.0;
    this->max_angular_velocity = 1.0;
    this->min_linear_velocity = -0.5;
    this->min_angular_velocity = -1.0;
    this->max_linear_acceleration = 2.0;
    this->max_angular_acceleration = 2.0;
    
    this->robot_radius = sqrt(pow(this->robot_length/2, 2) + pow(this->robot_width/2, 2));

    this->config.obstacles.min_obstacle_dist = this->robot_radius;
    this->config.obstacles.inflation_dist = this->robot_radius;
    this->config.obstacles.dynamic_obstacle_inflation_dist = this->robot_radius + 0.1;
    this->config.robot.max_vel_x = this->max_linear_velocity;
    this->config.robot.max_vel_x_backwards = this->min_linear_velocity;
    this->config.robot.max_vel_theta = 1.0;
    this->config.robot.acc_lim_x = this->max_linear_acceleration;
    this->config.robot.acc_lim_theta = this->max_angular_acceleration;

    this->robot_pose = PoseSE2(0.0, 0.0, 0.0);
    this->local_goal = PoseSE2(0.0, 0.0, 0.0);
    this->local_index = 0;

    this->obstacles = new ObstContainer;
    this->via_points = new ViaPointContainer;
    this->planner = PlannerInterfacePtr(new HomotopyClassPlanner(this->config, this->obstacles, this->via_points));
}

std::pair<std::vector<PoseSE2>, int> TEBVersion2::runTEB(PoseSE2 current_pose, PoseSE2 global_goal, int index_of_collision, Twist current_vel, std::vector<PoseSE2> global_trajectories, std::vector<std::vector<Eigen::Vector2d>> obstacles_)
{
    std::pair<std::vector<PoseSE2>, int> local_trajectories;
    this->robot_pose = current_pose;
    this->updateObstacles(obstacles_);
    this->findLocalGoal(obstacles_, index_of_collision, global_trajectories);
    bool success;
    if (this->local_goal.x() == global_goal.x() && this->local_goal.y() == global_goal.y())
    {
        success = this->planner->plan(this->robot_pose, this->local_goal, &this->current_vel, false);
    }
    else
    {
        success = this->planner->plan(this->robot_pose, this->local_goal, &this->current_vel, true);
    }
    if (!success)
    {
        this->planner->clearPlanner(); // force re-initialization for next time
        local_trajectories.second = -1;
        return local_trajectories;
    }
    if (this->planner->hasDiverged())
    {
        this->planner->clearPlanner(); // force re-initialization for next time
        local_trajectories.second = -1;
        return local_trajectories;
    }
    // Get best trajectory and time difference
    this->planner->getBestTrajectory(local_trajectories.first);
    local_trajectories.second = 0;
    if (this->localGoalReached() == true)
    {
        local_trajectories.second = 1; // Have no local goal
    }
    
    return local_trajectories;
}

void TEBVersion2::updateObstacles(std::vector<std::vector<Eigen::Vector2d>> obstacles_)
{
    this->obstacles->clear();
    for (int i = 0; i < obstacles_.size(); i++)
    {
        if (obstacles_[i].size() == 2 )
        {
            this->obstacles->push_back(ObstaclePtr(new LineObstacle(obstacles_[i][0], obstacles_[i][1])));
        }
        else
        {
            Point2dContainer vertices;
            for (int k = 0; k < obstacles_[i].size(); k++)
            {
                vertices.push_back(obstacles_[i][k]);
            }
            this->obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
        }
    }

    this->planner->updateObstacles(this->obstacles);
}

bool TEBVersion2::checkSafetyOfLocalGoal(std::vector<std::vector<Eigen::Vector2d>> obstacles_)
{
    if (obstacles_.size() == 0)
    {
        return true;
    }
    else
    {
        for (int i = 0; i < obstacles_.size(); i++)
        {
            if (this->checkCircleIntersectPolygon(this->local_goal.position(), 0.6, obstacles_[i]))
            {
                return false;
            } 
        }
    }
    return true;
}

void TEBVersion2::findLocalGoal(std::vector<std::vector<Eigen::Vector2d>> obstacles_, int index_of_collision, std::vector<PoseSE2> global_trajectories)
{
    int collision_index;
    if (this->local_index == 0)
    {
        collision_index = index_of_collision + 1;
    }
    else 
    {
        if (this->checkSafetyOfLocalGoal(obstacles_) == true)
        {
            return;
        }
        collision_index = this->local_index + 1;
    }
    bool flag;
    while (true)
    {
        flag = false;
        for (int i = 0; i < obstacles_.size(); i++)
        {
            if (this->checkCircleIntersectPolygon(global_trajectories[collision_index].position(), 0.6, obstacles_[i]))
            {
                collision_index += 2;
                flag = true;
                break;
            } 
        }
        if (flag == false)
        {
            this->local_goal = global_trajectories[collision_index];
            local_index = collision_index;
            break;
        }
    }
}

bool TEBVersion2::checkCircleIntersectSegment(Eigen::Vector2d center, double radius, Eigen::Vector2d line_start, Eigen::Vector2d line_end)
{
    // Distance Line - Circle
    // refer to http://www.spieleprogrammierer.de/wiki/2D-Kollisionserkennung#Kollision_Kreis-Strecke
    Eigen::Vector2d a = line_end - line_start; // not normalized!  a=y-x
    Eigen::Vector2d b = center - line_start; // b=m-x

    // Now find nearest point to circle v=x+a*t with t=a*b/(a*a) and bound to 0<=t<=1
    double t = a.dot(b)/a.dot(a);
    if (t<0) t=0; // bound t (since a is not normalized, t can be scaled between 0 and 1 to parametrize the line
    else if (t>1) t=1;
    Eigen::Vector2d nearest_point = line_start + a*t;

    // check collision
    double distance = (nearest_point - center).norm() - radius;
    
    return (distance < 0.05);
}

bool TEBVersion2::checkCircleIntersectPolygon(Eigen::Vector2d center, double radius, std::vector<Eigen::Vector2d> polygon)
{
    for (int i = 0; i < polygon.size(); i++)
    {
        if (i == polygon.size() - 1)
        {
            if (this->checkCircleIntersectSegment(center, radius, polygon[i], polygon[0]) == true) return true;
        }
        else
        {
            if (this->checkCircleIntersectSegment(center, radius, polygon[i], polygon[i+1]) == true) return true;
        }
    }

    return false;
}


bool TEBVersion2::localGoalReached()
{
    Eigen::Vector2d diff = this->local_goal.position() - this->robot_pose.position();
    if (diff.norm() < this->goal_tolerance)
    {
        this->local_index = 0;
        return true;
    }
    return false;
}

double TEBVersion2::findDifferenceOrientation(double angle1, double angle2)
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

int TEBVersion2::contxy2disc(double x, double cellsize)
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

std::vector<Eigen::Vector2d> TEBVersion2::extractRobotBoundingBox(PoseSE2 pose)
{
    std::vector<Eigen::Vector2d> bounding_box;
	double x = pose.x();
	double y = pose.y();
	double halfLength = this->robot_length / 2;
	double halfWidth = this->robot_width / 2;
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
