#include "robot_model.h"

void teb_local_planner::PolygonRobot::transformToWorld(const PoseSE2 &current_pose, Point2dContainer &polygon_world) const
{
    double cos_th = cos(current_pose.theta());
    double sin_th = sin(current_pose.theta());
    for (std::size_t i = 0; i < this->vertices_.size(); ++i)
    {
        polygon_world[i].x() = current_pose.x() + cos_th * vertices_[i].x() - sin_th * this->vertices_[i].y();
        polygon_world[i].y() = current_pose.y() + sin_th * vertices_[i].x() + cos_th * this->vertices_[i].y();
    }
}

double teb_local_planner::PolygonRobot::calculateDistance(const PoseSE2 &current_pose, const Obstacle *obstacle) const
{
    Point2dContainer polygon_world(this->vertices_.size());
    this->transformToWorld(current_pose, polygon_world);
    return obstacle->getMinimumDistance(polygon_world);
}

double teb_local_planner::PolygonRobot::estimateSpatioTemporalDistance(const PoseSE2 &current_pose, const Obstacle *obstacle, double t) const
{
    Point2dContainer polygon_world(this->vertices_.size());
    this->transformToWorld(current_pose, polygon_world);
    return obstacle->getMinimumSpatioTemporalDistance(polygon_world, t);
}

void teb_local_planner::PolygonRobot::visualizeRobot(const PoseSE2 &current_pose, cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
{
    Point2dContainer polygon_world(this->vertices_.size());
    this->transformToWorld(current_pose, polygon_world);
    std::vector<cv::Point> polygon;
    for (int i = 0; i < this->vertices_.size(); i++)
    {
        Point2DPixel point(polygon_world[i].x(), polygon_world[i].y(), gain_x, gain_y, map_height);
        polygon.push_back(cv::Point(point.x, point.y));
    }
    cv::fillConvexPoly(map, polygon, color);
}

double teb_local_planner::PolygonRobot::getInscribedRadius()
{
    double min_dist = std::numeric_limits<double>::max();
    Eigen::Vector2d center(0.0, 0.0);

    if (this->vertices_.size() <= 2)
        return 0.0;

    for (int i = 0; i < (int)this->vertices_.size() - 1; ++i)
    {
        // compute distance from the robot center point to the first vertex
        double vertex_dist = this->vertices_[i].norm();
        double edge_dist = distance_point_to_segment_2d(center, this->vertices_[i], this->vertices_[i + 1]);
        min_dist = std::min(min_dist, std::min(vertex_dist, edge_dist));
    }

    // we also need to check the last vertex and the first vertex
    double vertex_dist = this->vertices_.back().norm();
    double edge_dist = distance_point_to_segment_2d(center, this->vertices_.back(), this->vertices_.front());
    return std::min(min_dist, std::min(vertex_dist, edge_dist));
}
