#pragma once
#ifndef ROBOT_MODEL_H
#define ROBOT_MODEL_H

#include "pose_se2.h"
#include "obstacles.h"

namespace teb_local_planner
{
    /**
	 * @class PolygonRobot
	 * @brief Class that approximates the robot with a closed polygon
	 */
	class PolygonRobot
	{
	private:
		Point2dContainer vertices_;
		/**
		* @brief Transforms a polygon to the world frame manually
		* @param current_pose Current robot pose
		* @param[out] polygon_world polygon in the world frame
		*/
		void transformToWorld(const PoseSE2& current_pose, Point2dContainer& polygon_world) const;
	public:
		/**
		* @brief Default constructor of the abstract obstacle class
		* @param vertices footprint vertices (only x and y) around the robot center (0,0) (do not repeat the first and last vertex at the end)
		*/
		PolygonRobot(const Point2dContainer& vertices) : vertices_(vertices) {}
		/**
		* @brief Virtual destructor.
		*/
		~PolygonRobot() {}
		/**
		* @brief Set vertices of the contour/footprint
		* @param vertices footprint vertices (only x and y) around the robot center (0,0) (do not repeat the first and last vertex at the end)
		*/
		void setVertices(const Point2dContainer& vertices) {this->vertices_ = vertices; }
		/**
		* @brief Calculate the distance between the robot and an obstacle
		* @param current_pose Current robot pose
		* @param obstacle Pointer to the obstacle
		* @return Euclidean distance to the robot
		*/
		double calculateDistance(const PoseSE2& current_pose, const Obstacle* obstacle) const;
		/**
		* @brief Estimate the distance between the robot and the predicted location of an obstacle at time t
		* @param current_pose robot pose, from which the distance to the obstacle is estimated
		* @param obstacle Pointer to the dynamic obstacle (constant velocity model is assumed)
		* @param t time, for which the predicted distance to the obstacle is calculated
		* @return Euclidean distance to the robot
		*/
		double estimateSpatioTemporalDistance(const PoseSE2& current_pose, const Obstacle* obstacle, double t) const;
		/**
		* @brief Visualize the robot in map using OpenCV
		*
		* @param current_pose Current robot pose
		* @param map original map
		* @param color Color of the footprint
		* @param gain_x gain to add in x-coordinate
		* @param gain_y gain to add in y-coordinate
		* @param map_height height of map
		*/
		void visualizeRobot(const PoseSE2& current_pose, cv::Mat& map, const cv::Scalar& color,
									double gain_x, double gain_y, double map_height) const;
		/**
		* @brief Compute the inscribed radius of the footprint model
		* @return inscribed radius
		*/
		double getInscribedRadius();
	};
    //! Abbrev. for shared obstacle pointers
	typedef boost::shared_ptr<PolygonRobot> PolygonRobotPtr;
	//! Abbrev. for shared obstacle const pointers
	typedef boost::shared_ptr<const PolygonRobot> PolygonRobotConstPtr;
}
#endif