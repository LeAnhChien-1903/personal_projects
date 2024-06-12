#include "ultis.h"
#include "robot.h"
#include "cubic_spline.h"
#include "rrtstar.h"
#include <g2o/core/base_edge.h>
int main()
{
	cv::Mat map = cv::imread("map1.png", cv::IMREAD_COLOR);
	Pose2D init_pose, goal_pose;
	InputControl init_vel;
	init_pose.position.x = -13.0;
	init_pose.position.y = 13.0;
	init_pose.theta = 0.0;
	goal_pose.position.x = 13.0;
	goal_pose.position.y = -12.0;
	goal_pose.theta = 0.0;
	init_vel.left_vel = 0.0;
	init_vel.right_vel = 0.0;
	Robot robot(init_pose, goal_pose, init_vel, map, 0.1);
	std::vector<cv::Point> init_points = robot.robotVisualzation(init_pose);
	std::vector<cv::Point> goal_points = robot.robotVisualzation(goal_pose);
	cv::fillConvexPoly(map, init_points, cv::Scalar(0, 0, 255));
	cv::fillConvexPoly(map, goal_points, cv::Scalar(0, 255, 0));
	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image", map);
	cv::waitKey(0);
	int index = 1;
	while (robot.goal_reached == false)
	{
		cv::Mat visual = robot.robotControl();
		cv::imshow("Image", visual);
		std::ostringstream name;
		name << "D:/Local planner simulation/DWA_simulation/result/image" << index << ".png";
		cv::imwrite(name.str(), visual);
		if (cv::waitKey(1) && 0xFF == 'q') break;
		index++;
	}
	
	cv::destroyAllWindows();
	
	return 0;
}