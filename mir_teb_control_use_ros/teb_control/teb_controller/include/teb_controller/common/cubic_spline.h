#pragma once
#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H
#include "ultis.h"


class CubicSpline1D
{
private:
	std::vector<double> sample_input, x_input;
    double ds;
public:
	std::vector<double> x, dx, ddx, s;
	CubicSpline1D();
	~CubicSpline1D();
	void initialization(const std::vector<double> s, const std::vector<double> x, const double ds);
	void getCubicData(const Eigen::MatrixXd a, const Eigen::MatrixXd b, const Eigen::MatrixXd c, const Eigen::MatrixXd d, const double ds);
	Eigen::MatrixXd calculateMatrixA(const Eigen::VectorXd h, const int numberX);
	Eigen::VectorXd calculateMatrixB(const Eigen::VectorXd h, const Eigen::VectorXd a, const int numberX);
	Eigen::VectorXd differenceInVector(const Eigen::VectorXd h);
};
class CubicSpline2D
{
private:
	CubicSpline1D x_cubic, y_cubic;
	double ds;
	std::vector<double> sample;
public:
	CubicSpline2D();
	~CubicSpline2D();
	void initialization(Waypoints, const double, const double);
	ReferencePath computeCubicPath();
};

#endif