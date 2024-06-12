#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H
#include "ultis.h"


class CubicSpline1D
{
private:
	std::vector<double> sample_input, x_input;
public:
	std::vector<double> x, dx, ddx, s;
	CubicSpline1D();
	~CubicSpline1D();
	void initialization(const std::vector<double>, const std::vector<double>, const double);
	void getCubicData(const MatrixXd, const MatrixXd, const MatrixXd, const MatrixXd, const double);
	MatrixXd calculateMatrixA(const VectorXd, const int);
	VectorXd calculateMatrixB(const VectorXd, const VectorXd, const int);
	VectorXd differenceInVector(const VectorXd);
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