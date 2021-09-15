#ifndef PLOT_H
#define PLOT_H

#include <Eigen/Core>
#include <opencv2/core.hpp>

// -------------------------------------------------------
// Function prototypes
// -------------------------------------------------------
void hsv2rgb(const double & h, const double & s, const double & v, double & r, double & g, double & b);
void plotGaussianConfidenceEllipse(cv::Mat & img, const Eigen::VectorXd & murQOi, const Eigen::MatrixXd & SrQOi, const Eigen::Vector3d & color);


#endif
