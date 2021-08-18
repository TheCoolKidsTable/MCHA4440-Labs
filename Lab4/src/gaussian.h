#ifndef	GAUSSIAN_H
#define	GAUSSIAN_H

// Libraries needed to understand the header only
#include <Eigen/Core>

// Function prototypes
void pythagoreanQR(const Eigen::MatrixXd & S1, const Eigen::MatrixXd & S2, Eigen::MatrixXd & S);


int conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::
MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::
MatrixXd & Sxcond);

#endif