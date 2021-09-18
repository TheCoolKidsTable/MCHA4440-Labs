#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>

#include "imagefeatures.h"

struct SlamProcessModel
{
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & f, Eigen::MatrixXd & SQ);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd & dfdx);
};

struct SlamMeasurementModel
{
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & h, std::vector<Marker> & detected_markers);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & h, std::vector<Marker> & detected_markers, Eigen::MatrixXd & SR);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::VectorXd & h, std::vector<Marker> & detected_markers, Eigen::MatrixXd & SR, Eigen::MatrixXd & dhdx);
};


#endif