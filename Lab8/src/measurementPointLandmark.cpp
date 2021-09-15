#include "measurementPointLandmark.hpp"
#include <iostream>


#include <Eigen/Core>
#include <Eigen/QR>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>



void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi){
    assert(x.cols() == 1);
    const int nCameraStates     = 6;
    const int featureDim        = 3;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;
    
    // TODO

}
void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR){
    assert(x.cols() == 1);
    const int nCameraStates     = 6;
    const int featureDim        = 3;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;
    
    // TODO
}
void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & C){
    assert(x.cols() == 1);
    const int nCameraStates     = 6;
    const int featureDim        = 3;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;
    
    // TODO
}
