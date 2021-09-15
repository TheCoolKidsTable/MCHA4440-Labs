#ifndef MEASUREMENT_POINT_LANDMARK_SINGLE_H
#define MEASUREMENT_POINT_LANDMARK_SINGLE_H

#include <Eigen/Core>
#include "cameraModel.hpp"


struct MeasurementPointLandmarkSingle{
    template<typename DerivedOutA>
    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, const Eigen::MatrixBase<DerivedOutA>  & rQOi_){
        assert(x.cols() == 1);
        const int nCameraStates = 6;
        const int featureDim    = 3;
        int nLandmarkStates      = x.rows() - nCameraStates;
        int nLandmarks           = nLandmarkStates / featureDim;

        // Check that there are feature states
        assert(nLandmarkStates > 0);
        assert(j >= 0);
        // Check that the number of states for features is a multiple of featureDim
        assert((nLandmarkStates%featureDim) == 0);

        // TODO:
        // Some call to worldToPixel
        Eigen::MatrixBase<DerivedOutA> & rQOi = const_cast< Eigen::MatrixBase<DerivedOutA> &>(rQOi_);
        rQOi.derived().resize(2,1);
        Eigen::VectorXd eta = x.head(nCameraStates);
        Eigen::VectorXd rPNn = x.segment(nCameraStates+j*featureDim,3);
        int res = worldToPixel(rPNn,eta,param,rQOi); // what ever the return value of worldToPixel would be

        return res;
    }

    template<typename DerivedOutA, typename DerivedOutB>
    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, const Eigen::MatrixBase<DerivedOutA>  & rQOi_, const Eigen::MatrixBase<DerivedOutB>  & SR_){
        int res = operator()(j, x, param, rQOi_);
        
        Eigen::MatrixBase<DerivedOutB> & SR     = const_cast<Eigen::MatrixBase<DerivedOutB> & >(SR_);
        
        if (SR.size()==0){
            SR.derived().resize(2,2);
        }
    
        // TODO
        // SR
        SR(0,0) = 0.01;
        SR(0,1) = 0.0;
        SR(1,1) = 0.01;
        SR(1,0) = 0.0;

        return res;
    }

    template<typename DerivedOutA, typename DerivedOutB, typename DerivedOutC>
    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, const Eigen::MatrixBase<DerivedOutA> & rQOi_, const Eigen::MatrixBase<DerivedOutB> & SR_, const Eigen::MatrixBase<DerivedOutC> & J_){
        int res = operator()(j, x, param, rQOi_, SR_);

        assert(x.cols() == 1);

        const int nCameraStates = 6;
        const int featureDim    = 3;
        int nLandmarkStates     = x.rows() - nCameraStates;
        int nLandmarks          = nLandmarkStates / featureDim;

        Eigen::MatrixBase<DerivedOutC> & J      = const_cast<Eigen::MatrixBase<DerivedOutC> & >(J_);
        J.derived().resize(2,6 + 3*nLandmarks);

        assert(nLandmarkStates > 0);
        assert(j >= 0);
        // Check that the number of states for features is a multiple of featureDim
        assert((nLandmarkStates%featureDim) == 0);

        return res;
    }
};

struct MeasurementPointLandmarkBundle{
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi);
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR);
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J);
};


#endif