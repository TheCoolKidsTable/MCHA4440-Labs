#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>

template<typename Scalar>
void rot2rpy(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>  & Rnc, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & Thetanc){
    assert(Rnc.rows() == 3);
    assert(Rnc.cols() == 3);
    
    // TODO: 
    assert(0);
}

template<typename Scalar>
void rotx(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // TODO: 
    assert(0);
}

template<typename Scalar>
void roty(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // TODO: 
    assert(0);
}

template<typename Scalar>
void rotz(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // TODO: 
    assert(0);
}

template<typename Scalar>
void rpy2rot(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & Thetanc, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    assert(Thetanc.rows() == 3);

    // TODO: 
    assert(0);
}




#endif