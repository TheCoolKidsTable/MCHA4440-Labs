#include <Eigen/Core>
#include <cassert>

#include "gaussian.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// conditionGaussianOnMarginal
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

void conditionGaussianOnMarginal(
    const Eigen::VectorXd & muyx, 
    const Eigen::MatrixXd & Syx, 
    const Eigen::VectorXd & y,
    Eigen::VectorXd & muxGy, 
    Eigen::MatrixXd & SxGy)
{
    // TODO: Copy from Lab 4
    assert(0);
}



// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// gaussianConfidenceEllipse
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

void gaussianConfidenceEllipse3Sigma(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, Eigen::MatrixXd & x){
    assert(mu.rows() == 2);
    assert(S.rows() == 2);
    assert(S.cols() == 2);

    int nsamples  = 100;

    // TODO: 
    assert(0);


    assert(x.cols() == nsamples);
    assert(x.rows() == 2);
}


void gaussianConfidenceQuadric3Sigma(const Eigen::VectorXd &mu, const Eigen::MatrixXd & S, Eigen::MatrixXd & Q){
    const int nx  = 3;
    assert(mu.rows() == nx);
    assert(S.rows() == nx);
    assert(S.cols() == nx);

    // TODO: 
    assert(0);

}
