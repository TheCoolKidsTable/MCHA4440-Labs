#include <Eigen/Core>
#include <cassert>

#include "gaussian.hpp"


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// conditionGaussianOnMarginal
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

void conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::MatrixXd & Sxcond)
{
    // TODO: Copy from Lab 4
    
    int ny = y.rows();
    int nx = Syxjoint.cols() - ny;  

    Eigen::MatrixXd S1  = Syxjoint.topLeftCorner(ny,ny);
    Eigen::MatrixXd S2 =  Syxjoint.topRightCorner(ny,nx);
    Eigen::MatrixXd S3 =  Syxjoint.bottomRightCorner(nx,nx);
    Eigen::MatrixXd mux =  muyxjoint.head(nx);
    Eigen::MatrixXd muy = muyxjoint.tail(ny);

    muxcond = mux + S2.transpose()*(S1.triangularView<Eigen::Upper>().transpose().solve(y - muy));
    Sxcond = S3;
}

