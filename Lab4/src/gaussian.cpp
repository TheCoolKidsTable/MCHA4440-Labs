#include <iostream>
#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/QR>

#include "gaussian.h"

// Implement functions

void pythagoreanQR(const Eigen::MatrixXd & S1, const Eigen::MatrixXd & S2, Eigen::MatrixXd & S) {
    Eigen::MatrixXd A = Eigen::MatrixXd(S1.rows()+S2.rows(),S1.cols());
    A << S1, S2;
    auto QR = A.householderQr();
    Eigen::MatrixXd temp = QR.matrixQR().triangularView<Eigen::Upper>();
    S = temp.topRows(S1.cols());

}


int conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::
MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::
MatrixXd & Sxcond) {

    int ny = y.rows();
    int nx = Syxjoint.cols() - ny;  

    Eigen::MatrixXd S1  = Syxjoint.topLeftCorner(ny,ny);
    Eigen::MatrixXd S2 =  Syxjoint.topRightCorner(ny,nx);
    Eigen::MatrixXd S3 =  Syxjoint.bottomRightCorner(nx,nx);
    Eigen::MatrixXd mux =  muyxjoint.head(nx);
    Eigen::MatrixXd muy = muyxjoint.tail(ny);

    muxcond = mux + S2.transpose()*(S1.triangularView<Eigen::Upper>().transpose().solve(y - muy));
    Sxcond = S3;
    return 0;
}