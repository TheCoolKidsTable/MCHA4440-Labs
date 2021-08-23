#ifndef  BALLISTIC_PLOT_H
#define  BALLISTIC_PLOT_H


#include <Eigen/Core>

void plot_simulation(
    const Eigen::MatrixXd & thist, 
    const Eigen::MatrixXd & xhist, 
    const Eigen::MatrixXd & muhist, 
    const Eigen::MatrixXd & sigmahist, 
    const Eigen::MatrixXd & hhist, 
    const Eigen::MatrixXd & yhist);
    
#endif 