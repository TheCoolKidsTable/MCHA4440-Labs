#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>


#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>

#include "cameraModel.hpp"
#include "gaussian.hpp"
#include "plot.h"
#include "rotation.hpp"
#include "utility.h"




// Inputs
// H \in [0, 360]
// S \in [0, 1]
// V \in [0, 1]
// Outputs
// R \in [0, 1]
// R \in [0, 1]
// R \in [0, 1]
void hsv2rgb(const double & h, const double & s, const double & v, double & r, double & g, double & b){

    bool hIsValid = 0 <= h && h <=  360.0;
    bool sIsValid = 0 <= s && s <=  1.0;
    bool vIsValid = 0 <= v && v <=  1.0;

    assert(hIsValid);
    assert(sIsValid);
    assert(vIsValid);

    // https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB

    double  c, x, r1, g1, b1, m;
    int hp;
    // shift the hue to the range [0, 360] before performing calculations
    hp  = (int)(h / 60.);
    c   = v*s;
    x   = c * (1 - std::abs( std::fmod(h / 60, 2) - 1));

    switch(hp) {
        case 0: r1 = c; g1 = x; b1 = 0; break;
        case 1: r1 = x; g1 = c; b1 = 0; break;
        case 2: r1 = 0; g1 = c; b1 = x; break;
        case 3: r1 = 0; g1 = x; b1 = c; break;
        case 4: r1 = x; g1 = 0; b1 = c; break;
        case 5: r1 = c; g1 = 0; b1 = x; break;
    }
    m   = v - c;
    r   = r1 + m;
    g   = g1 + m;
    b   = b1 + m;
}



void plotGaussianConfidenceEllipse(cv::Mat & img, const Eigen::VectorXd & murQOi, const Eigen::MatrixXd & SrQOi, const Eigen::Vector3d & color){
    
    const int nx  = 2;
    assert(murQOi.rows() == nx);
    assert(SrQOi.rows() == nx);
    assert(SrQOi.cols() == nx);

    // TODO:
    // Copy some parts from plotFeatureGaussianConfidenceEllipse in Lab 7
}

