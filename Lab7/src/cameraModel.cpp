#define _USE_MATH_DEFINES

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>

#include <Eigen/Core>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cameraModel.hpp"
#include "gaussian.hpp"
#include "plot.h"
#include "rotation.hpp"
#include "settings.h"
#include "utility.h"

#define __DEBUG__(X) {std::cout << "In " << __FUNCTION__ << " at Line " << __LINE__ << ": " <<X << std::endl;};
#define DEBUG(X) __DEBUG__(X) 

static void write(cv::FileStorage& fs, const std::string&, const CameraParameters& x)
{
    x.write(fs);
}

void read(const cv::FileNode& node, CameraParameters& x, const CameraParameters& default_value){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

void CameraParameters::print() const
{

    std::bitset<8*sizeof(flag)> bitflag(flag);
    std::cout   << "Calibration data:" << std::endl;
    std::cout   << std::setw(30) << "cameraMatrix : " << std::endl << Kc         << std::endl;
    std::cout   << std::setw(30) << "distCoeffs : "   << std::endl << distCoeffs << std::endl;
    std::cout   << std::setw(30) << "flag : "                      << bitflag    << std::endl;
    std::cout   << std::setw(30) << "imageSize : "    << std::endl << imageSize  << std::endl;
    std::cout   << std::endl;
}

// Write serialization for this struct
void CameraParameters::write(cv::FileStorage& fs) const
{
    fs  << "{"
        << "camera_matrix"           << Kc
        << "distortion_coefficients" << distCoeffs
        << "flag"                    << flag
        << "imageSize"               << imageSize
        << "}";
}

// Read serialization for this struct
void CameraParameters::read(const cv::FileNode& node)
{
    node["camera_matrix"]           >> Kc;
    node["distortion_coefficients"] >> distCoeffs;
    node["flag"]                    >> flag;
    node["imageSize"]               >> imageSize;
}

// ------------------------------------------------------------
// Define camera calibration grid
// ------------------------------------------------------------
void generateCalibrationGrid(const Settings & s, std::vector<cv::Point3f> & rPNn_grid){
    // TODO: Copy from Lab 3 
    assert(0);
}
  

// ------------------------------------------------------------
// Export calibration data from a file
// ------------------------------------------------------------
void exportCalibrationData(const std::filesystem::path & calibrationFilePath, const CameraParameters & param){

    // TODO
    assert(0);
}

// ------------------------------------------------------------
// Import calibration data from a file
// ------------------------------------------------------------
void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param){

    // TODO
    assert(0);
}




// ------------------------------------------------------------
// Collect the points from the image set and then calibrate
// ------------------------------------------------------------
void calibrateCameraFromImageSet(const Settings & s, CameraParameters & param){

    std::vector<std::filesystem::path> imgFiles;
    imgFiles        = getFilesWithExtension(s.input_dir, s.input_ext);
    if (imgFiles.size()==0){
        std::cerr << "No files found in path " << s.input_dir << " with extension " << s.input_ext << std::endl;
        assert(0);
    }

    std::vector<std::vector<cv::Point2f> > rQOi_set;
    cv::Size imageSize;

    for(int k = 0; k<imgFiles.size();k++)
    {
        cv::Mat view;
        std::filesystem::path  inputPath;
        inputPath               = s.input_dir / imgFiles.at(k);
        view                    = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
     
        if(view.empty())          // If there are no more images stop the loop
        {
            break;
        }

        imageSize = view.size();  // Format input image.
        std::vector<cv::Point2f> rQOi;
        if (detectChessBoard(s, view, rQOi)){
            rQOi_set.push_back(rQOi);
            std::cout << "Found chess board corners in image: " << imgFiles.at(k) <<std::endl;
        }
        else{
            std::cout << "Found no chess board corners in image: " << imgFiles.at(k) <<std::endl;
        }

    }

    std::cout   << std::setw(30) << "No. images used : "            << rQOi_set.size() << std::endl;
    std::cout   << std::setw(30) << "inputImageSize : "             << imageSize << std::endl;

    runCalibration(s, rQOi_set, imageSize, param);
}

// ------------------------------------------------------------
// run calibration on a set of detected points
// ------------------------------------------------------------
void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param){

    std::vector<cv::Point3f> rPNn_base;
    std::vector<std::vector<cv::Point3f>> rPNn(1);
    generateCalibrationGrid(s, rPNn_base);

    
    double rms=-1;
    if( !rQOi_set.empty() ){
        // ------------------------------------------------------------
        // Run camera calibration
        // ------------------------------------------------------------
        // TODO: Copy from Lab3
        assert(0);

        // ------------------------------------------------------------
        // Write parameters to the data struct 
        // ------------------------------------------------------------
        // TODO


    }else{
        std::cerr << "No imagePoints found" << std::endl;
        assert(0);
    }
}

// ------------------------------------------------------------
// detectChessBoard in a single frame
// ------------------------------------------------------------
bool detectChessBoard(const Settings & s, const cv::Mat & view, std::vector<cv::Point2f> & rQOi ){

    // TODO: Copy from Lab3
    // return value should be the same as findChessboardCorners 
    assert(0);
    return false;

}

// ------------------------------------------------------------
// No touchy 
// https://i.pinimg.com/originals/c0/34/5a/c0345aa954c49970fe355ac19c773734.jpg
// ------------------------------------------------------------
bool getPoseFromCheckerBoardImage(const cv::Mat & view, const Settings & s, const CameraParameters & param, Eigen::VectorXd & eta){
    std::vector<cv::Point3f> rPNn;
    std::vector<cv::Point2f> rQOi;
    generateCalibrationGrid(s, rPNn);
    
    bool found = detectChessBoard(s, view, rQOi);
    if (!found){
        std::cout << "No chess board corners found. Exiting " << __FUNCTION__ << std::endl;
        return found;
    }

    cv::Mat Thetacn_rodrigues_cv, rNCc_cv;

    cv::solvePnP(rPNn, rQOi, param.Kc, param.distCoeffs, Thetacn_rodrigues_cv, rNCc_cv);

    cv::Mat outView, Rcn_cv;
    cv::Rodrigues(Thetacn_rodrigues_cv, Rcn_cv);

    Eigen::VectorXd rNCc(3);
    Eigen::MatrixXd Rcn(3,3), Rnc(3,3);

    cv2eigen(rNCc_cv, rNCc);
    cv2eigen(Rcn_cv, Rcn);
    
    Eigen::VectorXd rCNn, Thetanc;
    Rnc     = Rcn.transpose();
    rCNn    = - Rnc* rNCc;

    rot2rpy(Rnc, Thetanc);

    eta.resize(6,1);
    eta     <<  rCNn, 
                Thetanc;

    return found;
}



// ---------------------------------------------------------------------
// 
// WorldToPixelAdaptor
// 
// ---------------------------------------------------------------------
int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi){

    assert(rPNn.rows() == 3);
    assert(rPNn.cols() == 1);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    int err = worldToPixel(rPNn, eta, param, rQOi);
    if (err){
        return err;
    }

    assert(rQOi.rows() == 2);
    assert(rQOi.cols() == 1);

    return 0;
}

int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR){

    int res = operator()(rPNn, eta, param, rQOi);
    SR      = Eigen::MatrixXd::Zero(rQOi.rows(), rQOi.rows());
    return res;
}

int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J){

    assert(rPNn.rows() == 3);
    assert(rPNn.cols() == 1);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    // TODO
    // Use either the analytical expression or autodiff

    return -1;
}
