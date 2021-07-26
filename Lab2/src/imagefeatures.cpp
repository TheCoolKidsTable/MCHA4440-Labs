#include <iostream>
#include <string>  
#include <cstdlib>
#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>

#include "imagefeatures.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

struct features {
    int x;
    int y;
    float score;
};

bool sorter(features const& lhs, features const& rhs) {
        return lhs.score > rhs.score;
}

std::vector<features> detected_features;

int detectAndDrawHarris(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    
    // TODO:
    std::cout << "Using harris feature detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;
    int max_thresh = 255;
    int no_features_detected = 0;   
    const char* corners_window = "Corners detected";

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cornerHarris(img_gray, dst, blockSize, apertureSize, k);
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    for(int i = 0; i < dst_norm.rows ; i++)
    {
        for(int j = 0; j < dst_norm.cols; j++)
        {
            if((int) dst_norm.at<float>(i,j) > thresh)
            {
                no_features_detected++;
                features new_feature = {i,j,dst.at<float>(i,j)};
                detected_features.push_back(new_feature);
                cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
        }
    }

    std::cout << "Features detected: " << no_features_detected << std::endl;
    std::cout << "Test: " << detected_features[0].x << std::endl;
    //Sort struct
    std::sort(detected_features.begin(), detected_features.end(), &sorter);
    for(int i = 0; i < maxNumFeatures; i++){
        std::cout << "Idx: " << i << "    at point:" << "(" << detected_features[i].x << "," << detected_features[i].y << ")" << "    Harris score: " << detected_features[i].score << std::endl;
        cv::putText(imgout,"Id="+std::to_string(i),cv::Point(detected_features[i].x,detected_features[i].y),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0),2);
    }

    // Show image and detected
    cv::namedWindow(corners_window);
    cv::imshow(corners_window, imgout);
    int wait = cv::waitKey(0);
    return 0;
    
}
int detectAndDrawShiAndTomasi(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    
    // TODO: 

    return 0;
    
}
int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout){
    
    // TODO: 

    return 0;
    
}
int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    
    // TODO: 

    return 0;
    
}

