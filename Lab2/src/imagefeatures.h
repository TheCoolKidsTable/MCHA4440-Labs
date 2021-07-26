#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>

int detectAndDrawHarris(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);
int detectAndDrawShiAndTomasi(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);
int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout);
int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);


#endif