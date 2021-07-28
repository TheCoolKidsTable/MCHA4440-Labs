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

struct markers {
    float id;
    std::vector<cv::Point2f> corners;
};

bool sortByScore(features const& lhs, features const& rhs) {
        return lhs.score > rhs.score;
}

bool sortById(markers const& lhs, markers const& rhs) {
        return lhs.id < rhs.id;
}

std::vector<features> detected_features;
std::vector<markers> detected_markers;

int detectAndDrawHarris(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    // Print some stuff
    std::cout << "Using harris feature detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;
    // Initialize variables
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;
    int max_thresh = 255;
    int num_features_detected = 0;   
    const char* corners_window = "Corners detected";

    // Convert to gray
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cornerHarris(img_gray, dst, blockSize, apertureSize, k);

    //Normalize
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    for(int i = 0; i < dst_norm.rows ; i++)
    {
        for(int j = 0; j < dst_norm.cols; j++)
        {
            if((int) dst_norm.at<float>(i,j) > thresh)
            {
                num_features_detected++;
                features new_feature = {i,j,dst.at<float>(i,j)};
                detected_features.push_back(new_feature);
                cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
        }
    }

    std::cout << "Features detected: " << num_features_detected << std::endl;    
    //Sort struct
    std::sort(detected_features.begin(), detected_features.end(), &sortByScore);
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
    // Print some stuff
    std::cout << "Using Shi & Tomasi corner detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;
    // Initialize variables
    int blockSize = 2;
    double k = 0.04;
    float thresh = 0.3;
    int max_thresh = 255;
    int num_features_detected = 0;   
    const char* corners_window = "Corners detected";

    // Convert to gray
    cv::Mat min_eigen_values     = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cornerMinEigenVal(img_gray, min_eigen_values, blockSize, k);

    for(int i = 0; i < min_eigen_values.rows ; i++)
    {
        for(int j = 0; j < min_eigen_values.cols; j++)
        {
            if((float) min_eigen_values.at<float>(i,j) > thresh)
            {
                num_features_detected++;
                features new_feature = {i,j,min_eigen_values.at<float>(i,j)};
                detected_features.push_back(new_feature);
                cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
        }
    }

    std::cout << "Features detected: " << num_features_detected << std::endl;    
    //Sort struct
    std::sort(detected_features.begin(), detected_features.end(), &sortByScore);
    for(int i = 0; i < maxNumFeatures; i++){
        std::cout << "Idx: " << i << "    at point:" << "(" << detected_features[i].x << "," << detected_features[i].y << ")" << "    Min eigen val: " << detected_features[i].score << std::endl;
        cv::putText(imgout,"Id="+std::to_string(i),cv::Point(detected_features[i].x,detected_features[i].y),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0),2);
    }

    // Show image and detected
    cv::namedWindow(corners_window);
    cv::imshow(corners_window, imgout);
    int wait = cv::waitKey(0);
    return 0;
    
}
int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout){
    std::cout << "Using Marker detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    //Add all markers to a struct
    for(int i = 0; i < markerCorners.size();i++){
        markers new_marker = {markerIds[i],markerCorners[i]};
        detected_markers.push_back(new_marker);
    }

    //Sort struct by id
    std::sort(detected_markers.begin(),detected_markers.end(),&sortById);

    //Print out sorted markers
    for(int i = 0; i < markerCorners.size();i++){
        std::cout << "ID: " << detected_markers[i].id << "   with corners: " 
        << detected_markers[i].corners[0] << "," << detected_markers[i].corners[1] << ","
        << detected_markers[i].corners[2] << ","<< detected_markers[i].corners[3] << std::endl;
    }

    //Show markers detected
    const char* detected_markers = "Corners detected";
    cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);
    cv::namedWindow(detected_markers);
    cv::imshow(detected_markers, imgout);
    int wait = cv::waitKey(0);
    return 0;
    
}
int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    
    std::cout << "Using ORB feature detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;

    //Create orb detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.2f,                   // scaleFactor
        8,                      // nlevels
        31,                     // edgeThreshold
        0,                      // firstLevel
        2,                      // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,                     // patchSize
        20                      // fastThreshold 
    );

    //Create array to store keypoints
    std::vector<cv::KeyPoint> keypoints;

    //Create descriptors?
    cv::Mat descriptors;

    // Detect the position of the Oriented FAST corner point.
    orb->detect(img, keypoints);

    // Calculate the BRIEF descriptor according to the position of the corner point
    orb->compute(img, keypoints, descriptors);

    //Print some stuff
    std::cout << "Descriptor Width:" << descriptors.cols << std::endl;
    std::cout << "Descriptor Height:" << descriptors.rows << std::endl;
    for(int i = 0;i < descriptors.rows;i++){
        std::cout << "Keypoint " << i << " " << descriptors.row(i) << std::endl; 
    }

    //Draw keypoints on output image
    drawKeypoints(img, keypoints, imgout, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    const char* orb_features = "Corners detected";
    cv::namedWindow(orb_features);
    cv::imshow(orb_features,imgout);
    int wait = cv::waitKey(0);
    return 0;
    
}

