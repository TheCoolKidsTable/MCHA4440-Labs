#include <iostream>
#include <string>  
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imagefeatures.h"


// Texture objects
TextureFeature::TextureFeature(const double & _score, const double & _x, const double & _y)
{
    score   = _score;
    x       = _x;
    y       = _y;
}
TextureFeature::TextureFeature(const TextureFeature & _tf){
    score   = _tf.score;
    x       = _tf.x;
    y       = _tf.y;
}
TextureFeature::TextureFeature(){
    score   = 0;
    x       = 0;
    y       = 0;
}

bool operator<(const TextureFeature & a, const TextureFeature & b) { return (a.score > b.score); } // needed for std::sort

bool sortByScore(TextureFeature const& lhs, TextureFeature const& rhs) {
        return lhs.score > rhs.score;
}



void detectHarris(const cv::Mat & img, const int & maxNumFeatures, std::vector<TextureFeature> & features)
{

    // TODO
    // Copy Harris detection code from Lab 2
    // Save features above a certain texture threshold
    // Sort features by texture
    // Cap number of features to maxNumFeatures
        // Print some stuff
    std::cout << "Using harris feature detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;
    // Initialize variables
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    double thresh = 0.00005;
    int max_thresh = 255;
    int num_features_detected = 0;   
    const char* corners_window = "Corners detected";

    // Convert to gray
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cornerHarris(img_gray, dst, blockSize, apertureSize, k);

    //Normalize
    for(int i = 0; i < dst.rows; i++)
    {
        for(int j = 0; j < dst.cols; j++)
        {
            if(dst.at<float>(i,j) > thresh)
            {
                num_features_detected++;
                TextureFeature new_feature((double) dst.at<float>(i,j),(double) j, (double) i);
                features.push_back(new_feature);
            }
        }
    }

    std::cout << "Size of features : " << features.size() <<std::endl;

    if(features.size() > 0) {
        std::sort(features.begin(), features.end());
        if(features.size() > maxNumFeatures) {
            features.resize(maxNumFeatures);
        }
        std::cout << "Maximum texture score" << features[0].score << std::endl; 
        std::cout << "Minimum texture score" << features[features.size() - 1].score << std::endl; 
    }

}