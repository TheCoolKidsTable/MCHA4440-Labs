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



void detectHarris(const cv::Mat & img, const int & maxNumFeatures, std::vector<TextureFeature> & features)
{

    // TODO
    // Copy Harris detection code from Lab 2
    // Save features above a certain texture threshold
    // Sort features by texture
    // Cap number of features to maxNumFeatures


}