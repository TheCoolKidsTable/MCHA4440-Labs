#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 


#include <vector>
#include <opencv2/core.hpp>

struct TextureFeature
{
    double score, x, y;
    TextureFeature(const double & _score, const double & _x, const double & _y);
    TextureFeature(const TextureFeature & _tf);
    TextureFeature();
};



void detectHarris(const cv::Mat & img, const int & maxNumFeatures, std::vector<TextureFeature> & features);


#endif