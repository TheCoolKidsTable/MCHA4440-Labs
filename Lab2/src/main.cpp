#include <iostream>
#include <string>  
#include <cstdlib>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imagefeatures.h"


const int DETECTOR_HARRIS   = 0;
const int DETECTOR_SHI      = 1;
const int DETECTOR_ARUCO    = 2;
const int DETECTOR_ORB      = 3;


const std::vector<std::string> EXPECTED_CASES {
            "harris",
            "shi", 
            "aruco", 
            "orb", 
        };


int detectFrame(cv::Mat img, cv::Mat & imgout, int detector_idx, int maxNumFeatures){


    switch (detector_idx){
        case DETECTOR_HARRIS:
            detectAndDrawHarris(img, imgout, maxNumFeatures);
        break;
        case DETECTOR_SHI:
            detectAndDrawShiAndTomasi(img, imgout, maxNumFeatures);
        break;
        case DETECTOR_ORB:
            detectAndDrawORB(img, imgout, maxNumFeatures);
        break;
        case DETECTOR_ARUCO:
            detectAndDrawArUco(img, imgout);
        break;
    }
     
    return 0;
}


int main(int argc, char *argv[])
{

    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., path/to/image.png)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | number of features to find}"
        "{detector d      | orb      | feature detector to use (e.g., harris, shi, aruco, orb)}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Lab 2 for MCHA4400");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    bool hasInput               = parser.has("@input");
    bool hasExport              = parser.has("export");
    bool hasDetector            = parser.has("detector");
    int N = parser.get<int>("N");
    //Read path for input image 
    cv::String img_path = parser.get<cv::String>(0);
    //Convert path to matrix
    cv::Mat img = cv::imread(img_path);
    //Initialize output
    cv::Mat imgout = img;
    cv::String imgout_path = parser.get<cv::String>("e");
    int detector_idx = 0; 

    // TODO: Everything
    // if(hasInput && hasExport && hasDetector) {
        detectFrame(img,imgout,detector_idx,N);
    // }    
    return 0;
}



