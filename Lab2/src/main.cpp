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
    bool bSuccess = false;
    cv::VideoCapture cap;

    int N = parser.get<int>("N");
    //Read full path for input image/video
    cv::String img_path = parser.get<cv::String>(0);
    std::filesystem::path pathObj(img_path);
    std::string file_name = pathObj.stem().string();
    std::string file_extension = pathObj.extension().string();
    cv::String detector_type = parser.get<cv::String>("d");
    cv::String imgout_path = parser.get<cv::String>("e");
    //Output file name
    std::string output_file_name = "out/"+file_name+"_"+detector_type+file_extension;
    //Create output directory
    std::filesystem::create_directories("./out");

    // Assign detector case index based on input
    int detector_idx = -1;
    for(int i = 0; i < 4; i++){
        if (detector_type == EXPECTED_CASES[i]){
            detector_idx = i;
        }
    }
    if(detector_idx == 5){
        std::cout << "Invalid detector type" << std::endl;
        return 0;
    }

    if(hasInput && hasDetector) {
        cv::Mat frame;
        cv::VideoWriter video;
            if(file_extension == ".MOV") {
                cv::VideoCapture cap(img_path);
                cv::VideoWriter video(output_file_name,cv::VideoWriter::fourcc('m','p','4','v'),10, cv::Size(1920,1080));
                while(1) {
                    bSuccess = cap.read(frame);
                    if (bSuccess == false) {
                        break;
                    }
                    cv::Mat imgout = frame;
                    detectFrame(frame,imgout,detector_idx,N);
                    if(hasExport){
                        video.write(imgout);
                    }
                }
                video.release();
            } else {
                frame = cv::imread(img_path);
                if(frame.empty()){
                    std::cout << "Invalid path, unable to open image" << std::endl;
                    return 0;
                }
                cv::Mat imgout = frame;
                detectFrame(frame,imgout,detector_idx,N);
                if(hasExport){
                    cv::imwrite(output_file_name,imgout);
                }
            }
    } else {
        if(!hasInput) {
            std::cout << "No image path provided" << std::endl;
        }
        if(!hasDetector) {
            std::cout << "No detector type provided" << std::endl;
        }
    }    
    return 0;
}



