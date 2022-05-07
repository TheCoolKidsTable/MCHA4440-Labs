#include <filesystem>
#include <string>
#include "SLAM.h"

#include <iostream>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "imagefeatures.h"



void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, int scenario, int interactive, const std::filesystem::path &outputDirectory)
{
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // TODO
    std::cout << "Lets slam lads" << std::endl;
    // - Open input video at videoPath
    cv::VideoCapture cap(videoPath.string());

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat view;
    while(cap.isOpened()){
        cap >> view;
        if(view.empty())          // If there are no more images stop the loop
        {
            break;
        }
        cv::Mat imgout;
        detectAndDrawArUco(view, imgout);
    }
}