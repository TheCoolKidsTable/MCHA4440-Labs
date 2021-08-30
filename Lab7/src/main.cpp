#include <bitset>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>


#include "confidenceRegionDemo.h"
#include "cameraModel.hpp"
#include "settings.h"
#include "utility.h"


int main(int argc, char* argv [])
{

    const cv::String keys
        = "{help h usage ? |           | print this message            }"
          "{@settings      | <none>    | input settings file           }"
          "{calibrate c    |           | calibration flag. When set it runs calibration on the image set in the input settings file          }";
          
    cv::CommandLineParser parser(argc, argv, keys);
    
    parser.about("Camera calibration using openCV. Press ESC to skip rendering each frame.");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // ------------------------------------------------------------
    // Read settings
    // ------------------------------------------------------------
    bool hasSettings = parser.has("@settings");
    if (!hasSettings){
        std::cout << "No settings file specified!" << std::endl << std::endl;
        parser.printMessage();
        return -1;
    }
    Settings s;
    const std::string inputSettingsFile = parser.get<std::string>("@settings");

    if (!std::filesystem::exists(inputSettingsFile)){
        std::cout << "No file on path: " << inputSettingsFile << std::endl << std::endl;
        parser.printMessage();
        return -1;
    }

    cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl << std::endl;
        parser.printMessage();
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();

    if (!s.isInputGood())
    {
        std::cout << "Invalid input detected. Application stopping. " << std::endl;
        return -1;
    }

    // ------------------------------------------------------------
    // Load camera parameters
    // ------------------------------------------------------------
    bool doCalibrate    = parser.has("calibrate");


    CameraParameters param;
    std::filesystem::path calibrationFilePath = "data/gopro/camera.xml";
    if (doCalibrate){
        std::cout << "Calibrating camera. " << std::endl;
        calibrateCameraFromImageSet(s, param);
        exportCalibrationData(calibrationFilePath, param);
    }else{
        std::cout << "Loading calibration data. " << std::endl;
        importCalibrationData(calibrationFilePath, param);
        param.print();
    }

    // ------------------------------------------------------------
    // Show calibration data with the ellipsoids
    // ------------------------------------------------------------
    calibrationConfidenceRegionDemo(s, param);



    return EXIT_SUCCESS;
}
