#include <bitset>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>


#include "associationDemo.h"
#include "cameraModel.hpp"
#include "settings.h"
#include "utility.h"


int main(int argc, char* argv [])
{

    const cv::String keys
        = "{help h usage ? |           | print this message            }"
          "{matcher m      | 0         | matching flag. 0 => Descriptor matcher, 1=> Descriptor matcher with compatibility, 2 => geometric data association}"
          "{infill         |           | sets calibration grid area to with a coloured infill}";
          
    cv::CommandLineParser parser(argc, argv, keys);
    
    // TODO: 
    parser.about("Demonstration of brute force, nearest neighbours, and geometric based matchers.");
    bool hasMatcher     = parser.has("matcher");
    bool hasInFill      = parser.has("infill");

    int matchType                       = parser.get<int>("matcher");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }


    // ------------------------------------------------------------
    // Read settings
    // ------------------------------------------------------------
  
    Settings s;
    const std::string inputSettingsFile = "data/calibrationConfig.xml";
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

    CameraParameters param;
    std::filesystem::path calibrationFilePath = "data/camera.xml";
    importCalibrationData(calibrationFilePath, param);
    assert(param.Kc.rows == 3);
    assert(param.Kc.cols == 3);
    assert(param.distCoeffs.cols == 1);

    // ------------------------------------------------------------
    // Run matcher
    // ------------------------------------------------------------
    
    enum {
        DESCRIPTOR_ONLY,
        DESCRIPTOR_COMPATIBILITY,
        DESCRIPTOR_GEOMETRIC_ASSOCIATION
    };
     
    switch (matchType){
        case DESCRIPTOR_ONLY:
            std::cout << "Running brute force descriptor matcher " << std::endl;
            runDescriptorMatcher(s, param, hasInFill);
            break;
        case DESCRIPTOR_COMPATIBILITY:
            std::cout << "Running brute force descriptor matcher with compatibility filtering" << std::endl;
            runCompatibleDescriptorMatcher(s, param, hasInFill);
            break;
        case DESCRIPTOR_GEOMETRIC_ASSOCIATION:
            std::cout << "Running geometric data association " << std::endl;
            runGeometricMatcher(s, param, hasInFill);
            break;
        default:
            std::cout << "Unknown match type: " << matchType << std::endl;
            assert(0);
    }


    return EXIT_SUCCESS;
}
