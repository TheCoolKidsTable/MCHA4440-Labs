#include <iostream>
#include <filesystem>
#include <opencv2/core/persistence.hpp>
#include "settings.h"

Settings::Settings()
    : _goodInput(false)
{}

// Write serialization for this class
void Settings::write(cv::FileStorage& fs) const
{
    fs << "{"
        << "BoardSize_Width"            << boardSize.width
        << "BoardSize_Height"           << boardSize.height
        << "Square_Size"                << squareSize
        << "Write_outputFileName"       << outputFileName
        << "Input_Directory"            << input_dir
        << "Input_Extension"            << input_ext
        << "}";
}

// Read serialization for this class
void Settings::read(const cv::FileNode& node)
{
    node["BoardSize_Width" ]            >> boardSize.width;
    node["BoardSize_Height"]            >> boardSize.height;
    node["Square_Size"]                 >> squareSize;
    node["Write_outputFileName"]        >> outputFileName;
    node["Input_Directory"]             >> input_dir;
    node["Input_Extension"]             >> input_ext;

    validate();
}

bool Settings::isInputGood() const {return _goodInput;}

void Settings::validate()
{
    _goodInput = true;
    if (boardSize.width <= 0 || boardSize.height <= 0)
    {
        std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
        _goodInput = false;
    }
    if (squareSize <= 10e-6)
    {
        std::cerr << "Invalid square size " << squareSize << std::endl;
        _goodInput = false;
    }
    if (!std::filesystem::is_directory(input_dir)){
        std::cerr << "Expected input path: " << input_dir << " to be a directory" << std::endl;
        _goodInput = false;
    }
}

void read(const cv::FileNode& node, Settings& x, const Settings& default_value)
{
    // std::cout << "Calling the thing" <<std::endl;
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}