#ifndef SLAM_H
#define SLAM_H

#include <filesystem>

void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, int scenario = 2, int interactive = 0, const std::filesystem::path &outputDirectory = "");

#endif