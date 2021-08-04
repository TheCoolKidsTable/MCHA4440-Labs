#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <bitset>


#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include "settings.h"

using namespace cv;
namespace fs =  std::filesystem;

const double    FONT_SIZE          = 2;
const int       FONT_THICKNESS     = 2;
int wait;

cv::Size screenSize(1920,1080);

int main(int argc, char* argv[])
{
    const cv::String keys
        = "{help h usage ? |           | print this message            }"
          "{@settings      | <none>    | input settings file           }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Camera calibration using openCV.");
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
    // Freebies!!
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
    // Declare variables
    // ------------------------------------------------------------

    cv::Size patternsize(s.boardSize.width,s.boardSize.height);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    cv::Mat input_image;
    cv::Mat img_gray;
    bool showImage = true;
    int num_chessboards_detected = 0;
    std::vector<cv::Point2f> corners; //this will be filled by the detected corners
    std::vector< std::filesystem::path > array_of_image_paths;

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objectPoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imagePoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i < s.boardSize.height; i++) {
        for(int j{0}; j < s.boardSize.width; j++){
            objp.push_back(cv::Point3f(j,i,0));
        }
    }
    
    // Camera calibration stuff 
    cv::Mat cameraMatrix,distCoeffs;
    std::vector<cv::Mat> R, T;
    int calibFlag = cv::CALIB_THIN_PRISM_MODEL;
    double rms;

    for (const auto & entry : fs::directory_iterator(s.input_dir)){
        if(entry.path().extension().string() == ".JPG") {
            array_of_image_paths.push_back(entry.path());    
        }
    }
    // ------------------------------------------------------------
    // Task 1: Do the things
    // ------------------------------------------------------------
    for (const auto & image_path : array_of_image_paths) {
        input_image = cv::imread(image_path.string());
        cv::cvtColor(input_image, img_gray, cv::COLOR_BGR2GRAY);
        bool patternfound = findChessboardCorners(img_gray, patternsize, corners);
        if(patternfound) {
            std::cout << "Found chessboard corners in image: " << image_path.stem().string()+image_path.extension().string() << std::endl;
            cornerSubPix(img_gray,corners, Size(11, 11), Size(-1, -1), termcrit);
            drawChessboardCorners(input_image, patternsize, Mat(corners), patternfound);
            num_chessboards_detected++;
            putText(input_image,"Image "+image_path.stem().string()+image_path.extension().string()+" | chessBoardCorners found "+std::to_string(num_chessboards_detected),cv::Point(25,25),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);
            objectPoints.push_back(objp);
            imagePoints.push_back(corners);
        } else {
            std::cout << "Found no chessboard corners in image: " << image_path.stem().string()+image_path.extension().string() << std::endl;
        }
        if(showImage) {
            resize(input_image, input_image,screenSize);
            imshow("Detected Chessboard",input_image);
            wait = cv::waitKey(0);
        }
        if(wait == 27) {
            showImage = false;
        }    
    }
    showImage = true;
    int flag = cv::CALIB_THIN_PRISM_MODEL;
    rms = cv::calibrateCamera(objectPoints, imagePoints, cv::Size(img_gray.rows,img_gray.cols), cameraMatrix, distCoeffs, R, T, flag);

    //  Calculate the reprojection error for each frame
    std::vector<Point2f> imagePoints2;
    int totalPoints = 0;
    double totalErr = 0, err;
    int i = 0;
    for (const auto & image_path : array_of_image_paths) {
        // Calculate error
        input_image = cv::imread(image_path.string());
        cv::cvtColor(input_image, img_gray, cv::COLOR_BGR2GRAY);
        bool patternfound = findChessboardCorners(img_gray, patternsize, corners);
        if(patternfound) {
            projectPoints(Mat(objectPoints[i]), R[i], T[i],
                cameraMatrix, distCoeffs, imagePoints2);
            err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
            int n = (int)objectPoints[i].size();
            totalErr += err*err;
            totalPoints += n;
            drawChessboardCorners(input_image, patternsize, Mat(imagePoints2), patternfound);
            putText(input_image,"Image "+image_path.stem().string()+image_path.extension().string()+" | re-projection error "+std::to_string(err),cv::Point(25,25),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);
            i++;
        }
        if(showImage) {
            resize(input_image, input_image,screenSize);
            imshow("Detected Chessboard",input_image);
            wait = cv::waitKey(0);
        }
        if(wait == 27) {
            showImage = false;
        }    
    }
    showImage = true;
    double average_rms = std::sqrt(totalErr/totalPoints);
    std::cout << "No. of images used : " << num_chessboards_detected << std::endl;
    std::cout << "Flag for calibrateCamera : " << "CALIB_THIN_PRISM_MODEL" << std::endl;
    std::cout << "inputImageSize : [" << input_image.cols << "x" << input_image.rows << "]" << std::endl;
    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Focal lengths : (fx, fy) = (" << cameraMatrix.at<double>(0,0) << "," << cameraMatrix.at<double>(1,1) << ")" << std::endl;
    std::cout << "Prinicpal points : (cx, cy) = (" << cameraMatrix.at<double>(0,2) << "," << cameraMatrix.at<double>(1,2) << ")" << std::endl;
    std::cout << "aspect ratio: " << cameraMatrix.at<double>(0,0)/cameraMatrix.at<double>(1,1) << std::endl;
    std::cout << "camera skew: " << cameraMatrix.at<double>(0,1) << std::endl;
    std::cout << "reproj avg: " << average_rms << std::endl;
    std::cout << "reproj err calibrateCamera: " << rms << std::endl;

    // Draw box
    i = 0;
    for (const auto & image_path : array_of_image_paths) {
        // Calculate error
        input_image = cv::imread(image_path.string());
        cv::cvtColor(input_image, img_gray, cv::COLOR_BGR2GRAY);
        bool patternfound = findChessboardCorners(img_gray, patternsize, corners);
        if(patternfound) {
            std::vector<Point3f> point1, point2;
            //Bottom
            point1.push_back(cv::Point3f(0,0,0));
            point2.push_back(cv::Point3f(0,s.boardSize.height-1,0));
            point1.push_back(cv::Point3f(0,0,0));
            point2.push_back(cv::Point3f(s.boardSize.width-1,0,0));
            point1.push_back(cv::Point3f(s.boardSize.width-1,0,0));
            point2.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,0));
            point1.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,0));
            point2.push_back(cv::Point3f(0,s.boardSize.height-1,0));
            //Top
            point1.push_back(cv::Point3f(0,0,-10.454545));
            point2.push_back(cv::Point3f(0,s.boardSize.height-1,-10.454545));
            point1.push_back(cv::Point3f(0,0,-10.454545));
            point2.push_back(cv::Point3f(s.boardSize.width-1,0,-10.454545));
            point1.push_back(cv::Point3f(s.boardSize.width-1,0,-10.454545));
            point2.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,-10.454545));
            point1.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,-10.454545));
            point2.push_back(cv::Point3f(0,s.boardSize.height-1,-10.454545));
            //Pillars
            point1.push_back(cv::Point3f(0,0,0));
            point2.push_back(cv::Point3f(0,0,-10.454545));
            point1.push_back(cv::Point3f(0,s.boardSize.height-1,0));
            point2.push_back(cv::Point3f(0,s.boardSize.height-1,-10.454545));
            point1.push_back(cv::Point3f(s.boardSize.width-1,0,0));
            point2.push_back(cv::Point3f(s.boardSize.width-1,0,-10.454545));
            point1.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,0));
            point2.push_back(cv::Point3f(s.boardSize.width-1,s.boardSize.height-1,-10.454545));         

            std::vector<Point2f> cubePoints1;
            std::vector<Point2f> cubePoints2;

            projectPoints(point1, R[i], T[i],cameraMatrix, distCoeffs, cubePoints1);
            projectPoints(point2, R[i], T[i],cameraMatrix, distCoeffs, cubePoints2);

            for (int p = 0; p < cubePoints1.size(); p++) {
                cv::Mat Rcn;
                Rodrigues(R[i],Rcn);
                cv::Mat rPNn1 = (cv::Mat_<double>(3,1) << cubePoints1[p].x, cubePoints1[p].y, 0.0);
                cv::Mat rPNn2 = (cv::Mat_<double>(3,1) << cubePoints2[p].x, cubePoints2[p].y, 0.0);
                cv::Mat rPCc1 = Rcn*(rPNn1 + T[i]);
                cv::Mat rPCc2 = Rcn*(rPNn2 + T[i]);
                std::cout << rPCc1 << std::endl;
                std::cout << rPCc2 << std::endl;
                if(rPCc1.at<double>(2) < 0.0 || rPCc2.at<double>(2) < 0.0) {
                    std::cout << "Removed points" << rPCc1.at<double>(2) << rPCc2.at<double>(2) << std::endl;
                    cubePoints1.erase(cubePoints1.begin()+p);
                    cubePoints2.erase(cubePoints2.begin()+p);
                }
            }

            for(int j = 0; j < cubePoints1.size(); j++) {
                if(cubePoints1[j].x > 0 && cubePoints1[j].y > 0  && cubePoints2[j].x > 0 && cubePoints2[j].y > 0 && cubePoints1[j].x < 1920 && cubePoints1[j].y < 1440  && cubePoints2[j].x < 1920 && cubePoints2[j].y < 1440) {
                    line(input_image,cubePoints1[j],cubePoints2[j],Scalar(std::rand() % 255,std::rand() % 255,std::rand() % 255),FONT_THICKNESS,0);
                } 
            }
            putText(input_image,"Image "+image_path.stem().string()+image_path.extension().string()+" | Projection of shape",cv::Point(25,25),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);
            i++;
        }
        if(showImage) {
            resize(input_image, input_image,screenSize);
            imshow("Detected Chessboard",input_image);
            wait = cv::waitKey(0);
        }
        if(wait == 27) {
            showImage = false;
        }     
    }

    return 0;
}

