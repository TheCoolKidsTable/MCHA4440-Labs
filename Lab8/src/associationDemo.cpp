#include <filesystem>
#include <functional>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "associationDemo.h"
#include "cameraModel.hpp"
#include "dataAssociation.h"
#include "gaussian.hpp"
#include "imagefeatures.h"
#include "measurementPointLandmark.hpp"
#include "plot.h"
#include "utility.h"

// Local Prototypes
void setFeatureDescriptors(cv::Mat & descriptors);
void setFeatureMean(Eigen::VectorXd & murPNn);
void setSquareRootCovariance(Eigen::MatrixXd & S);
void findDescriptorsWithinConfidenceInterval(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, const cv::Mat & view, cv::Mat & descriptorsOut);
void inFillChessBoard(const Eigen::VectorXd & eta, const CameraParameters & param, cv::Mat & img);


// ------------------------------------------------------------------------------------------
// 
// runDescriptorMatcher
// 
// ------------------------------------------------------------------------------------------

bool sortByDistance(cv::DMatch const& lhs, cv::DMatch const& rhs) {
        return lhs.distance < rhs.distance;
}

void runDescriptorMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill){
    

    cv::Mat viewRawA, viewRawB;

    std::filesystem::path  inputPath;
    std::filesystem::path  inputDir("data");

    // View A
    inputPath                       = inputDir / "imageA.jpg";
    viewRawA                        = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawA.empty())          
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 

    Eigen::VectorXd etaA;
    assert( getPoseFromCheckerBoardImage(viewRawA, s, param, etaA));
    cv::Mat outViewA;
    if (doCalibrationGridInFill){
        inFillChessBoard(etaA, param, viewRawA);
    }
    outViewA    = viewRawA.clone();


    // View B
    inputPath                       = inputDir / "imageB.jpg";
    viewRawB                        = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawB.empty())
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 

    Eigen::VectorXd etaB;
    assert( getPoseFromCheckerBoardImage(viewRawB, s, param, etaB));
    cv::Mat outViewB;
    if (doCalibrationGridInFill){
        inFillChessBoard(etaB, param, viewRawB);
    }
    outViewB    = viewRawB.clone();

    // Initialise ORB detector
    // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
    int maxNumFeatures = 500;

    // TODO:
    // Initialise ORB
    // Detect keypoints in both frames
    // Find descriptors in both frames
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.2f,                   // scaleFactor
        8,                      // nlevels
        10,                     // edgeThreshold
        0,                      // firstLevel
        2,                          // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,                     // patchSize
        20                      // fastThreshold 
    );

    //Create array to store keypoints
    std::vector<cv::KeyPoint> keypointsA;
    std::vector<cv::KeyPoint> keypointsB;

    //Create descriptors?
    cv::Mat descriptorsA;
    cv::Mat descriptorsB;

    // Detect the position of the Oriented FAST corner point.
    orb->detect(viewRawA, keypointsA);
    orb->detect(viewRawB, keypointsB);

    // Calculate the BRIEF descriptor according to the position of the corner point
    orb->compute(outViewA, keypointsA, descriptorsA);
    orb->compute(outViewB, keypointsB, descriptorsB);

    // Run Brute force matcher
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsA,descriptorsB,matches);
    //Sort by distance 
    std::sort(matches.begin(), matches.end(), &sortByDistance);
    std::cout << "Keypoints A Size: " << keypointsA.size() << std::endl;
    std::cout << "Keypoints B Size: " << keypointsB.size() << std::endl;
    std::cout << "Matches Size: " << matches.size() << std::endl;
    std::cout << "First 10 match distance:" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << i << matches[i].distance << ",";   
    }
    std::cout<< "" << std::endl;

    // ...
    // Display results with drawMatches
    cv::Mat imgout;   
    cv::drawMatches(outViewA,keypointsA,outViewB,keypointsB,matches,imgout);              
    cv::namedWindow("matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("matches", 960, 540); 
    cv::imshow("matches",imgout);
    int wait = cv::waitKey(0);
}



// ------------------------------------------------------------------------------------------
// 
// runCompatibleDescriptorMatcher
// 
// ------------------------------------------------------------------------------------------
void runCompatibleDescriptorMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill){
    

    cv::Mat viewRawA, viewRawB;

    // Get Feature mean
    Eigen::VectorXd murPNn;
    setFeatureMean(murPNn);
    assert((murPNn.rows() % 3) == 0);

    int n  = murPNn.rows()/3;
    int nx  = 6 + 3*n;
    
    // Initialise state distribution 
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(nx, 1);
    Eigen::MatrixXd S;
    setSquareRootCovariance(S);

    std::filesystem::path  inputPath;
    std::filesystem::path  inputDir("data");

    // ------------------------------------------------------------------------
    // Frame A
    // ------------------------------------------------------------------------
    inputPath                       = inputDir / "imageA.jpg";
    viewRawA                        = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawA.empty())          
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 

    Eigen::VectorXd etaA;
    assert( getPoseFromCheckerBoardImage(viewRawA, s, param, etaA));
    if (doCalibrationGridInFill){
        inFillChessBoard(etaA, param, viewRawA);
    }

    
    std::cout << "Camera pose at frame A" << std::endl;
    std::cout << "etaA.transpose() = " <<etaA.transpose() << std::endl;


    // Generate output view with confidence ellipses
    // ------------------------------------------------------------------------
    cv::Mat outViewA;
    outViewA    = viewRawA.clone();
    
    // Initialise key-points and descriptors for A
    mu  << etaA, murPNn;
    
    cv::Mat descriptorsA;
    findDescriptorsWithinConfidenceInterval(mu, S, param, outViewA, descriptorsA);

    // Detect descriptors in frame A
    std::vector<cv::KeyPoint> keypointsA;

    MeasurementPointLandmarkSingle w2pAdaptor;
    for (int j = 0; j < n; ++j)
    {
        Eigen::VectorXd rQOi_centre;
        Eigen::VectorXd murQOi;
        Eigen::MatrixXd SrQOi;
        auto jointPoseFeature      = std::bind(w2pAdaptor,
                                        j,
                                        std::placeholders::_1,
                                        param,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4);
        int pixFlag     = w2pAdaptor(j, mu, param, rQOi_centre);

        double r,g,b;

        double hueAngle  = (300.0*j)/(1.0*(n-1));
        hsv2rgb(hueAngle, 1, 1, r, g, b);
        
        Eigen::Vector3d color;
        color << 255*r,255*g,255*b;
        
        if (!pixFlag){
            affineTransform(mu, S, jointPoseFeature, murQOi, SrQOi);
            plotGaussianConfidenceEllipse(outViewA, murQOi, SrQOi, color);

            cv::KeyPoint kp(murQOi(0), murQOi(1), 1);
            keypointsA.push_back(kp);
        }
    }


    // ------------------------------------------------------------------------
    // Frame B
    // ------------------------------------------------------------------------
    inputPath                   = inputDir / "imageB.jpg";
    viewRawB                       = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawB.empty())          
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 
    Eigen::VectorXd etaB;
    assert( getPoseFromCheckerBoardImage(viewRawB, s, param, etaB));
    if (doCalibrationGridInFill){
        inFillChessBoard(etaB, param, viewRawB);
    }

    std::cout       << "Camera pose at frame B" << std::endl;
    std::cout       << "etaB.transpose() = " <<etaB.transpose() << std::endl;
    
    // Generate output view with confidence ellipses
    // ------------------------------------------------------------------------
    mu              << etaB, murPNn;
    cv::Mat outViewB;
    outViewB        = viewRawB.clone();
    for (int j = 0; j < n; ++j)
    {
        Eigen::VectorXd rQOi_centre;
        Eigen::VectorXd murQOi;
        Eigen::MatrixXd SrQOi;
        auto jointPoseFeature      = std::bind(w2pAdaptor,
                                        j,
                                        std::placeholders::_1,
                                        param,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4);

        int pixFlag     = w2pAdaptor(j, mu, param, rQOi_centre);

        double r,g,b;

        Eigen::Vector3d color;
        double hueAngle  = (300.0*j)/(1.0*(n-1));
        hsv2rgb(hueAngle, 1, 1, r, g, b);
        color << 255*r,255*g,255*b;
        if (!pixFlag){
            affineTransform(mu, S, jointPoseFeature, murQOi, SrQOi);
            plotGaussianConfidenceEllipse(outViewB, murQOi, SrQOi, color);
        }
    }

    // ------------------------------------------------------------------------
    // ORB detector
    // ------------------------------------------------------------------------
    // TODO
    cv::Mat descriptorsB;
    std::vector<cv::KeyPoint> keypointsB;
    int maxNumFeatures = 5000;

    // Initialise ORB
    // Detect keypoints in both frames
    // Find descriptors in both frames
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.2f,                   // scaleFactor
        8,                      // nlevels
        31,                     // edgeThreshold
        0,                      // firstLevel
        2,                          // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,                     // patchSize
        20                      // fastThreshold 
    );


    // Detect the position of the Oriented FAST corner point.
    orb->detect(viewRawB, keypointsB);

    // Calculate the BRIEF descriptor according to the position of the corner point
    orb->compute(viewRawB, keypointsB, descriptorsB);

    // Run Brute force matcher
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsA,descriptorsB,matches);
    //Sort by distance 
    std::cout << "Keypoints A Size: " << keypointsA.size() << std::endl;
    std::cout << "Keypoints B Size: " << keypointsB.size() << std::endl;
    std::cout << "Matches Size: " << matches.size() << std::endl;
    std::cout << "First 10 match distance:" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << i << matches[i].distance << ",";   
    }
    std::cout<< "" << std::endl;

    // ...


    //TODO: Store match associations
    Eigen::MatrixXd y;
    y.resize(2,matches.size());
    for(int i =0; i < matches.size(); i++) {
        std::cout << "matches train idx: " << matches[i].trainIdx << std::endl;
        std::cout << "matches query idx: " << matches[i].queryIdx << std::endl;
        std::cout << "matches img idx: " << matches[i].imgIdx << std::endl;
        y(0,i) = keypointsB[matches[i].trainIdx].pt.x;
        y(1,i) = keypointsB[matches[i].trainIdx].pt.y;
    }

    std::cout << "y: " << y << std::endl;
    // ---------------------------------------------------------------
    // Draw matches without compatibility check
    // ----------------------------------------------------------------
    // TODO

    // Display results with drawMatches
    cv::Mat imgout;   
    cv::drawMatches(outViewA,keypointsA,outViewB,keypointsB,matches,imgout);              
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 960, 540); 
    cv::imshow("Matches",imgout);
    int wait = cv::waitKey(0);

    // Probability mass enclosed by nstd standard deviations
    // TODO
    double c        = 0; 

    std::cout << std::endl;
    std::cout << "Generate chi2inv look up table" << std::endl;


    // ------------------------------------------------------------------------
    // Generate chi2LUT
    // ------------------------------------------------------------------------
    double nstd     = 3;
    std::vector<double> chi2LUT;
    std::cout << "chi2LUT: " << std::endl;
    c = 2*normcdf(3) - 1;
    for(int i=0; i < y.cols(); i++) {
            chi2LUT.push_back(chi2inv(c, (i+1)*2));
            std::cout << chi2LUT[i] << ",";
    }

    // Form landmark bundle
    // ----------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "Form landmark bundle" << std::endl;
    MeasurementPointLandmarkBundle landmarkBundle;
    // Function handle for use in affine transform
    auto h  = std::bind(
                landmarkBundle, 
                std::placeholders::_1,      // x
                param, 
                std::placeholders::_2,      // h
                std::placeholders::_3,      // SR
                std::placeholders::_4);     // C

    Eigen::VectorXd muY;
    Eigen::MatrixXd SYY;    
    // TODO
    // 
    // ----------------------------------------------------------------
    // Check compatibility and generated isCompatible flag vector
    // ----------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "Check compatibility" << std::endl;

    affineTransform(mu,S,h,muY,SYY);
    
    std::vector<char> isCompatible;
    // TODO
    for(int i = 0; i < matches.size(); i++) {
        bool res = individualCompatibility(i,i,2,y,muY,SYY,chi2LUT);
        isCompatible.push_back(res);
        if(res){
            std::cout << "Pixel at [" << y(0,i) << "," << y(1,i) << " ] in Image B, matches with landmark " << i << "." <<std::endl;
        }
    }


    // ----------------------------------------------------------------
    // Draw matches with compatibility check
    // ----------------------------------------------------------------
    cv::drawMatches(outViewA,keypointsA,outViewB,keypointsB,matches,imgout,cv::Scalar::all(-1),cv::Scalar::all(-1),isCompatible);              
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 960, 540); 
    cv::imshow("Matches",imgout);
    wait = cv::waitKey(0);
    // TODO

    cv::waitKey(0);
}


// ------------------------------------------------------------------------------------------
// 
// runGeometricMatcher
// 
// ------------------------------------------------------------------------------------------
void runGeometricMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill){


    cv::Mat viewRawA, viewRawB;

    // Get Feature mean
    Eigen::VectorXd murPNn;
    setFeatureMean(murPNn);
    assert((murPNn.rows() % 3) == 0);

    int n  = murPNn.rows()/3;
    int nx  = 6 + 3*n;
    
    // Initialise state distribution 
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(nx, 1);
    Eigen::MatrixXd S;
    setSquareRootCovariance(S);


    std::filesystem::path  inputPath;
    std::filesystem::path  inputDir("data");

    // ------------------------------------------------------------------------
    // Frame A
    // ------------------------------------------------------------------------
    inputPath                   = inputDir / "imageA.jpg";
    viewRawA                       = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawA.empty())          
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 

    Eigen::VectorXd etaA;
    assert( getPoseFromCheckerBoardImage(viewRawA, s, param, etaA));
    if (doCalibrationGridInFill){
        inFillChessBoard(etaA, param, viewRawA);
    }
    
    cv::Mat outViewA;
    outViewA    = viewRawA.clone();

    std::cout << "Camera pose at frame A" << std::endl;
    std::cout << "etaA.transpose() = " <<etaA.transpose() << std::endl;

    // Initialise key-points and descriptors for A
    mu  << etaA, murPNn;

    // Detect descriptors in frame A
    std::vector<cv::KeyPoint> keypointsA;

    MeasurementPointLandmarkSingle w2pAdaptor;
    for (int i = 0; i < n; ++i)
    {
        Eigen::VectorXd rQOi_centre;
        Eigen::VectorXd murQOi;
        Eigen::MatrixXd SrQOi;
        auto jointPoseFeature      = std::bind(w2pAdaptor,
                                        i,
                                        std::placeholders::_1,
                                        param,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4);

        int pixFlag     = w2pAdaptor(i, mu, param, rQOi_centre);

        double r,g,b;

        Eigen::Vector3d color;
        double hueAngle  = (300.0*i)/(1.0*(n-1));
        hsv2rgb(hueAngle, 1, 1, r, g, b);
        color << 255*r,255*g,255*b;
        if (!pixFlag){

            affineTransform(mu, S, jointPoseFeature, murQOi, SrQOi);
            plotGaussianConfidenceEllipse(outViewA, murQOi, SrQOi, color);

            cv::KeyPoint kp(murQOi(0), murQOi(1), 1);
            keypointsA.push_back(kp);
        }
    }


    // ------------------------------------------------------------------------
    // Frame B
    // ------------------------------------------------------------------------
    inputPath                       = inputDir / "imageB.jpg";
    viewRawB                        = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    if(viewRawB.empty())          
    {
        std::cout << "View from " << inputPath <<" is empty. Exiting " << __FUNCTION__ << std::endl;
        assert(0);
    } 
    Eigen::VectorXd etaB;
    assert( getPoseFromCheckerBoardImage(viewRawB, s, param, etaB));
    if (doCalibrationGridInFill){
        inFillChessBoard(etaB, param, viewRawB);
    }

    std::cout << "Camera pose at frame B" << std::endl;
    std::cout << "etaB.transpose() = " <<etaB.transpose() << std::endl;

    cv::Mat outViewB;
    outViewB        = viewRawB.clone();
    mu  << etaB, murPNn;
    for (int j = 0; j < n; ++j)
    {
        Eigen::VectorXd rQOi_centre;
        Eigen::VectorXd murQOi;
        Eigen::MatrixXd SrQOi;
        auto jointPoseFeature      = std::bind(w2pAdaptor,
                                        j,
                                        std::placeholders::_1,
                                        param,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4);

        int pixFlag     = w2pAdaptor(j, mu, param, rQOi_centre);

        double r,g,b;

        Eigen::Vector3d color;
        double hueAngle  = (300.0*j)/(1.0*(n-1));
        hsv2rgb(hueAngle, 1, 1, r, g, b);

        color << 255*r,255*g,255*b;
        if (!pixFlag){

            affineTransform(mu, S, jointPoseFeature, murQOi, SrQOi);
            plotGaussianConfidenceEllipse(outViewB, murQOi, SrQOi, color);
        }
    }

    // ------------------------------------------------------------------------
    // Harris Corner detector
    // ------------------------------------------------------------------------
    int maxNumFeatures = 10000;

    std::cout << "Detect Harris corners" << std::endl;
    std::vector<TextureFeature>  features;
    // TODO
    detectHarris(viewRawB,maxNumFeatures,features);

    // ------------------------------------------------------------------------
    // Populate keypointsB and measurement set y with elements of the features vector
    // ------------------------------------------------------------------------
    std::cout << "Populate keypointsB and measurement set y with elements of the features vector " << std::endl;
    std::vector<cv::KeyPoint> keypointsB;
    int m   = features.size();
    Eigen::MatrixXd y(2, m);
    // TODO
    for(int i = 0; i < m;i++) {
        cv::KeyPoint temp;
        temp.pt.x = features[i].x;
        temp.pt.y = features[i].y;
        keypointsB.push_back(temp);
        y(0,i) = features[i].x; 
        y(1,i) = features[i].y;
    }

    cv::Mat output_image;
    // ------------------------------------------------------------------------
    // Run surprisal nearest neighbours
    // ------------------------------------------------------------------------

    std::cout << "Run surprisal nearest neighbours." << std::endl;
    // TODO
    std::vector<int> idx;
    snn(mu,S,y,param,idx,false);

    // ------------------------------------------------------------------------
    // Populate matches and isCompatible vectors for drawMatches
    // ------------------------------------------------------------------------
    std::cout << "Populate matches and isCompatible vectors for drawMatches." << std::endl;
    std::vector< cv::DMatch > matches;
    std::vector<char>isCompatible;
    // TODO
    for(int j = 0; j < idx.size(); j++){
        cv::DMatch temp;
        int i = idx[j];
        bool isMatch = i >= 0;
        temp.queryIdx = j;
        temp.trainIdx = i;

        matches.push_back(temp);
        isCompatible.push_back(isMatch);
        if(isMatch){
            std::cout << "Pixel " << i << " in y located at [ " << y(0,i) << "," << y(1,i) << "] in imageB, matches with landmark " << j << "." << std::endl;
        } else {
            std::cout << "No pixel association with landmark " << j << "." << std::endl;
        }
    }

    std::cout << "Keypoints A Size: " << keypointsA.size() << std::endl;
    std::cout << "Keypoints B Size: " << keypointsB.size() << std::endl;
    std::cout << "Matches Size: " << matches.size() << std::endl;
    

    // ------------------------------------------------------------------------
    // Call drawMatches
    // ------------------------------------------------------------------------


    // TODO

    cv::drawMatches(outViewA,keypointsA,outViewB,keypointsB,matches,output_image,cv::Scalar::all(-1),cv::Scalar::all(-1),isCompatible);              
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 960, 540); 
    cv::imshow("Matches",output_image);
    int wait = cv::waitKey(0);

    cv::waitKey(0);

}



// ----------------------------------------------------------------------------
// Helper functions (you do not need to touch)
// ----------------------------------------------------------------------------


void findDescriptorsWithinConfidenceInterval(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, const cv::Mat & view, cv::Mat & descriptorsOut){
    int nx_all  = mu.rows();
    int nx      = 6;
    assert(nx_all>0);
    assert(((nx_all - nx)%3)==0);
    // Number of landmarks in the map
    int n      = (nx_all - nx)/3;
    assert(n>0);

    int maxNumFeatures = 50000;

    cv::Ptr<cv::ORB> orb;
    orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.3f,                   // scaleFactor 
        10,                     // nlevels
        31,                     // edgeThreshold
        0,                      // firstLevel
        2,                      // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,                     // patchSize
        20                      // fastThreshold
        );


    // Detect descriptors in frame B
    std::vector<cv::KeyPoint> keypointsImage;
    cv::Mat descriptorsImage;
    orb->detect(view, keypointsImage);
    orb->compute(view, keypointsImage, descriptorsImage);

    int m      = keypointsImage.size();
    Eigen::MatrixXd y_features(2, m);
    for (int i = 0; i < keypointsImage.size(); ++i)
    {
        y_features.col(i) << keypointsImage[i].pt.x, keypointsImage[i].pt.y;
    }


    assert(descriptorsOut.rows == 0);
    assert(descriptorsOut.cols == 0);
    descriptorsOut = cv::Mat(0, 32, CV_8U);

    cv::Mat descriptorsFake;
    setFeatureDescriptors(descriptorsFake);

    // Surprisal nearest neighbours
    std::vector<int> idx;
    snn(mu, S, y_features, param, idx);

    assert(idx.size() == n);
    for (int j = 0; j < n; ++j)
    {
        cv::Mat descriptorsJ;

        if(idx[j]>=0){
            // Use descriptor from ORB
            descriptorsJ    = descriptorsImage.row(idx[j]);

        }else{
            // Use fake descriptor
            descriptorsJ    = descriptorsFake.row(j);

        }

        cv::Mat temp;
        cv::vconcat(descriptorsOut, descriptorsJ, temp);
        descriptorsOut = temp;
    }

    assert(descriptorsOut.rows == n);

}


void setFeatureMean(Eigen::VectorXd & murPNn){
    murPNn.resize(27, 1);
// murPNn - [27 x 1]: 
murPNn <<               -0.2,
                         0.1,
                           0,
                           0,
                        -0.3,
                           0,
                           0,
                         0.6,
                           0,
                         0.4,
                         0.6,
                           0,
                         0.4,
                        -0.4,
                           0,
                       0.088,
                       0.066,
                           0,
                       0.022,
                       0.132,
                           0,
                       0.297,
                           0,
                           0,
                     0.29029,
                     0.09502,
                    -0.14843;
}



void setSquareRootCovariance(Eigen::MatrixXd & S){

    S.resize(33, 33);
// S - [33 x 33]: 
S <<     0.001427830283,    0.001229097682,    0.003320690394,   -0.002585376784,    0.004103269664,   -0.002795866555,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,
                      0,   0.0003924364969,    0.001926923125,    0.004464152406,    -0.00231243553,   0.0001452205508,                 0,                 0,                -0,                -0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                -0,                -0,
                      0,                 0,   0.0005341369799,   0.0006053459295,   0.0009182405056,    -0.00099283847,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                -0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,    0.001328048581,     0.00308475703,  -0.0007088613796,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,     0.00300904338,   -0.001734500276,                 0,                 0,                 0,                -0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,
                      0,                 0,                 0,                 0,                 0,    0.002043219303,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,             0.006,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,             0.006,                 0,                -0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015;

}


void setFeatureDescriptors(cv::Mat & descriptors){


    // descriptors_c - [9 x 32]: 
    uint8_t descriptors_c[9][32] = {
        { 100,  50,   7,  52,  20, 210, 254, 238, 233, 227, 179,  65, 151,  65, 157, 251,  26, 101, 153, 250,  87, 157,   2,  20,  36,  89, 226, 243,  86, 245, 244, 228},
        {  82, 210, 186, 136, 225, 161, 133,  39,   6, 149, 120, 226,  62, 151, 172, 234, 137, 176, 191,  61, 142, 239,  71,  71,  18,  91, 197, 227,  23, 245, 122,  98},
        { 208, 153, 185, 229,  49, 244,  44, 101, 210,  79, 211,  98, 159, 110,   3,  89,  94, 104, 139, 169, 137, 153, 137, 235,  19,  65,  85, 133,  48,  10,  71, 114},
        { 157, 176,  85,  78, 217, 151,  20, 118,  49,   1, 127,  76, 163,  98, 173, 199, 155,  35, 179,  23, 100, 250, 196, 196,   4, 233, 204,  96, 135,  15, 129,  26},
        { 194, 126, 216, 223, 171,  51,  95, 144, 175,  59,  93, 183,  84,  33, 129, 140, 245, 115, 238,  54, 178, 113,  14, 153, 248, 186,  83, 182, 107,   2, 165,  58},
        { 110,  23, 110, 232,  88, 110,  32, 199,  91,  38,  71,  49,  18,  63,  59, 209, 101,  86, 220, 129, 174, 189, 141, 213,  25,   7,  79, 117, 205,  80,  29, 248},
        {  26,  60, 221,  85,  39,  86, 162, 125, 249,  52, 236,  31, 225,  21,  45, 124, 101,  56,  43,  27, 176,  10, 205, 229, 215, 118, 237, 155, 238,  15, 194,  73},
        {  30, 118, 168, 175, 163, 253, 106, 108,  32, 229,  99, 214, 118,  78,  48, 151, 202, 237,  68, 100, 144, 196, 166, 249, 175,  72, 250, 253,  46,  75, 218, 232},
        {  96,  51, 149, 230, 217,  97, 253,  67,  48, 223, 126, 190, 195, 190, 216, 201, 165, 176,  19, 138,  92, 133, 238, 255, 125,   8, 134, 198, 220,  91, 216, 228}
    };

    // OpenCV cast descriptors - [9 x 32]: 
    descriptors          = cv::Mat(9, 32, CV_8U, &descriptors_c);
    descriptors          = descriptors.clone();

}


void inFillChessBoard(const Eigen::VectorXd & eta, const CameraParameters & param, cv::Mat & img){

    //  +--> n1
    //  |
    //  v    1 ----- 2 ----- 3 ----- 4
    //  n2   |(11)                   |
    //       |                       |
    //       |                       |
    //       10                      5
    //       |                       |
    //       |                       |
    //       |                       |
    //       9 ----- 8 ----- 7 ----- 6
    // 
    // 
    // rPNn - [3 x 11]: 
    Eigen::MatrixXd rPNn(3, 11); 
    rPNn <<             -0.031,             0.063,             0.157,             0.251,             0.251,             0.251,             0.157,             0.063,            -0.031,            -0.031,            -0.031,
                        -0.031,            -0.031,            -0.031,            -0.031,             0.077,             0.185,             0.185,             0.185,             0.185,             0.077,            -0.031,
                             0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0;

    // Sample colour of one of the white cells
    Eigen::VectorXd rPNn_cloc(3);
    rPNn_cloc << 0.01, -0.01, 0;
    Eigen::VectorXd rQOi;
    worldToPixel(rPNn_cloc, eta, param, rQOi);
    cv::Vec3b & color       = img.at<cv::Vec3b>(cv::Point(rQOi(0), rQOi(1)));
    cv::Scalar fillColour   = cv::Scalar(color[0], color[1], color[2]);

    // Project the points on to the image
    std::vector<cv::Point> rQOi_cv;
    for (int i = 0; i < rPNn.cols(); ++i)
    {
        Eigen::VectorXd rQOi, rPNn_i;
        rPNn_i      = rPNn.col(i);
        int flag    = worldToPixel(rPNn_i, eta, param, rQOi);
        rQOi_cv.push_back(cv::Point(rQOi(0), rQOi(1)));

    }

    // https://gist.github.com/MareArts/54011c365ec0d66d59562945df13dbfe
    const cv::Point * pts =  (const cv::Point*) cv::Mat(rQOi_cv).data;
    int npts =  cv::Mat(rQOi_cv).rows;
    cv::fillPoly(img, &pts, &npts, 1, fillColour);


}





