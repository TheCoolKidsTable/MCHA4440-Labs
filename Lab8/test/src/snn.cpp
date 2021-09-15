#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <vector>

#include "../../src/dataAssociation.h"



SCENARIO("SNN: 1 landmark, 2 features, no association"){
    
    // mu - [9 x 1]: 
    Eigen::VectorXd mu(9); 
    mu <<           0.110866,
                  -0.0516179,
                  -0.0888025,
                    0.854756,
                   0.0448205,
                     3.14158,
                           0,
                           0,
                           0;

    // S - [9 x 9]: 
    Eigen::MatrixXd S(9, 9); 
    S <<              0.002,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,
                          0,             0.004,                 0,                 0,                 0,                 0,                 0,                 0,                -0,
                          0,                 0,             0.006,                 0,                 0,                 0,                -0,                 0,                 0,
                          0,                 0,                 0,            0.0005,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,             0.001,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352;

    // y - [2 x 2]: 
    Eigen::MatrixXd y(2, 2); 
    y <<                519,              1721,
                        617,              1390;
    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        REQUIRE(param.Kc.rows == 3);
        REQUIRE(param.Kc.cols == 3);
        REQUIRE(param.distCoeffs.cols == 1);

        THEN("Calling snn"){
            std::vector<int> idx;
            double s = snn(mu, S, y, param, idx);

            THEN("Surprisal is correct"){
                REQUIRE(s == Approx(14.832478857591873));
            }

            THEN("idx has the correct size"){
                REQUIRE(idx.size() == 1);

                AND_THEN("Landmark #1 unassociated"){
                    REQUIRE(idx[0] == -1);
                }
            }
        }
    }        
}

SCENARIO("SNN: 1 landmark, 5 features, 1 feature associated"){
    
    // mu - [9 x 1]: 
    Eigen::VectorXd mu(9); 
    mu <<           0.110866,
                  -0.0516179,
                  -0.0888025,
                    0.854756,
                   0.0448205,
                     3.14158,
                           0,
                           0,
                           0;

    // S - [9 x 9]: 
    Eigen::MatrixXd S(9, 9); 
    S <<              0.002,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,
                          0,             0.004,                 0,                 0,                 0,                 0,                 0,                 0,                -0,
                          0,                 0,             0.006,                 0,                 0,                 0,                -0,                 0,                 0,
                          0,                 0,                 0,            0.0005,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,             0.001,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352;

    // y - [2 x 5]: 
    Eigen::MatrixXd y(2, 5); 
    y <<                519,              1721,       1578.363785,       1921.012306,       1679.435374,
                        617,              1390,       959.8737768,       849.6044461,       911.0285137;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        REQUIRE(param.Kc.rows == 3);
        REQUIRE(param.Kc.cols == 3);
        REQUIRE(param.distCoeffs.cols == 1);

        THEN("Calling snn"){
            std::vector<int> idx;
            double s = snn(mu, S, y, param, idx);

            THEN("Surprisal is correct"){
                REQUIRE(s == Approx(10.354214967374880));
            }

            THEN("idx has the correct size"){
                REQUIRE(idx.size() == 1);

                AND_THEN("Landmark #1 associated"){
                    REQUIRE(idx[0] == 4);
                }
            }
        }
    }        
}



SCENARIO("SNN: 2 landmarks, 8 features, 2 features associated"){
    
    // mu - [12 x 1]: 
    Eigen::VectorXd mu(12); 
    mu <<           0.110866,
                  -0.0516179,
                  -0.0888025,
                    0.854756,
                   0.0448205,
                     3.14158,
                           0,
                           0,
                           0,
                       0.198,
                       0.044,
                           0;

    // S - [12 x 12]: 
    Eigen::MatrixXd S(12, 12); 
    S <<              0.002,                 0,                 0,                 0,                 0,                 0,                -0,                -0,                -0,                -0,                -0,                 0,
                          0,             0.004,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,
                          0,                 0,             0.006,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,            0.0005,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,             0.001,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352;

    // y - [2 x 8]: 
    Eigen::MatrixXd y(2, 8); 
    y <<               1091,               391,       1925.500995,       1570.723008,       1758.000518,       348.5849085,       587.1722137,       484.7122863,
                        364,              1072,       889.8387959,        823.664027,       946.3816739,       544.7453755,        895.294163,       710.2556653;


    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        REQUIRE(param.Kc.rows == 3);
        REQUIRE(param.Kc.cols == 3);
        REQUIRE(param.distCoeffs.cols == 1);

        THEN("Calling snn"){
            std::vector<int> idx;
            double s = snn(mu, S, y, param, idx);

            THEN("Surprisal is correct"){
                REQUIRE(s == Approx(22.025573025596437));
            }

            THEN("idx has the correct size"){
                REQUIRE(idx.size() == 2);

                AND_THEN("Landmark #1 associated"){
                    REQUIRE(idx[0] == 4);
                }

                AND_THEN("Landmark #2 associated"){
                    REQUIRE(idx[1] == 7);
                }
            }
        }
    }        
}

SCENARIO("SNN: 2 landmarks, 6 features, 1 feature associated"){
    
    // mu - [12 x 1]: 
    Eigen::VectorXd mu(12); 
    mu <<           0.110866,
                  -0.0516179,
                  -0.0888025,
                    0.854756,
                   0.0448205,
                     3.14158,
                           0,
                           0,
                           0,
                       0.198,
                       0.044,
                           0;

    // S - [12 x 12]: 
    Eigen::MatrixXd S(12, 12); 
    S <<              0.002,                 0,                 0,                 0,                 0,                 0,                -0,                -0,                -0,                -0,                -0,                 0,
                          0,             0.004,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,
                          0,                 0,             0.006,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,            0.0005,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,             0.001,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352,                 0,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.01056,                 0,
                          0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,           0.00352;

    // y - [2 x 6]: 
    Eigen::MatrixXd y(2, 6); 
    y <<               1091,               391,       1925.500995,       1570.723008,       500.7720585,       348.5849085,
                        364,              1072,       889.8387959,        823.664027,       693.8874361,       544.7453755;


    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        REQUIRE(param.Kc.rows == 3);
        REQUIRE(param.Kc.cols == 3);
        REQUIRE(param.distCoeffs.cols == 1);

        THEN("Calling snn"){
            std::vector<int> idx;
            double s = snn(mu, S, y, param, idx);

            THEN("Surprisal is correct"){
                REQUIRE(s == Approx(25.872930498597370));
            }

            THEN("idx has the correct size"){
                REQUIRE(idx.size() == 2);

                AND_THEN("Landmark #1 unassociated"){
                    REQUIRE(idx[0] == -1);
                }

                AND_THEN("Landmark #2 associated"){
                    REQUIRE(idx[1] == 4);
                }
            }
        }
    }        
}

