#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <filesystem>

#include "../../src/cameraModel.hpp"
#include "../../src/rotation.hpp"



SCENARIO("WorldToPixelAdaptor: rPCc = [0;0;1]"){

    Eigen::VectorXd rPCc(3), eta(6);
    Eigen::VectorXd rQOi;
    
    // rPCc - [3 x 1]: 
    rPCc <<                  0,
                             0,
                             1;

    // eta - [6 x 1]: 
    eta <<                  1,
                            2,
                            3,
                            4,
                            5,
                            6;

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd rCNn, Thetanc, rPNn;
    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);
    rpy2rot(Thetanc, Rnc);
    rPNn    = Rnc * rPCc + rCNn;

    WorldToPixelAdaptor w2p;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;
    importCalibrationData(calibrationFilePath, param);
    // param.print();

    //--------------------------------------------------------------------------------
    // four arguments
    //--------------------------------------------------------------------------------
    WHEN("Calling WorldToPixelAdaptor with four arguments"){
        w2p(rPNn, eta, param, rQOi);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(                 964));
                    CHECK(rQOi(1,0) == Approx(                 725));
                }
            }
        }
    }

    //--------------------------------------------------------------------------------
    // five arguments
    //--------------------------------------------------------------------------------
    Eigen::MatrixXd SR;
    WHEN("Calling WorldToPixelAdaptor with five arguments"){
        w2p(rPNn, eta, param, rQOi, SR);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(                 964));
                    CHECK(rQOi(1,0) == Approx(                 725));
                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for SR 
        //--------------------------------------------------------------------------------
        THEN("SR is not empty"){
            REQUIRE(SR.size()>0);
            
            AND_THEN("SR has the right dimensions"){
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);
                AND_THEN("SR is correct"){

                    // SR(:,1)
                    CHECK(SR(0,0) == Approx(                   0));
                    CHECK(SR(1,0) == Approx(                   0));

                    // SR(:,2)
                    CHECK(SR(0,1) == Approx(                   0));
                    CHECK(SR(1,1) == Approx(                   0));

                }
            }
        }
    }

    //--------------------------------------------------------------------------------
    // six arguments
    //--------------------------------------------------------------------------------
    Eigen::MatrixXd J;
    WHEN("Calling WorldToPixelAdaptor with six arguments"){
        w2p(rPNn, eta, param, rQOi, SR, J);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(                 964));
                    CHECK(rQOi(1,0) == Approx(                 725));
                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for SR 
        //--------------------------------------------------------------------------------
        THEN("SR is not empty"){
            REQUIRE(SR.size()>0);
            
            AND_THEN("SR has the right dimensions"){
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);
                AND_THEN("SR is correct"){

                    // SR(:,1)
                    CHECK(SR(0,0) == Approx(                   0));
                    CHECK(SR(1,0) == Approx(                   0));

                    // SR(:,2)
                    CHECK(SR(0,1) == Approx(                   0));
                    CHECK(SR(1,1) == Approx(                   0));

                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for J 
        //--------------------------------------------------------------------------------
        THEN("J is not empty"){
            REQUIRE(J.size()>0);
            
            AND_THEN("J has the right dimensions"){
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==3);
                AND_THEN("J is correct"){

                    // J(:,1)
                    CHECK(J(0,0) == Approx(       229.875217627));
                    CHECK(J(1,0) == Approx(       432.933711761));

                    // J(:,2)
                    CHECK(J(0,1) == Approx(      -66.8951115755));
                    CHECK(J(1,1) == Approx(       -699.18464137));

                    // J(:,3)
                    CHECK(J(0,2) == Approx(       809.332087816));
                    CHECK(J(1,2) == Approx(      -180.757402317));

                }
            }
        }
    }
    
}



SCENARIO("WorldToPixelAdaptor: rPCc = [0.5;0;1]"){

    Eigen::VectorXd rPCc(3), eta(6);
    Eigen::VectorXd rQOi;
    
    // rPCc - [3 x 1]: 
    rPCc <<                0.5,
                             0,
                             1;

    // eta - [6 x 1]: 
    eta <<                  1,
                            2,
                            3,
                            4,
                            5,
                            6;

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd rCNn, Thetanc, rPNn;
    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);
    rpy2rot(Thetanc, Rnc);
    rPNn    = Rnc * rPCc + rCNn;

    WorldToPixelAdaptor w2p;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;
    importCalibrationData(calibrationFilePath, param);
    // param.print();

    //--------------------------------------------------------------------------------
    // four arguments
    //--------------------------------------------------------------------------------
    WHEN("Calling WorldToPixelAdaptor with four arguments"){
        w2p(rPNn, eta, param, rQOi);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(       1358.65236277));
                    CHECK(rQOi(1,0) == Approx(        724.91843125));
                }
            }
        }
    }

    //--------------------------------------------------------------------------------
    // five arguments
    //--------------------------------------------------------------------------------
    Eigen::MatrixXd SR;
    WHEN("Calling WorldToPixelAdaptor with five arguments"){
        w2p(rPNn, eta, param, rQOi, SR);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(       1358.65236277));
                    CHECK(rQOi(1,0) == Approx(        724.91843125));
                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for SR 
        //--------------------------------------------------------------------------------
        THEN("SR is not empty"){
            REQUIRE(SR.size()>0);
            
            AND_THEN("SR has the right dimensions"){
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);
                AND_THEN("SR is correct"){

                    // SR(:,1)
                    CHECK(SR(0,0) == Approx(                   0));
                    CHECK(SR(1,0) == Approx(                   0));

                    // SR(:,2)
                    CHECK(SR(0,1) == Approx(                   0));
                    CHECK(SR(1,1) == Approx(                   0));

                }
            }
        }
    }

    //--------------------------------------------------------------------------------
    // six arguments
    //--------------------------------------------------------------------------------
    Eigen::MatrixXd J;
    WHEN("Calling WorldToPixelAdaptor with six arguments"){
        w2p(rPNn, eta, param, rQOi, SR, J);
        
        //--------------------------------------------------------------------------------
        // Checks for rQOi 
        //--------------------------------------------------------------------------------
        THEN("rQOi is not empty"){
            REQUIRE(rQOi.size()>0);
            
            AND_THEN("rQOi has the right dimensions"){
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);
                AND_THEN("rQOi is correct"){

                    // rQOi(:,1)
                    CHECK(rQOi(0,0) == Approx(       1358.65236277));
                    CHECK(rQOi(1,0) == Approx(        724.91843125));
                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for SR 
        //--------------------------------------------------------------------------------
        THEN("SR is not empty"){
            REQUIRE(SR.size()>0);
            
            AND_THEN("SR has the right dimensions"){
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);
                AND_THEN("SR is correct"){

                    // SR(:,1)
                    CHECK(SR(0,0) == Approx(                   0));
                    CHECK(SR(1,0) == Approx(                   0));

                    // SR(:,2)
                    CHECK(SR(0,1) == Approx(                   0));
                    CHECK(SR(1,1) == Approx(                   0));

                }
            }
        }

        //--------------------------------------------------------------------------------
        // Checks for J 
        //--------------------------------------------------------------------------------
        THEN("J is not empty"){
            REQUIRE(J.size()>0);
            
            AND_THEN("J has the right dimensions"){
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==3);
                AND_THEN("J is correct"){

                    // J(:,1)
                    CHECK(J(0,0) == Approx(      -93.2408693987));
                    CHECK(J(1,0) == Approx(       404.886520958));

                    // J(:,2)
                    CHECK(J(0,1) == Approx(      -246.237415702));
                    CHECK(J(1,1) == Approx(      -653.703254062));

                    // J(:,3)
                    CHECK(J(0,2) == Approx(       729.734434231));
                    CHECK(J(1,2) == Approx(      -169.369941932));

                }
            }
        }
    }
    
}
