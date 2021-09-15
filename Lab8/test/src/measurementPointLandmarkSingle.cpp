#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <filesystem>

#include "../../src/measurementPointLandmark.hpp"
#include "../../src/rotation.hpp"



SCENARIO("MeasurementPointLandmarkSingle: rPCc = [0;0;1], eta=[0;0;0;0;0;0]"){

    Eigen::VectorXd rPCc(3), eta(6);
    Eigen::VectorXd rQOi;
    
    // rPCc - [3 x 1]: 
    rPCc <<                  0,
                             0,
                             1;

    // eta - [6 x 1]: 
    eta <<                  0,
                            0,
                            0,
                            0,
                            0,
                            0;

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd rCNn, Thetanc, rPNn;
    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);
    rpy2rot(Thetanc, Rnc);
    rPNn    = Rnc * rPCc + rCNn;

    Eigen::VectorXd x(9);
    x <<    eta, 
            rPNn;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 0;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            
            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));

                REQUIRE(J.size()>0);
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==9);

                // J(:,1)
                CHECK(J(0,0) == Approx(                -844).margin(1e-12));
                CHECK(J(1,0) == Approx(                   0).margin(1e-12));

                // J(:,2)
                CHECK(J(0,1) == Approx(                   0).margin(1e-12));
                CHECK(J(1,1) == Approx(                -842).margin(1e-12));

                // J(:,3)
                CHECK(J(0,2) == Approx(                   0).margin(1e-12));
                CHECK(J(1,2) == Approx(                   0).margin(1e-12));

                // J(:,4)
                CHECK(J(0,3) == Approx(                   0).margin(1e-12));
                CHECK(J(1,3) == Approx(                 842).margin(1e-12));

                // J(:,5)
                CHECK(J(0,4) == Approx(                -844).margin(1e-12));
                CHECK(J(1,4) == Approx(                   0).margin(1e-12));

                // J(:,6)
                CHECK(J(0,5) == Approx(                   0).margin(1e-12));
                CHECK(J(1,5) == Approx(                   0).margin(1e-12));

                // J(:,7)
                CHECK(J(0,6) == Approx(                 844).margin(1e-12));
                CHECK(J(1,6) == Approx(                   0).margin(1e-12));

                // J(:,8)
                CHECK(J(0,7) == Approx(                   0).margin(1e-12));
                CHECK(J(1,7) == Approx(                 842).margin(1e-12));

                // J(:,9)
                CHECK(J(0,8) == Approx(                   0).margin(1e-12));
                CHECK(J(1,8) == Approx(                   0).margin(1e-12));                
            }

        }
    }        
}


SCENARIO("MeasurementPointLandmarkSingle: rPCc = [0;0;1], eta=0.1*(1:6)'"){

    Eigen::VectorXd rPCc(3), eta(6);
    Eigen::VectorXd rQOi;
    
    // rPCc - [3 x 1]: 
    rPCc <<                  0,
                             0,
                             1;

        // eta - [6 x 1]: 
    eta <<                0.1,
                          0.2,
                          0.3,
                          0.4,
                          0.5,
                          0.6;

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd rCNn, Thetanc, rPNn;
    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);
    rpy2rot(Thetanc, Rnc);
    rPNn    = Rnc * rPCc + rCNn;

    Eigen::VectorXd x(9);
    x <<    eta, 
            rPNn;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 0;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(                 964).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(                 725).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));

                REQUIRE(J.size()>0);
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==9);

                // J(:,1)
                CHECK(J(0,0) == Approx(      -611.309320989).margin(1e-12));
                CHECK(J(1,0) == Approx(        308.15717508).margin(1e-12));

                // J(:,2)
                CHECK(J(0,1) == Approx(      -418.219207771).margin(1e-12));
                CHECK(J(1,1) == Approx(      -728.836507903).margin(1e-12));

                // J(:,3)
                CHECK(J(0,2) == Approx(       404.635154582).margin(1e-12));
                CHECK(J(1,2) == Approx(      -287.750760545).margin(1e-12));

                // J(:,4)
                CHECK(J(0,3) == Approx(  -4.21662704753e-17).margin(1e-12));
                CHECK(J(1,3) == Approx(                 842).margin(1e-12));

                // J(:,5)
                CHECK(J(0,4) == Approx(      -777.375478938).margin(1e-12));
                CHECK(J(1,4) == Approx(    4.6798077341e-14).margin(1e-12));

                // J(:,6)
                CHECK(J(0,5) == Approx(      -288.434254038).margin(1e-12));
                CHECK(J(1,5) == Approx(      -403.676303505).margin(1e-12));

                // J(:,7)
                CHECK(J(0,6) == Approx(       611.309320989).margin(1e-12));
                CHECK(J(1,6) == Approx(       -308.15717508).margin(1e-12));

                // J(:,8)
                CHECK(J(0,7) == Approx(       418.219207771).margin(1e-12));
                CHECK(J(1,7) == Approx(       728.836507903).margin(1e-12));

                // J(:,9)
                CHECK(J(0,8) == Approx(      -404.635154582).margin(1e-12));
                CHECK(J(1,8) == Approx(       287.750760545).margin(1e-12));
               
            }

        }
    }        
}


SCENARIO("MeasurementPointLandmarkSingle: rPCc = [0;1;1], eta=[0;0;0;0;0;0]"){

    Eigen::VectorXd rPCc(3), eta(6);
    Eigen::VectorXd rQOi;
    
    // rPCc - [3 x 1]: 
    rPCc <<                  0,
                             1,
                             1;

    // eta - [6 x 1]: 
    eta <<                  0,
                            0,
                            0,
                            0,
                            0,
                            0;

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd rCNn, Thetanc, rPNn;
    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);
    rpy2rot(Thetanc, Rnc);
    rPNn    = Rnc * rPCc + rCNn;

    Eigen::VectorXd x(9);
    x <<    eta, 
            rPNn;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 0;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(          964.158672).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       1403.90249706).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(          964.158672).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       1403.90249706).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(          964.158672).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       1403.90249706).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));

                REQUIRE(J.size()>0);
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==9);

                // J(:,1)
                CHECK(J(0,0) == Approx(      -680.835812064).margin(1e-12));
                CHECK(J(1,0) == Approx(             0.18524).margin(1e-12));

                // J(:,2)
                CHECK(J(0,1) == Approx(           -0.381488).margin(1e-12));
                CHECK(J(1,1) == Approx(        -455.6311836).margin(1e-12));

                // J(:,3)
                CHECK(J(0,2) == Approx(            0.381488).margin(1e-12));
                CHECK(J(1,2) == Approx(         455.6311836).margin(1e-12));

                // J(:,4)
                CHECK(J(0,3) == Approx(            0.762976).margin(1e-12));
                CHECK(J(1,3) == Approx(         911.2623672).margin(1e-12));

                // J(:,5)
                CHECK(J(0,4) == Approx(      -680.835812064).margin(1e-12));
                CHECK(J(1,4) == Approx(             0.18524).margin(1e-12));

                // J(:,6)
                CHECK(J(0,5) == Approx(       680.835812064).margin(1e-12));
                CHECK(J(1,5) == Approx(            -0.18524).margin(1e-12));

                // J(:,7)
                CHECK(J(0,6) == Approx(       680.835812064).margin(1e-12));
                CHECK(J(1,6) == Approx(            -0.18524).margin(1e-12));

                // J(:,8)
                CHECK(J(0,7) == Approx(            0.381488).margin(1e-12));
                CHECK(J(1,7) == Approx(         455.6311836).margin(1e-12));

                // J(:,9)
                CHECK(J(0,8) == Approx(           -0.381488).margin(1e-12));
                CHECK(J(1,8) == Approx(        -455.6311836).margin(1e-12));

               
            }

        }
    }        
}

SCENARIO("MeasurementPointLandmarkSingle: eta=[0;0;0;0;0;0], 3 features, j = 0"){

    Eigen::VectorXd x(15);
    Eigen::VectorXd rQOi;
    
    // x - [15 x 1]: 
    x <<                  0,
                          0,
                          0,
                          0,
                          0,
                          0,
              -0.6490137652,
               -1.109613039,
                10.39676747,
                1.181166042,
                -0.84555124,
                10.53881673,
              -0.7584532973,
              -0.5726648665,
                10.41919451;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 0;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       911.552340853).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       635.534860031).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       911.552340853).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       635.534860031).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       911.552340853).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       635.534860031).margin(1e-12));


                //--------------------------------------------------------------------------------
                // Checks for SR 
                //--------------------------------------------------------------------------------
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));


                //--------------------------------------------------------------------------------
                // Checks for J 
                //--------------------------------------------------------------------------------
                REQUIRE(J.size()>0);
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==15);

                // J(:,1)
                CHECK(J(0,0) == Approx(      -80.6279631687).margin(1e-12));
                CHECK(J(1,0) == Approx(      0.311293532363).margin(1e-12));

                // J(:,2)
                CHECK(J(0,1) == Approx(      0.318987682262).margin(1e-12));
                CHECK(J(1,1) == Approx(      -80.0899367939).margin(1e-12));

                // J(:,3)
                CHECK(J(0,2) == Approx(      -4.99912162057).margin(1e-12));
                CHECK(J(1,2) == Approx(      -8.52830502861).margin(1e-12));

                // J(:,4)
                CHECK(J(0,3) == Approx(       2.23064977162).margin(1e-12));
                CHECK(J(1,3) == Approx(       842.139568328).margin(1e-12));

                // J(:,5)
                CHECK(J(0,4) == Approx(      -841.514683731).margin(1e-12));
                CHECK(J(1,4) == Approx(      -2.29854088511).margin(1e-12));

                // J(:,6)
                CHECK(J(0,5) == Approx(      -89.6728665965).margin(1e-12));
                CHECK(J(1,5) == Approx(       52.3248867949).margin(1e-12));

                // J(:,7)
                CHECK(J(0,6) == Approx(       80.6279631687).margin(1e-12));
                CHECK(J(1,6) == Approx(     -0.311293532363).margin(1e-12));

                // J(:,8)
                CHECK(J(0,7) == Approx(     -0.318987682262).margin(1e-12));
                CHECK(J(1,7) == Approx(       80.0899367939).margin(1e-12));

                // J(:,9)
                CHECK(J(0,8) == Approx(       4.99912162057).margin(1e-12));
                CHECK(J(1,8) == Approx(       8.52830502861).margin(1e-12));

                // J(:,10)
                CHECK(J(0,9) == Approx(                   0).margin(1e-12));
                CHECK(J(1,9) == Approx(                   0).margin(1e-12));

                // J(:,11)
                CHECK(J(0,10) == Approx(                   0).margin(1e-12));
                CHECK(J(1,10) == Approx(                   0).margin(1e-12));

                // J(:,12)
                CHECK(J(0,11) == Approx(                   0).margin(1e-12));
                CHECK(J(1,11) == Approx(                   0).margin(1e-12));

                // J(:,13)
                CHECK(J(0,12) == Approx(                   0).margin(1e-12));
                CHECK(J(1,12) == Approx(                   0).margin(1e-12));

                // J(:,14)
                CHECK(J(0,13) == Approx(                   0).margin(1e-12));
                CHECK(J(1,13) == Approx(                   0).margin(1e-12));

                // J(:,15)
                CHECK(J(0,14) == Approx(                   0).margin(1e-12));
                CHECK(J(1,14) == Approx(                   0).margin(1e-12));

               
            }
        }
    }        
}


SCENARIO("MeasurementPointLandmarkSingle: eta=[0;0;0;0;0;0], 3 features, j = 1"){

    Eigen::VectorXd x(15);
    Eigen::VectorXd rQOi;
    
    // x - [15 x 1]: 
    x <<                  0,
                          0,
                          0,
                          0,
                          0,
                          0,
              -0.6490137652,
               -1.109613039,
                10.39676747,
                1.181166042,
                -0.84555124,
                10.53881673,
              -0.7584532973,
              -0.5726648665,
                10.41919451;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 1;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       1058.06363387).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       657.818525814).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       1058.06363387).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       657.818525814).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                

            //--------------------------------------------------------------------------------
            // Checks for rQOi 
            //--------------------------------------------------------------------------------
            REQUIRE(rQOi.size()>0);
            REQUIRE(rQOi.rows()==2);
            REQUIRE(rQOi.cols()==1);

            // rQOi(:,1)
            CHECK(rQOi(0,0) == Approx(       1058.06363387).margin(1e-12));
            CHECK(rQOi(1,0) == Approx(       657.818525814).margin(1e-12));


            //--------------------------------------------------------------------------------
            // Checks for SR 
            //--------------------------------------------------------------------------------
            REQUIRE(SR.size()>0);
            REQUIRE(SR.rows()==2);
            REQUIRE(SR.cols()==2);

            // SR(:,1)
            CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
            CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

            // SR(:,2)
            CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
            CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));

            //--------------------------------------------------------------------------------
            // Checks for J 
            //--------------------------------------------------------------------------------
            REQUIRE(J.size()>0);
            REQUIRE(J.rows()==2);
            REQUIRE(J.cols()==15);

            // J(:,1)
            CHECK(J(0,0) == Approx(       -79.046571512).margin(1e-12));
            CHECK(J(1,0) == Approx(     -0.414630215207).margin(1e-12));

            // J(:,2)
            CHECK(J(0,1) == Approx(     -0.416556861977).margin(1e-12));
            CHECK(J(1,1) == Approx(      -79.1516117489).margin(1e-12));

            // J(:,3)
            CHECK(J(0,2) == Approx(       8.82593446497).margin(1e-12));
            CHECK(J(1,2) == Approx(      -6.30402805263).margin(1e-12));

            // J(:,4)
            CHECK(J(0,3) == Approx(      -3.07276340342).margin(1e-12));
            CHECK(J(1,3) == Approx(        839.49470916).margin(1e-12));

            // J(:,5)
            CHECK(J(0,4) == Approx(      -843.482224695).margin(1e-12));
            CHECK(J(1,4) == Approx(       3.07639201292).margin(1e-12));

            // J(:,6)
            CHECK(J(0,5) == Approx(      -67.3299493803).margin(1e-12));
            CHECK(J(1,5) == Approx(      -93.8417870573).margin(1e-12));

            // J(:,7)
            CHECK(J(0,6) == Approx(                   0).margin(1e-12));
            CHECK(J(1,6) == Approx(                   0).margin(1e-12));

            // J(:,8)
            CHECK(J(0,7) == Approx(                   0).margin(1e-12));
            CHECK(J(1,7) == Approx(                   0).margin(1e-12));

            // J(:,9)
            CHECK(J(0,8) == Approx(                   0).margin(1e-12));
            CHECK(J(1,8) == Approx(                   0).margin(1e-12));

            // J(:,10)
            CHECK(J(0,9) == Approx(        79.046571512).margin(1e-12));
            CHECK(J(1,9) == Approx(      0.414630215207).margin(1e-12));

            // J(:,11)
            CHECK(J(0,10) == Approx(      0.416556861977).margin(1e-12));
            CHECK(J(1,10) == Approx(       79.1516117489).margin(1e-12));

            // J(:,12)
            CHECK(J(0,11) == Approx(      -8.82593446497).margin(1e-12));
            CHECK(J(1,11) == Approx(       6.30402805263).margin(1e-12));

            // J(:,13)
            CHECK(J(0,12) == Approx(                   0).margin(1e-12));
            CHECK(J(1,12) == Approx(                   0).margin(1e-12));

            // J(:,14)
            CHECK(J(0,13) == Approx(                   0).margin(1e-12));
            CHECK(J(1,13) == Approx(                   0).margin(1e-12));

            // J(:,15)
            CHECK(J(0,14) == Approx(                   0).margin(1e-12));
            CHECK(J(1,14) == Approx(                   0).margin(1e-12));



               
            }
        }
    }        
}



SCENARIO("MeasurementPointLandmarkSingle: eta=0.1*(1:6)', 3 features, j = 2"){

    Eigen::VectorXd x(15);
    Eigen::VectorXd rQOi;
    
    // x - [15 x 1]: 
    x <<                0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                6.111202501,
               -1.831334847,
                8.635727749,
                7.423164237,
              -0.7061080449,
                 7.96335449,
                5.848526736,
                -1.42239742,
                8.889824082;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 2;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("calling with four arguments"){
                REQUIRE(h(j, x, param, rQOi)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       902.713529777).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       678.832205868).margin(1e-12));

            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd SR;
            AND_THEN("calling with five arguments"){
                REQUIRE(h(j, x, param, rQOi, SR)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       902.713529777).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       678.832205868).margin(1e-12));
                
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            Eigen::MatrixXd J;
            AND_THEN("calling with six arguments"){
                REQUIRE(h(j, x, param, rQOi, SR, J)==0);
                
                //--------------------------------------------------------------------------------
                // Checks for rQOi 
                //--------------------------------------------------------------------------------
                REQUIRE(rQOi.size()>0);
                REQUIRE(rQOi.rows()==2);
                REQUIRE(rQOi.cols()==1);

                // rQOi(:,1)
                CHECK(rQOi(0,0) == Approx(       902.713529777).margin(1e-12));
                CHECK(rQOi(1,0) == Approx(       678.832205868).margin(1e-12));


                //--------------------------------------------------------------------------------
                // Checks for SR 
                //--------------------------------------------------------------------------------
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==2);
                REQUIRE(SR.cols()==2);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));


                //--------------------------------------------------------------------------------
                // Checks for J 
                //--------------------------------------------------------------------------------
                REQUIRE(J.size()>0);
                REQUIRE(J.rows()==2);
                REQUIRE(J.cols()==15);

                // J(:,1)
                CHECK(J(0,0) == Approx(      -61.8333095878).margin(1e-12));
                CHECK(J(1,0) == Approx(       27.0104925834).margin(1e-12));

                // J(:,2)
                CHECK(J(0,1) == Approx(       -39.326695786).margin(1e-12));
                CHECK(J(1,1) == Approx(      -69.2487088293).margin(1e-12));

                // J(:,3)
                CHECK(J(0,2) == Approx(        33.952604944).margin(1e-12));
                CHECK(J(1,2) == Approx(      -31.1554070001).margin(1e-12));

                // J(:,4)
                CHECK(J(0,3) == Approx(       1.35296092105).margin(1e-12));
                CHECK(J(1,3) == Approx(       841.003743827).margin(1e-12));

                // J(:,5)
                CHECK(J(0,4) == Approx(      -759.091815989).margin(1e-12));
                CHECK(J(1,4) == Approx(      -25.1030844429).margin(1e-12));

                // J(:,6)
                CHECK(J(0,5) == Approx(      -326.388764128).margin(1e-12));
                CHECK(J(1,5) == Approx(      -354.256300686).margin(1e-12));

                // J(:,7)
                CHECK(J(0,6) == Approx(                   0).margin(1e-12));
                CHECK(J(1,6) == Approx(                   0).margin(1e-12));

                // J(:,8)
                CHECK(J(0,7) == Approx(                   0).margin(1e-12));
                CHECK(J(1,7) == Approx(                   0).margin(1e-12));

                // J(:,9)
                CHECK(J(0,8) == Approx(                   0).margin(1e-12));
                CHECK(J(1,8) == Approx(                   0).margin(1e-12));

                // J(:,10)
                CHECK(J(0,9) == Approx(                   0).margin(1e-12));
                CHECK(J(1,9) == Approx(                   0).margin(1e-12));

                // J(:,11)
                CHECK(J(0,10) == Approx(                   0).margin(1e-12));
                CHECK(J(1,10) == Approx(                   0).margin(1e-12));

                // J(:,12)
                CHECK(J(0,11) == Approx(                   0).margin(1e-12));
                CHECK(J(1,11) == Approx(                   0).margin(1e-12));

                // J(:,13)
                CHECK(J(0,12) == Approx(       61.8333095878).margin(1e-12));
                CHECK(J(1,12) == Approx(      -27.0104925834).margin(1e-12));

                // J(:,14)
                CHECK(J(0,13) == Approx(        39.326695786).margin(1e-12));
                CHECK(J(1,13) == Approx(       69.2487088293).margin(1e-12));

                // J(:,15)
                CHECK(J(0,14) == Approx(       -33.952604944).margin(1e-12));
                CHECK(J(1,14) == Approx(       31.1554070001).margin(1e-12));
               
            }
        }
    }        
}


SCENARIO("MeasurementPointLandmarkSingle: can call using Eigen expressions"){

    Eigen::VectorXd x(15);
    Eigen::VectorXd rQOi(4);
    Eigen::MatrixXd SR(4, 4);
    Eigen::MatrixXd J(4, 15);
    
    // x - [15 x 1]: 
    x <<                  0,
                          0,
                          0,
                          0,
                          0,
                          0,
              -0.6490137652,
               -1.109613039,
                10.39676747,
                1.181166042,
                -0.84555124,
                10.53881673,
              -0.7584532973,
              -0.5726648665,
                10.41919451;

    MeasurementPointLandmarkSingle h;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;

    int j = 0;
    WHEN("Importing calibration data"){
        importCalibrationData(calibrationFilePath, param);
        //--------------------------------------------------------------------------------
        // four arguments
        //--------------------------------------------------------------------------------
        THEN("Calling MeasurementPointLandmarkSingle"){
            

            AND_THEN("with four arguments"){
                h(j, x, param, rQOi.segment(0,2));
            }

            //--------------------------------------------------------------------------------
            // five arguments
            //--------------------------------------------------------------------------------
            AND_THEN("with five arguments"){
                h(j, x, param, rQOi.segment(0,2), SR.block(0,0,2,2));
            }

            //--------------------------------------------------------------------------------
            // six arguments
            //--------------------------------------------------------------------------------
            AND_THEN("with six arguments"){
                h(j, x, param, rQOi.segment(2,2), SR.block(2,2,2,2), J.block(2,0,2,15));
            }
        }
    }
}

