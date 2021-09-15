#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <iostream>

#include "../../src/measurementPointLandmark.hpp"
#include "../../src/rotation.hpp"



TEST_CASE("MeasurementPointLandmarkBundle: 1 landmark"){

    
    Eigen::VectorXd h;
    
    // x - [9 x 1]: 
    Eigen::VectorXd x(9); 
    x <<                0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                 5.04103981,
               0.1801522503,
                9.097976544;
    MeasurementPointLandmarkBundle bundle;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;
    importCalibrationData(calibrationFilePath, param);
    REQUIRE(param.Kc.rows == 3);
    REQUIRE(param.Kc.cols == 3);
    REQUIRE(param.distCoeffs.cols == 1);


    SECTION("Calling MeasurementPointLandmarkBundle"){
        
        SECTION("with three arguments"){
            bundle(x, param, h);

            //--------------------------------------------------------------------------------
            // four arguments
            //--------------------------------------------------------------------------------
            SECTION("with four arguments"){
                Eigen::MatrixXd SR;
                bundle(x, param, h, SR);

                //--------------------------------------------------------------------------------
                // five arguments
                //--------------------------------------------------------------------------------
                SECTION("with five arguments"){
                    Eigen::MatrixXd C;
                    bundle(x, param, h, SR, C);

                    //--------------------------------------------------------------------------------
                    // Checks for C 
                    //--------------------------------------------------------------------------------
                    REQUIRE(C.size()>0);
                    REQUIRE(C.rows()==2);
                    REQUIRE(C.cols()==9);

                    // C(:,1)
                    CHECK(C(0,0) == Approx(      -63.6574996454).margin(1e-12));
                    CHECK(C(1,0) == Approx(       35.8352469271).margin(1e-12));

                    // C(:,2)
                    CHECK(C(0,1) == Approx(      -41.4356540013).margin(1e-12));
                    CHECK(C(1,1) == Approx(      -72.7777272998).margin(1e-12));

                    // C(:,3)
                    CHECK(C(0,2) == Approx(       35.6572711795).margin(1e-12));
                    CHECK(C(1,2) == Approx(      -20.2896489769).margin(1e-12));

                    // C(:,4)
                    CHECK(C(0,3) == Approx(      -2.53053842337).margin(1e-12));
                    CHECK(C(1,3) == Approx(       842.021403683).margin(1e-12));

                    // C(:,5)
                    CHECK(C(0,4) == Approx(      -813.086828353).margin(1e-12));
                    CHECK(C(1,4) == Approx(      -18.8147186391).margin(1e-12));

                    // C(:,6)
                    CHECK(C(0,5) == Approx(      -205.998674084).margin(1e-12));
                    CHECK(C(1,5) == Approx(      -358.886398837).margin(1e-12));

                    // C(:,7)
                    CHECK(C(0,6) == Approx(       63.6574996454).margin(1e-12));
                    CHECK(C(1,6) == Approx(      -35.8352469271).margin(1e-12));

                    // C(:,8)
                    CHECK(C(0,7) == Approx(       41.4356540013).margin(1e-12));
                    CHECK(C(1,7) == Approx(       72.7777272998).margin(1e-12));

                    // C(:,9)
                    CHECK(C(0,8) == Approx(      -35.6572711795).margin(1e-12));
                    CHECK(C(1,8) == Approx(       20.2896489769).margin(1e-12));
                }

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
            }

        }

        //--------------------------------------------------------------------------------
        // Checks for h 
        //--------------------------------------------------------------------------------
        REQUIRE(h.size()>0);
        REQUIRE(h.rows()==2);
        REQUIRE(h.cols()==1);

        CHECK(h(0,0) == Approx(        909.52217386).margin(1e-12));
        CHECK(h(1,0) == Approx(       823.909657549).margin(1e-12));

    }        
}


TEST_CASE("MeasurementPointLandmarkBundle: 2 landmarks"){

    
    Eigen::VectorXd h;
    
    // x - [12 x 1]: 
    Eigen::VectorXd x(12); 
    x <<                0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                5.859677673,
               -1.512200032,
                 8.58558094,
                7.406880156,
              -0.9207531776,
                7.716902034;

    MeasurementPointLandmarkBundle bundle;

    const std::string calibrationFilePath = "test/data/camera.xml";

    REQUIRE(std::filesystem::exists(calibrationFilePath));

    CameraParameters param;
    importCalibrationData(calibrationFilePath, param);
    REQUIRE(param.Kc.rows == 3);
    REQUIRE(param.Kc.cols == 3);
    REQUIRE(param.distCoeffs.cols == 1);


    SECTION("Calling MeasurementPointLandmarkBundle"){
        
        SECTION("with three arguments"){
            bundle(x, param, h);

            //--------------------------------------------------------------------------------
            // four arguments
            //--------------------------------------------------------------------------------
            SECTION("with four arguments"){
                Eigen::MatrixXd SR;
                bundle(x, param, h, SR);

                //--------------------------------------------------------------------------------
                // five arguments
                //--------------------------------------------------------------------------------
                SECTION("with five arguments"){
                    Eigen::MatrixXd C;
                    bundle(x, param, h, SR, C);

                    //--------------------------------------------------------------------------------
                    // Checks for C 
                    //--------------------------------------------------------------------------------
                    REQUIRE(C.size()>0);
                    REQUIRE(C.rows()==4);
                    REQUIRE(C.cols()==12);

                    // C(:,1)
                    CHECK(C(0,0) == Approx(       -62.842747615).margin(1e-12));
                    CHECK(C(1,0) == Approx(       26.6677971142).margin(1e-12));
                    CHECK(C(2,0) == Approx(      -52.6785557072).margin(1e-12));
                    CHECK(C(3,0) == Approx(       23.9679914391).margin(1e-12));

                    // C(:,2)
                    CHECK(C(0,1) == Approx(      -40.2632787465).margin(1e-12));
                    CHECK(C(1,1) == Approx(      -70.5673451073).margin(1e-12));
                    CHECK(C(2,1) == Approx(      -40.9812110489).margin(1e-12));
                    CHECK(C(3,1) == Approx(      -69.1455786331).margin(1e-12));

                    // C(:,3)
                    CHECK(C(0,2) == Approx(       35.3644705579).margin(1e-12));
                    CHECK(C(1,2) == Approx(      -33.1205896333).margin(1e-12));
                    CHECK(C(2,2) == Approx(       45.7045366475).margin(1e-12));
                    CHECK(C(3,2) == Approx(      -34.0609012808).margin(1e-12));

                    // C(:,4)
                    CHECK(C(0,3) == Approx(       1.59572375446).margin(1e-12));
                    CHECK(C(1,3) == Approx(       841.552524073).margin(1e-12));
                    CHECK(C(2,3) == Approx(      -4.16932468769).margin(1e-12));
                    CHECK(C(3,3) == Approx(       839.861917098).margin(1e-12));

                    // C(:,5)
                    CHECK(C(0,4) == Approx(      -752.031490091).margin(1e-12));
                    CHECK(C(1,4) == Approx(      -22.3530558657).margin(1e-12));
                    CHECK(C(2,4) == Approx(       -740.79751002).margin(1e-12));
                    CHECK(C(3,4) == Approx(       40.9978940214).margin(1e-12));

                    // C(:,6)
                    CHECK(C(0,5) == Approx(      -339.502862133).margin(1e-12));
                    CHECK(C(1,5) == Approx(      -360.784559009).margin(1e-12));
                    CHECK(C(2,5) == Approx(      -358.484456461).margin(1e-12));
                    CHECK(C(3,5) == Approx(       -478.37625379).margin(1e-12));

                    // C(:,7)
                    CHECK(C(0,6) == Approx(        62.842747615).margin(1e-12));
                    CHECK(C(1,6) == Approx(      -26.6677971142).margin(1e-12));
                    CHECK(C(2,6) == Approx(                   0).margin(1e-12));
                    CHECK(C(3,6) == Approx(                   0).margin(1e-12));

                    // C(:,8)
                    CHECK(C(0,7) == Approx(       40.2632787465).margin(1e-12));
                    CHECK(C(1,7) == Approx(       70.5673451073).margin(1e-12));
                    CHECK(C(2,7) == Approx(                   0).margin(1e-12));
                    CHECK(C(3,7) == Approx(                   0).margin(1e-12));

                    // C(:,9)
                    CHECK(C(0,8) == Approx(      -35.3644705579).margin(1e-12));
                    CHECK(C(1,8) == Approx(       33.1205896333).margin(1e-12));
                    CHECK(C(2,8) == Approx(                   0).margin(1e-12));
                    CHECK(C(3,8) == Approx(                   0).margin(1e-12));

                    // C(:,10)
                    CHECK(C(0,9) == Approx(                   0).margin(1e-12));
                    CHECK(C(1,9) == Approx(                   0).margin(1e-12));
                    CHECK(C(2,9) == Approx(       52.6785557072).margin(1e-12));
                    CHECK(C(3,9) == Approx(      -23.9679914391).margin(1e-12));

                    // C(:,11)
                    CHECK(C(0,10) == Approx(                   0).margin(1e-12));
                    CHECK(C(1,10) == Approx(                   0).margin(1e-12));
                    CHECK(C(2,10) == Approx(       40.9812110489).margin(1e-12));
                    CHECK(C(3,10) == Approx(       69.1455786331).margin(1e-12));

                    // C(:,12)
                    CHECK(C(0,11) == Approx(                   0).margin(1e-12));
                    CHECK(C(1,11) == Approx(                   0).margin(1e-12));
                    CHECK(C(2,11) == Approx(      -45.7045366475).margin(1e-12));
                    CHECK(C(3,11) == Approx(       34.0609012808).margin(1e-12));

                }

                //--------------------------------------------------------------------------------
                // Checks for SR 
                //--------------------------------------------------------------------------------
                REQUIRE(SR.size()>0);
                REQUIRE(SR.rows()==4);
                REQUIRE(SR.cols()==4);

                // SR(:,1)
                CHECK(SR(0,0) == Approx(                0.01).margin(1e-12));
                CHECK(SR(1,0) == Approx(                   0).margin(1e-12));
                CHECK(SR(2,0) == Approx(                   0).margin(1e-12));
                CHECK(SR(3,0) == Approx(                   0).margin(1e-12));

                // SR(:,2)
                CHECK(SR(0,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,1) == Approx(                0.01).margin(1e-12));
                CHECK(SR(2,1) == Approx(                   0).margin(1e-12));
                CHECK(SR(3,1) == Approx(                   0).margin(1e-12));

                // SR(:,3)
                CHECK(SR(0,2) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,2) == Approx(                   0).margin(1e-12));
                CHECK(SR(2,2) == Approx(                0.01).margin(1e-12));
                CHECK(SR(3,2) == Approx(                   0).margin(1e-12));

                // SR(:,4)
                CHECK(SR(0,3) == Approx(                   0).margin(1e-12));
                CHECK(SR(1,3) == Approx(                   0).margin(1e-12));
                CHECK(SR(2,3) == Approx(                   0).margin(1e-12));
                CHECK(SR(3,3) == Approx(                0.01).margin(1e-12));
            }

        }

        //--------------------------------------------------------------------------------
        // Checks for h 
        //--------------------------------------------------------------------------------
        REQUIRE(h.size()>0);
        REQUIRE(h.rows()==4);
        REQUIRE(h.cols()==1);

        // h(:,1)
        CHECK(h(0,0) == Approx(       910.378043307).margin(1e-12));
        CHECK(h(1,0) == Approx(       662.479994008).margin(1e-12));
        CHECK(h(2,0) == Approx(       1059.66660101).margin(1e-12));
        CHECK(h(3,0) == Approx(       635.336558597).margin(1e-12));

    }        
}