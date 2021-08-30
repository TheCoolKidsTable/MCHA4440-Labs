#include <catch2/catch.hpp>
#include <Eigen/Core>

#include "../../src/rotation.hpp"

SCENARIO("rot2rpy: Rnc is identity"){

    Eigen::MatrixXd Rnc(3,3);
    Eigen::VectorXd Thetanc;

    // Rnc - [3 x 3]: 
    Rnc <<                  1,                 0,                 0,
                            0,                 1,                 0,
                            0,                 0,                 1;
    WHEN("rot2rpy is called"){
        rot2rpy(Rnc, Thetanc);
        //--------------------------------------------------------------------------------
        // Checks for Thetanc 
        //--------------------------------------------------------------------------------
        THEN("Thetanc is not empty"){
            REQUIRE(Thetanc.size()>0);
            
            AND_THEN("Thetanc has the right dimensions"){
                REQUIRE(Thetanc.rows()==3);
                REQUIRE(Thetanc.cols()==1);
                AND_THEN("Thetanc is correct"){

                    // Thetanc(:,1)
                    CHECK(Thetanc(0,0) == Approx(                   0));
                    CHECK(Thetanc(1,0) == Approx(                  -0));
                    CHECK(Thetanc(2,0) == Approx(                   0));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: Rnc is rotation about x axis by 0.5 rad"){

    Eigen::MatrixXd Rnc(3,3);
    Eigen::VectorXd Thetanc;

    // Rnc - [3 x 3]: 
    Rnc <<                  1,                 0,                 0,
                            0,      0.8775825619,     -0.4794255386,
                            0,      0.4794255386,      0.8775825619;
    WHEN("rot2rpy is called"){
        rot2rpy(Rnc, Thetanc);
        //--------------------------------------------------------------------------------
        // Checks for Thetanc 
        //--------------------------------------------------------------------------------
        THEN("Thetanc is not empty"){
            REQUIRE(Thetanc.size()>0);
            
            AND_THEN("Thetanc has the right dimensions"){
                REQUIRE(Thetanc.rows()==3);
                REQUIRE(Thetanc.cols()==1);
                AND_THEN("Thetanc is correct"){

                    // Thetanc(:,1)
                    CHECK(Thetanc(0,0) == Approx(                 0.5));
                    CHECK(Thetanc(1,0) == Approx(                  -0));
                    CHECK(Thetanc(2,0) == Approx(                   0));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: Rnc is rotation about y axis by 0.5 rad"){

    Eigen::MatrixXd Rnc(3,3);
    Eigen::VectorXd Thetanc;

    // Rnc - [3 x 3]: 
    Rnc <<       0.8775825619,                 0,      0.4794255386,
                            0,                 1,                 0,
                -0.4794255386,                 0,      0.8775825619;
    WHEN("rot2rpy is called"){
        rot2rpy(Rnc, Thetanc);
        //--------------------------------------------------------------------------------
        // Checks for Thetanc 
        //--------------------------------------------------------------------------------
        THEN("Thetanc is not empty"){
            REQUIRE(Thetanc.size()>0);
            
            AND_THEN("Thetanc has the right dimensions"){
                REQUIRE(Thetanc.rows()==3);
                REQUIRE(Thetanc.cols()==1);
                AND_THEN("Thetanc is correct"){

                    // Thetanc(:,1)
                    CHECK(Thetanc(0,0) == Approx(                   0));
                    CHECK(Thetanc(1,0) == Approx(                 0.5));
                    CHECK(Thetanc(2,0) == Approx(                   0));

                }
            }
        }
    }
}


SCENARIO("rot2rpy: Rnc is rotation about z axis by 0.5 rad"){

    Eigen::MatrixXd Rnc(3,3);
    Eigen::VectorXd Thetanc;

    // Rnc - [3 x 3]: 
    Rnc <<       0.8775825619,     -0.4794255386,                 0,
                 0.4794255386,      0.8775825619,                 0,
                            0,                 0,                 1;
    WHEN("rot2rpy is called"){
        rot2rpy(Rnc, Thetanc);
        //--------------------------------------------------------------------------------
        // Checks for Thetanc 
        //--------------------------------------------------------------------------------
        THEN("Thetanc is not empty"){
            REQUIRE(Thetanc.size()>0);
            
            AND_THEN("Thetanc has the right dimensions"){
                REQUIRE(Thetanc.rows()==3);
                REQUIRE(Thetanc.cols()==1);
                AND_THEN("Thetanc is correct"){

                    // Thetanc(:,1)
                    CHECK(Thetanc(0,0) == Approx(                   0));
                    CHECK(Thetanc(1,0) == Approx(                  -0));
                    CHECK(Thetanc(2,0) == Approx(                 0.5));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: Rnc is a rotation about all axes"){

    Eigen::MatrixXd Rnc(3,3);
    Eigen::VectorXd Thetanc;

    // Rnc - [3 x 3]: 
    Rnc <<       0.4119822457,     -0.8337376518,     -0.3676304629,
               -0.05872664493,     -0.4269176213,      0.9023815855,
                -0.9092974268,     -0.3501754884,     -0.2248450954;
    WHEN("rot2rpy is called"){
        rot2rpy(Rnc, Thetanc);
        //--------------------------------------------------------------------------------
        // Checks for Thetanc 
        //--------------------------------------------------------------------------------
        THEN("Thetanc is not empty"){
            REQUIRE(Thetanc.size()>0);
            Eigen::MatrixXd R_act;
            rpy2rot(Thetanc, R_act);

            AND_THEN("rot2rpy is an inverse mapping"){
                // R_act(:,1)
                CHECK(R_act(0,0) == Approx(Rnc(0,0)));
                CHECK(R_act(1,0) == Approx(Rnc(1,0)));
                CHECK(R_act(2,0) == Approx(Rnc(2,0)));

                // R_act(:,2)
                CHECK(R_act(0,1) == Approx(Rnc(0,1)));
                CHECK(R_act(1,1) == Approx(Rnc(1,1)));
                CHECK(R_act(2,1) == Approx(Rnc(2,1)));

                // R_act(:,3)
                CHECK(R_act(0,2) == Approx(Rnc(0,2)));
                CHECK(R_act(1,2) == Approx(Rnc(1,2)));
                CHECK(R_act(2,2) == Approx(Rnc(2,2)));
            }
        }
    }
}
