#include <catch2/catch.hpp>
#include <Eigen/Core>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "../../src/rotation.hpp"

SCENARIO("rotx: x = 0"){

    Eigen::MatrixXd Rnc;
    double x = 0;

    WHEN("Calling rotx"){
        rotx(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                   1));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                   1));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(                  -0));
                    CHECK(Rnc(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotx: x = 0.1"){

    Eigen::MatrixXd Rnc;
    double x = 0.1;

    WHEN("Calling rotx"){
        rotx(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                   1));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(      0.995004165278));
                    CHECK(Rnc(2,1) == Approx(     0.0998334166468));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(    -0.0998334166468));
                    CHECK(Rnc(2,2) == Approx(      0.995004165278));

                }
            }
        }
    }
}

SCENARIO("rotx: x = pi*5/3"){

    Eigen::MatrixXd Rnc;
    double x = M_PI*5/3;

    WHEN("Calling rotx"){
        rotx(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                   1));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                 0.5));
                    CHECK(Rnc(2,1) == Approx(     -0.866025403784));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(      0.866025403784));
                    CHECK(Rnc(2,2) == Approx(                 0.5));

                }
            }
        }
    }
}



SCENARIO("roty: x = 0"){

    Eigen::MatrixXd Rnc;
    double x = 0;

    WHEN("Calling roty"){
        roty(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                   1));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(                  -0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                   1));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("roty: x = 0.1"){

    Eigen::MatrixXd Rnc;
    double x = 0.1;

    WHEN("Calling roty"){
        roty(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(      0.995004165278));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(    -0.0998334166468));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                   1));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(     0.0998334166468));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(      0.995004165278));

                }
            }
        }
    }
}

SCENARIO("roty: x = pi*5/3"){

    Eigen::MatrixXd Rnc;
    double x = M_PI*5/3;

    WHEN("Calling roty"){
        roty(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                 0.5));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(      0.866025403784));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                   1));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(     -0.866025403784));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(                 0.5));

                }
            }
        }
    }
}


SCENARIO("rotz: x = 0"){

    Eigen::MatrixXd Rnc;
    double x = 0;

    WHEN("Calling rotz"){
        rotz(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                   1));
                    CHECK(Rnc(1,0) == Approx(                   0));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(                   0));
                    CHECK(Rnc(1,1) == Approx(                   1));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotz: x = 0.1"){

    Eigen::MatrixXd Rnc;
    double x = 0.1;

    WHEN("Calling rotz"){
        rotz(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(      0.995004165278));
                    CHECK(Rnc(1,0) == Approx(     0.0998334166468));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(    -0.0998334166468));
                    CHECK(Rnc(1,1) == Approx(      0.995004165278));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotz: x = pi*5/3"){

    Eigen::MatrixXd Rnc;
    double x = M_PI*5/3;

    WHEN("Calling rotz"){
        rotz(x, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(                 0.5));
                    CHECK(Rnc(1,0) == Approx(     -0.866025403784));
                    CHECK(Rnc(2,0) == Approx(                   0));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(      0.866025403784));
                    CHECK(Rnc(1,1) == Approx(                 0.5));
                    CHECK(Rnc(2,1) == Approx(                   0));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(                   0));
                    CHECK(Rnc(1,2) == Approx(                   0));
                    CHECK(Rnc(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rpy2rot: Thetanc = [1;2;3]"){

    Eigen::MatrixXd Rnc;
    Eigen::VectorXd Thetanc(3);
    // Thetanc - [3 x 1]: 
    Thetanc <<                  1,
                                2,
                                3;

    WHEN("Calling rotz"){
        rpy2rot(Thetanc, Rnc);
        //--------------------------------------------------------------------------------
        // Checks for Rnc 
        //--------------------------------------------------------------------------------
        THEN("Rnc is not empty"){
            REQUIRE(Rnc.size()>0);
            
            AND_THEN("Rnc has the right dimensions"){
                REQUIRE(Rnc.rows()==3);
                REQUIRE(Rnc.cols()==3);
                AND_THEN("Rnc is correct"){

                    // Rnc(:,1)
                    CHECK(Rnc(0,0) == Approx(      0.411982245666));
                    CHECK(Rnc(1,0) == Approx(    -0.0587266449276));
                    CHECK(Rnc(2,0) == Approx(     -0.909297426826));

                    // Rnc(:,2)
                    CHECK(Rnc(0,1) == Approx(     -0.833737651774));
                    CHECK(Rnc(1,1) == Approx(     -0.426917621276));
                    CHECK(Rnc(2,1) == Approx(     -0.350175488374));

                    // Rnc(:,3)
                    CHECK(Rnc(0,2) == Approx(     -0.367630462925));
                    CHECK(Rnc(1,2) == Approx(      0.902381585483));
                    CHECK(Rnc(2,2) == Approx(     -0.224845095366));

                }
            }
        }
    }
}
