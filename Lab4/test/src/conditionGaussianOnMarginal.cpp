#include <catch2/catch.hpp>
#include <Eigen/Core>
#include "../../src/gaussian.h"

SCENARIO("conditionGaussianOnMarginal with ny = 1, nx = 1")
{
    Eigen::VectorXd muyxjoint, muxcond, y, muxcond_exp;
    Eigen::MatrixXd Syxjoint, S1, S2, S3, Sxcond, Sxcond_exp;
    
    int ny  = 1;
    int nx  = 1;

    y.resize(ny, 1);
    muxcond_exp.resize(nx, 1);
    muyxjoint.resize(ny + nx, 1);

    S1.resize(ny, ny);
    S2.resize(ny, nx);
    S3.resize(nx, nx);

    Syxjoint.resize(ny+nx, ny+nx);
    Sxcond_exp.resize(nx, nx);
    S1      <<                    1;
    S2      <<   -0.649013765191241;
    S3      <<                    1;
    muyxjoint    <<               1,
                                  1;
    y       <<                    0;
    muxcond_exp << 1.64901376519124;
    Sxcond_exp <<                 1;
    Syxjoint << S1,                                 S2, 
                Eigen::MatrixXd::Zero(nx, ny),      S3;

    GIVEN("S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs for conditionGaussianOnMarginal")
    {
        // Check that S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs
        REQUIRE(S1.rows() == 1);
        REQUIRE(S2.rows() == 1);
        REQUIRE(S3.rows() == 1);
        REQUIRE(muyxjoint.rows() == 2);
        REQUIRE(Syxjoint.rows() == 2);
        REQUIRE(S1.cols() == 1);
        REQUIRE(S2.cols() == 1);
        REQUIRE(S3.cols() == 1);
        REQUIRE(muyxjoint.cols() == 1);
        REQUIRE(Syxjoint.cols() == 2);
        Eigen::MatrixXd S1UT = S1.triangularView<Eigen::Upper>();
        bool isSUpperTri = (S1UT.array() == S1.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd S3UT = S3.triangularView<Eigen::Upper>();
        isSUpperTri = (S3UT.array() == S3.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd SyxjointUT = Syxjoint.triangularView<Eigen::Upper>();
        isSUpperTri = (SyxjointUT.array() == Syxjoint.array()).all();
        REQUIRE(isSUpperTri);

        WHEN("conditionGaussianOnMarginal is called"){
            // Call conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            // Check dimensions
            REQUIRE(muxcond.rows() == 1);
            REQUIRE(Sxcond.rows() == 1);
            REQUIRE(muxcond.cols() == 1);
            REQUIRE(Sxcond.rows() == 1);
            Eigen::MatrixXd SxcondUT = Sxcond.triangularView<Eigen::Upper>();
            isSUpperTri = (SxcondUT.array() == Sxcond.array()).all();
            REQUIRE(isSUpperTri);
            // Check values of muxcond and Sxcond after being conditioned on measurements
            CHECK(muxcond(0,0) == Approx(muxcond_exp(0)));
            CHECK(Sxcond(0,0) == Approx(Sxcond_exp(0)));
        }
    }
}

SCENARIO("conditionGaussianOnMarginal with ny = 3, nx = 1")
{
    Eigen::VectorXd muyxjoint, muxcond, y, muxcond_exp;
    Eigen::MatrixXd Syxjoint, S1, S2, S3, Sxcond, Sxcond_exp;
    
    int ny  = 3;
    int nx  = 1;

    muyxjoint.resize(ny + nx, 1);
    S1.resize(ny, ny);
    S2.resize(ny, nx);
    S3.resize(nx, nx);
    Syxjoint.resize(ny+nx, ny+nx);
    y.resize(ny, 1);
    muxcond_exp.resize(nx, 1);
    Sxcond_exp.resize(nx, nx);

    S1 <<   -0.649013765191241,   -1.10961303850152,  -0.558680764473972,
                             0,  -0.845551240007797,   0.178380225849766,
                             0,                   0,  -0.196861446475943;
    S2 <<    0.586442621667069,
            -0.851886969622469,
             0.800320709801823;
    S3 <<    -1.50940472473439;
    muyxjoint <<             1,
                             1,
                             1,
                             1;
    y <<     0.875874147834533,
             -0.24278953633334,
             0.166813439453503;

    muxcond_exp <<    3.91058676524518;
    Sxcond_exp  <<   -1.50940472473439;


    Syxjoint << S1,                                 S2, 
                Eigen::MatrixXd::Zero(nx, ny),      S3;

    GIVEN("S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs for conditionGaussianOnMarginal")
    {
        // Check that S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs
        REQUIRE(S1.rows() == 3);
        REQUIRE(S2.rows() == 3);
        REQUIRE(S3.rows() == 1);
        REQUIRE(muyxjoint.rows() == 4);
        REQUIRE(Syxjoint.rows() == 4);
        REQUIRE(S1.cols() == 3);
        REQUIRE(S2.cols() == 1);
        REQUIRE(S3.cols() == 1);
        REQUIRE(muyxjoint.cols() == 1);
        REQUIRE(Syxjoint.cols() == 4);
        //??
        Eigen::MatrixXd S1UT = S1.triangularView<Eigen::Upper>();
        bool isSUpperTri = (S1UT.array() == S1.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd S3UT = S3.triangularView<Eigen::Upper>();
        isSUpperTri = (S3UT.array() == S3.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd SyxjointUT = Syxjoint.triangularView<Eigen::Upper>();
        isSUpperTri = (SyxjointUT.array() == Syxjoint.array()).all();
        REQUIRE(isSUpperTri);

        WHEN("conditionGaussianOnMarginal is called"){
            // Call conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            // Check dimensions
            REQUIRE(muxcond.rows() == 1);
            REQUIRE(Sxcond.rows() == 1);
            REQUIRE(muxcond.cols() == 1);
            REQUIRE(Sxcond.rows() == 1);
            Eigen::MatrixXd SxcondUT = Sxcond.triangularView<Eigen::Upper>();
            isSUpperTri = (SxcondUT.array() == Sxcond.array()).all();
            REQUIRE(isSUpperTri);
            // Check values of muxcond and Sxcond after being conditioned on measurements
            CHECK(muxcond(0,0) == Approx(muxcond_exp(0)));
            CHECK(Sxcond(0,0) == Approx(Sxcond_exp(0)));
        }
    }
}


SCENARIO("conditionGaussianOnMarginal with ny = 1, nx = 3")
{
    Eigen::VectorXd muyxjoint, muxcond, y, muxcond_exp;
    Eigen::MatrixXd Syxjoint, S1, S2, S3, Sxcond, Sxcond_exp;
    
    int ny  = 1;
    int nx  = 3;

    muyxjoint.resize(ny + nx, 1);
    S1.resize(ny, ny);
    S2.resize(ny, nx);
    S3.resize(nx, nx);
    Syxjoint.resize(ny+nx, ny+nx);
    y.resize(ny, 1);
    muxcond_exp.resize(nx, 1);
    Sxcond_exp.resize(nx, nx);

    S1 <<   -0.649013765191241;
    S2 <<     1.18116604196553,  -0.758453297283692,   -1.10961303850152;
    S3 <<   -0.845551240007797,   0.178380225849766,  -0.851886969622469,
                             0,  -0.196861446475943,   0.800320709801823,
                             0,                   0,   -1.50940472473439;
    muyxjoint <<             1,
                             1,
                             1,
                             1;
    y <<     0.875874147834533;

    muxcond_exp <<   1.22590159002973,
                    0.854943505203921,
                    0.787783139025826;
    Sxcond_exp << -0.845551240007797,   0.178380225849766,  -0.851886969622469,
                                   0,  -0.196861446475943,   0.800320709801823,
                                   0,                   0,   -1.50940472473439;


    Syxjoint << S1,                                 S2, 
                Eigen::MatrixXd::Zero(nx, ny),      S3;

    GIVEN("S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs for conditionGaussianOnMarginal")
    {
        // Check that S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs
        REQUIRE(S1.rows() == 1);
        REQUIRE(S2.rows() == 1);
        REQUIRE(S3.rows() == 3);
        REQUIRE(muyxjoint.rows() == 4);
        REQUIRE(Syxjoint.rows() == 4);
        REQUIRE(S1.cols() == 1);
        REQUIRE(S2.cols() == 3);
        REQUIRE(S3.cols() == 3);
        REQUIRE(muyxjoint.cols() == 1);
        REQUIRE(Syxjoint.cols() == 4);

        Eigen::MatrixXd S1UT = S1.triangularView<Eigen::Upper>();
        bool isSUpperTri = (S1UT.array() == S1.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd S3UT = S3.triangularView<Eigen::Upper>();
        isSUpperTri = (S3UT.array() == S3.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd SyxjointUT = Syxjoint.triangularView<Eigen::Upper>();
        isSUpperTri = (SyxjointUT.array() == Syxjoint.array()).all();
        REQUIRE(isSUpperTri);

        WHEN("conditionGaussianOnMarginal is called"){
            // Call conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            // Check dimensions
            REQUIRE(muxcond.rows() == 3);
            REQUIRE(Sxcond.rows() == 3);
            REQUIRE(muxcond.cols() == 1);
            REQUIRE(Sxcond.rows() == 3);
            Eigen::MatrixXd SxcondUT = Sxcond.triangularView<Eigen::Upper>();
            isSUpperTri = (SxcondUT.array() == Sxcond.array()).all();
            REQUIRE(isSUpperTri);
            // Check values of muxcond and Sxcond after being conditioned on measurements
            CHECK(muxcond(0,0) == Approx(muxcond_exp(0)));
            CHECK(Sxcond(0,0) == Approx(Sxcond_exp(0)));
        }
    }
}


SCENARIO("conditionGaussianOnMarginal with ny = 3, nx = 3")
{
    Eigen::VectorXd muyxjoint, muxcond, y, muxcond_exp;
    Eigen::MatrixXd Syxjoint, S1, S2, S3, Sxcond, Sxcond_exp;
    
    int ny  = 3;
    int nx  = 3;

    muyxjoint.resize(ny + nx, 1);
    S1.resize(ny, ny);
    S2.resize(ny, nx);
    S3.resize(nx, nx);
    Syxjoint.resize(ny+nx, ny+nx);
    y.resize(ny, 1);
    muxcond_exp.resize(nx, 1);
    Sxcond_exp.resize(nx, nx);

    S1 <<   -0.649013765191241,   -1.10961303850152,  -0.558680764473972,
                             0,  -0.845551240007797,   0.178380225849766,
                             0,                   0,  -0.196861446475943;
    S2 <<    0.586442621667069,   -1.50940472473439,   0.166813439453503,
            -0.851886969622469,   0.875874147834533,   -1.96541870928278,
             0.800320709801823,   -0.24278953633334,   -1.27007139263854;
    S3 <<     1.17517126546302,   0.603658445825815,   -1.86512257453063,
                             0,     1.7812518932425,   -1.05110705924059,
                             0,                   0,  -0.417382047996795;
    muyxjoint <<             1,
                             1,
                             1,
                             1,
                             1,
                             1;
    y <<     1.40216228633781,
            -1.36774699097611,
           -0.292534999151873;

    muxcond_exp <<  6.84085527167069,
                    2.28421796254053,
                   -20.9360494953007;
    Sxcond_exp <<   1.17517126546302,   0.603658445825815,   -1.86512257453063,
                                   0,     1.7812518932425,   -1.05110705924059,
                                   0,                   0,  -0.417382047996795;


    Syxjoint << S1,                                 S2, 
                Eigen::MatrixXd::Zero(nx, ny),      S3;

    GIVEN("S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs for conditionGaussianOnMarginal")
    {
        // Check that S1, S2, S3, muyxjoint, and Syxjoint are appropriate inputs
        REQUIRE(S1.rows() == 3);
        REQUIRE(S2.rows() == 3);
        REQUIRE(S3.rows() == 3);
        REQUIRE(muyxjoint.rows() == 6);
        REQUIRE(Syxjoint.rows() == 6);
        REQUIRE(S1.cols() == 3);
        REQUIRE(S2.cols() == 3);
        REQUIRE(S3.cols() == 3);
        REQUIRE(muyxjoint.cols() == 1);
        REQUIRE(Syxjoint.cols() == 6);

        Eigen::MatrixXd S1UT = S1.triangularView<Eigen::Upper>();
        bool isSUpperTri = (S1UT.array() == S1.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd S3UT = S3.triangularView<Eigen::Upper>();
        isSUpperTri = (S3UT.array() == S3.array()).all();
        REQUIRE(isSUpperTri);

        Eigen::MatrixXd SyxjointUT = Syxjoint.triangularView<Eigen::Upper>();
        isSUpperTri = (SyxjointUT.array() == Syxjoint.array()).all();
        REQUIRE(isSUpperTri);

        WHEN("conditionGaussianOnMarginal is called"){
            // Call conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            conditionGaussianOnMarginal(muyxjoint, Syxjoint, y, muxcond, Sxcond);
            // Check dimensions
            REQUIRE(muxcond.rows() == 3);
            REQUIRE(Sxcond.rows() == 3);
            REQUIRE(muxcond.cols() == 1);
            REQUIRE(Sxcond.rows() == 3);
            Eigen::MatrixXd SxcondUT = Sxcond.triangularView<Eigen::Upper>();
            isSUpperTri = (SxcondUT.array() == Sxcond.array()).all();
            REQUIRE(isSUpperTri);
            // Check values of muxcond and Sxcond after being conditioned on measurements
            CHECK(muxcond(0,0) == Approx(muxcond_exp(0)));
            CHECK(Sxcond(0,0) == Approx(Sxcond_exp(0)));
        }
    }
}