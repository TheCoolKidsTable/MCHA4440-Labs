#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <iostream>
#include "../../src/gaussian.h"

SCENARIO("PythagoreanQR with S1 = I and S2 = 0")
{

    Eigen::MatrixXd S, S1, S2;
    // Set S1 and S2 from the lab document
    S1 = Eigen::MatrixXd::Identity(3,3);
    S2 = Eigen::MatrixXd::Zero(3,3);

    GIVEN("S1 and S2 are matrices with the same number of columns")
    {
        // Check that S1, S2 are appropriate inputs
        REQUIRE(S1.cols()==S2.cols());

        WHEN("pythagoreanQR is called"){
            // Call pythagoreanQR(S1, S2, S);
            pythagoreanQR(S1,S2,S);
            // Check S dimensions
            REQUIRE(S.rows()==S1.cols());
            REQUIRE(S.cols()==S1.cols());
            REQUIRE(S.cols()==S2.cols());
            // Check upper triangular
            Eigen::MatrixXd SUT = S.triangularView<Eigen::Upper>();
            bool isSUpperTri = (SUT.array() == S.array()).all();
            REQUIRE(isSUpperTri);
            // Check S satisfies S.'*S = S1.'*S1 + S2.'*S2
            REQUIRE(S.transpose()*S == S1.transpose()*S1+S2.transpose()*S2);          
        }
    }
}


SCENARIO("PythagoreanQR with S1 = 0 and S2 = I")
{

    Eigen::MatrixXd S, S1, S2;
    // Set S1 and S2 from the lab document
    S1 = Eigen::MatrixXd::Zero(3,3);
    S2 = Eigen::MatrixXd::Identity(3,3);

    GIVEN("S1 and S2 are matrices with the same number of columns")
    {
        // Check that S1, S2 are appropriate inputs
        REQUIRE(S1.cols()==S2.cols());

        WHEN("pythagoreanQR is called"){
            // Call pythagoreanQR(S1, S2, S);
            pythagoreanQR(S1,S2,S);
            // Check S dimensions, upper triangular, etc
            REQUIRE(S.rows()==S1.cols());
            REQUIRE(S.cols()==S1.cols());
            REQUIRE(S.cols()==S2.cols());
            Eigen::MatrixXd SUT = S.triangularView<Eigen::Upper>();
            bool isSUpperTri = (SUT.array() == S.array()).all();
            REQUIRE(isSUpperTri);
            // Check S satisfies S.'*S = S1.'*S1 + S2.'*S2
            REQUIRE(S.transpose()*S == S1.transpose()*S1+S2.transpose()*S2);          
        }
    }
}



SCENARIO("PythagoreanQR with S1 = 3*I and S2 = 4*I")
{

    Eigen::MatrixXd S, S1, S2;
    // Set S1 and S2 from the lab document
    S1 = Eigen::MatrixXd::Identity(3,3);
    S2 = Eigen::MatrixXd::Identity(3,3);
    S1 = 3*S1;
    S2 = 4*S2;

    GIVEN("S1 and S2 are matrices with the same number of columns")
    {
        // Check that S1, S2 are appropriate inputs
        REQUIRE(S1.cols()==S2.cols());

        WHEN("pythagoreanQR is called"){
            // Call pythagoreanQR(S1, S2, S);
            pythagoreanQR(S1,S2,S);
            // Check S dimensions, upper triangular, etc
            REQUIRE(S.rows()==S1.cols());
            REQUIRE(S.cols()==S1.cols());
            REQUIRE(S.cols()==S2.cols());
            Eigen::MatrixXd SUT = S.triangularView<Eigen::Upper>();
            bool isSUpperTri = (SUT.array() == S.array()).all();
            REQUIRE(isSUpperTri);
            // Check S satisfies S.'*S = S1.'*S1 + S2.'*S2
            REQUIRE(S.transpose()*S == S1.transpose()*S1+S2.transpose()*S2);          
        }
    }
}

SCENARIO("PythagoreanQR with S1 is upper triangular and S2 is a row vector")
{

    Eigen::MatrixXd S, S1, S2;
    // Set S1 and S2 from the lab document
    S1 = Eigen::MatrixXd::Zero(4,4);
    S1.row(0) << 1,2,3,4;
    S1.row(1) << 0,5,6,7;
    S1.row(2) << 0,0,8,9;
    S1.row(3) << 0,0,0,16;
    S2 = Eigen::MatrixXd::Zero(1,4);
    S2.row(0) << 0,10,-1,-3;

    GIVEN("S1 and S2 are matrices with the same number of columns")
    {
        // Check that S1, S2 are appropriate inputs
        REQUIRE(S1.cols()==S2.cols());

        WHEN("pythagoreanQR is called"){
            // Call pythagoreanQR(S1, S2, S);
            pythagoreanQR(S1,S2,S);
            // Check S dimensions, upper triangular, etc
            REQUIRE(S.rows()==S1.cols());
            REQUIRE(S.cols()==S1.cols());
            REQUIRE(S.cols()==S2.cols());
            Eigen::MatrixXd SUT = S.triangularView<Eigen::Upper>();
            bool isSUpperTri = (SUT.array() == S.array()).all();
            REQUIRE(isSUpperTri);
            // Check S satisfies S.'*S = S1.'*S1 + S2.'*S2
            // REQUIRE(S.transpose()*S == S1.transpose()*S1+S2.transpose()*S2);
            Eigen::MatrixXd ST_S = S.transpose()*S;
            CHECK(ST_S(0,0) == Approx(1));
            CHECK(ST_S(0,1) == Approx(2));
            CHECK(ST_S(0,2) == Approx(3));
            CHECK(ST_S(0,3) == Approx(4));
            CHECK(ST_S(1,0) == Approx(2));
            CHECK(ST_S(1,1) == Approx(129));
            CHECK(ST_S(1,2) == Approx(26));
            CHECK(ST_S(1,3) == Approx(13));
            CHECK(ST_S(2,0) == Approx(3));
            CHECK(ST_S(2,1) == Approx(26));
            CHECK(ST_S(2,2) == Approx(110));
            CHECK(ST_S(2,3) == Approx(129));
            CHECK(ST_S(3,0) == Approx(4));
            CHECK(ST_S(3,1) == Approx(13));
            CHECK(ST_S(3,2) == Approx(129));
            CHECK(ST_S(3,3) == Approx(411));
        }
    }
}
