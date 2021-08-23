#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
// #define EIGEN_NO_DEBUG // Disable runtime assertions, e.g., bounds checking
#include <Eigen/Core>
#include "../../src/rosenbrock.h"


SCENARIO("RosenbrockAnalytical at origin")
{
    Eigen::VectorXd x(2);

    GIVEN("x = (0,0)")
    {
        x << 0, 0;

        RosenbrockAnalytical func;
        double f;

        // One argument
        WHEN("Evaluating f = RosenbrockAnalytical(x)")
        {
            f = func(x);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(1.0));
            }
        }

        // Two arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x,g)")
        {
            Eigen::VectorXd g;
            f = func(x, g);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(1.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == Approx(-2.0));
                CHECK(g(1) == Approx(0.0));
            }
        }
        
        // Three arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x,g,H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            f = func(x, g, H);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(1.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == Approx(-2.0));
                CHECK(g(1) == Approx(0.0));
            }
            THEN("H matches expected values")
            {
                REQUIRE(H.rows() == 2);
                REQUIRE(H.cols() == 2);
                CHECK(H(0,0) == Approx(2.0));
                CHECK(H(0,1) == Approx(0.0));
                CHECK(H(1,0) == Approx(0.0));
                CHECK(H(1,1) == Approx(200));
            }
        }
    }
}

SCENARIO("RosenbrockAnalytical at minimiser")
{
    Eigen::VectorXd x(2);

    GIVEN("x = (1,1)")
    {
        x << 1, 1;

        RosenbrockAnalytical func;
        double f;

        // One argument
        WHEN("Evaluating f = RosenbrockAnalytical(x)")
        {
            f = func(x);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(0.0));
            }
        }

        // Two arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x,g)")
        {
            Eigen::VectorXd g;
            f = func(x, g);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(0.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == Approx(0.0));
                CHECK(g(1) == Approx(0.0));
            }
        }
        
        // Three arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x,g,H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            f = func(x, g, H);
            THEN("f matches expected value")
            {
                CHECK(f == Approx(0.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == Approx(0.0));
                CHECK(g(1) == Approx(0.0));
            }
            THEN("H matches expected values")
            {
                REQUIRE(H.rows() == 2);
                REQUIRE(H.cols() == 2);
                CHECK(H(0,0) == Approx(802.0));
                CHECK(H(0,1) == Approx(-400.0));
                CHECK(H(1,0) == Approx(-400.0));
                CHECK(H(1,1) == Approx(200));
            }
        }
    }
}

// TEST_CASE("Rosenbrock gradient performance")
// {
//     Eigen::VectorXd x(2);
//     x << 0, 0;

//     double f;
//     Eigen::VectorXd g;

//     RosenbrockAnalytical func;
//     BENCHMARK("Analytical derivatives")
//     {
//         f = func(x,g);
//     };

//     RosenbrockFwdAutoDiff funcFwd;
//     BENCHMARK("Forward-mode autodifferentiation")
//     {
//         f = funcFwd(x,g);
//     };

//     RosenbrockRevAutoDiff funcRev;
//     BENCHMARK("Reverse-mode autodifferentiation")
//     {
//         f = funcRev(x,g);
//     };
// }

TEST_CASE("Rosenbrock Hessian performance")
{
    Eigen::VectorXd x(2);
    x << 0, 0;

    double f;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;

    RosenbrockAnalytical func;
    BENCHMARK("Analytical derivatives")
    {
        f = func(x,g,H);
    };

    RosenbrockFwdAutoDiff funcFwd;
    BENCHMARK("Forward-mode autodifferentiation")
    {
        f = funcFwd(x,g,H);
    };

    RosenbrockRevAutoDiff funcRev;
    BENCHMARK("Reverse-mode autodifferentiation")
    {
        f = funcRev(x,g,H);
    };
}
