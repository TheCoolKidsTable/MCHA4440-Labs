#include <iostream>
#include <cstdlib>

#include <Eigen/Core>

int main(int argc, char *argv[])
{

    // ------------------------------------------------
    // Task 1: Preliminaries
    // ------------------------------------------------

    // Print Version
    std::cout << "1. Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    // Create a vector of type vectorXd
    Eigen::VectorXd x(3);
    x << 1, 3.2, 0.01;
    std::cout << "2. Create a vector" << std::endl;
    std::cout << "x = "<< std::endl;
    std::cout << x << std::endl;
    // Create a random matrix A 
    std::cout << "2. Create a random matrix" << std::endl;
    std::srand(42);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(4,3);
    std::cout << "A.size() = (" << A.rows() << "," << A.cols() << ")" << std::endl;
    std::cout << "A.rows() = " << A.rows() << std::endl;
    std::cout << "A.cols() = " << A.cols() << std::endl;
    std::cout << "A =" << std::endl;
    std::cout << A << std::endl;
    std::cout << "A.transpose =" << std::endl;
    std::cout << A.transpose() << std::endl;
    std::cout << "4. Matrix multiplication" << std::endl;
    std::cout << "A*x = " << std::endl;
    std::cout << A*x << std::endl;
    std::cout << "5. Using the << operator, create a matrix B and C" << std::endl;
    Eigen::MatrixXd B(4,6);
    B << A, A;
    // Create Matrix B and C
    std::cout << "B.size() = (" << B.rows() << "," << B.cols() << ")" << std::endl;
    std::cout << "B.rows() = " << B.rows() << std::endl;
    std::cout << "B.cols() = " << B.cols() << std::endl;
    std::cout << "B = " << std::endl;
    std::cout << B << std::endl;
    Eigen::MatrixXd C(8,3);
    C << A, A;
        std::cout << "C.size() = (" << C.rows() << "," << C.cols() << ")" << std::endl;
    std::cout << "C.rows() = " << C.rows() << std::endl;
    std::cout << "C.cols() = " << C.cols() << std::endl;
    std::cout << "C = " << std::endl;
    std::cout << C << std::endl;
    // Use block 
    Eigen::MatrixXd D(1,3);
    D = B.block(1,2,1,3);
    std::cout << "D.size() = (" << D.rows() << "," << D.cols() << ")" << std::endl;
    std::cout << "D.rows() = " << D.rows() << std::endl;
    std::cout << "D.cols() = " << D.cols() << std::endl;
    std::cout << "D = " << std::endl;
    std::cout << D << std::endl;
    //TODO
    // Select
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(4,6);
    Eigen::MatrixXd E(4,6);
    E = (B.array() > 0.5).select(Z, B);
    std::cout << "E.size() = (" << E.rows() << "," << E.cols() << ")" << std::endl;
    std::cout << "E.rows() = " << E.rows() << std::endl;
    std::cout << "E.cols() = " << E.cols() << std::endl;
    std::cout << "E = " << std::endl;
    std::cout << E << std::endl;
    // Row wise
    Eigen::MatrixXd F(4,6);
    Eigen::VectorXd v(6);   
    v << 1,3,5,7,4,6;
    F = B;
    F.rowwise() += v.transpose();
    std::cout << "F.size() = (" << F.rows() << "," << F.cols() << ")" << std::endl;
    std::cout << "F.rows() = " << F.rows() << std::endl;
    std::cout << "F.cols() = " << F.cols() << std::endl; 
    std::cout << "F = " << std::endl;
    std::cout << F << std::endl;
    // Use typedef
    typedef Eigen::Matrix<double, 7, 1>  Vector7d;
    Vector7d v1, v2;
    v1 = Vector7d::Random(7,1);
    v2.fill(42.0);
        std::cout << "v1.size() = (" << v1.rows() << "," << v1.cols() << ")" << std::endl;
    std::cout << "v1.rows() = " << v1.rows() << std::endl;
    std::cout << "v1.cols() = " << v1.cols() << std::endl; 
    std::cout << "v1 = " << std::endl;
    std::cout << v1 << std::endl;
    std::cout << "v2.size() = (" << v2.rows() << "," << v2.cols() << ")" << std::endl;
    std::cout << "v2.rows() = " << v2.rows() << std::endl;
    std::cout << "v2.cols() = " << v2.cols() << std::endl; 
    std::cout << "v2 = " << std::endl;
    std::cout << v2 << std::endl;
    std::cout << "Size of v1 + v2 = (" << v2.rows() << "," << v2.cols() << ")" << std::endl;
    std::cout << "v1 + v2 = " << std::endl;
    std::cout << v1 + v2 << std::endl;
    return 0;

    // ------------------------------------------------
    // Task 2 and 3 do not belong in here!
    // ------------------------------------------------
}
