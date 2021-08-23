#include <cmath>
#include <Eigen/Core>
#include "rosenbrock.h"

using std::pow;

// Templated version of Rosenbrock function
// Note: templates normally should live in a template header (.hpp), but
//       since all instantiations of this template are used only in this
//       compilation unit, its definition can live here
template<typename Scalar>
static Scalar rosenbrock(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &x)
{   
    Scalar x2 = x(0)*x(0);
    Scalar ymx2 = x(1) - x2;
    Scalar _1mx = 1 - x(0);
    return (_1mx*_1mx + 100*ymx2*ymx2);
}

// Functor for Rosenbrock function and its derivatives
double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x)
{
    return rosenbrock(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    // TODO: Gradient
    g.resize(2,1);
    g(0) = 2*x(0) - 400*x(0)*(x(1) - std::pow(x(0),2)) - 2;
    g(1) = 200*x(1) - 200*std::pow(x(0),2);
    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    // TODO: Hessian
    H.resize(2,2);
    H(0,0) = 1200*std::pow(x(0),2) - 400*x(1) + 2;
    H(0,1) = -400*x(0);
    H(1,1) = 200;
    H(1,0) = -400*x(0);
    return operator()(x, g);
}

// Functor for Rosenbrock function and its derivatives using forward-mode autodifferentiation
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd &x)
{   
    return rosenbrock(x);
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual,Eigen::Dynamic,1> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(rosenbrock<autodiff::dual>, wrt(xdual), at(xdual), fdual);
    return val(fdual);
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    // Forward-mode autodifferentiation
    using dual2nd = autodiff::HigherOrderDual<2>;
    Eigen::Matrix<dual2nd,Eigen::Dynamic,1> xdual = x.cast<dual2nd>();
    dual2nd fdual;
    H = hessian(rosenbrock<dual2nd>, wrt(xdual), at(xdual), fdual, g);
    return val(fdual);
}


// Functor for Rosenbrock function and its derivatives using reverse-mode autodifferentiation
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd &x)
{   
    return rosenbrock(x);
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    // Reverse-mode autodifferentiation
    Eigen::Matrix<autodiff::var,Eigen::Dynamic,1> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    g = gradient(fvar, xvar);
    return val(fvar);
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    // Reverse-mode autodifferentiation
    Eigen::Matrix<autodiff::var,Eigen::Dynamic,1> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    H = hessian(fvar, xvar, g);
    return val(fvar);
}
