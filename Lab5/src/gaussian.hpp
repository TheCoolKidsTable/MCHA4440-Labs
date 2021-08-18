#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <cmath>
#include <cassert>
#include <functional>
#include <Eigen/Core>
#include <Eigen/QR>

void conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::MatrixXd & Sxcond);



// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// affineTransform
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <typename Func>
void affineTransform(
    const Eigen::VectorXd       & mux,      // Input
    const Eigen::MatrixXd       & Sxx,      // Input
    Func                            h,      // Model
    Eigen::VectorXd             & muy,      // Output
    Eigen::MatrixXd             & Syy       // Output
    )
{
    assert(mux.size()>0);
    assert(mux.cols() == 1);
    assert(Sxx.cols() == Sxx.rows());
    assert(mux.rows() == Sxx.rows());

    // TODO: Transform function

    // MATLAB:
    // % Evaluate h and obtain additive noise and Jacobian
    // [muy, SR, C] = h(mux);
    // ny = length(muy);
    // R = triu(qr([Sxx*C.';SR],0));
    // Syy = R(1:ny,:);


    // TODO: Check outputs of h

}

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// Augment Gradients
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <
    typename ProcessFunc, 
    typename ParamStruct
>
void augmentGradients(ProcessFunc func, const Eigen::MatrixXd & X, const Eigen::VectorXd & u, ParamStruct & param, Eigen::MatrixXd & dX)
{
    assert(X.size()>0);
    int nx          = X.rows();
    assert(X.cols()==(2*nx + 1));

    Eigen::VectorXd x, f;
    Eigen::MatrixXd SQ, Jx;
    x               = X.col(0);

    func(x, u, param, f, SQ, Jx);
    assert(f.rows()==nx);
    assert(SQ.rows()==nx);
    assert(Jx.rows()==nx);

    dX.resize(nx, 2*nx + 1);
    dX << f, Jx*X.block(0, 1, nx, 2*nx);
}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// RK4SDEHelper
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <
    typename ProcessFunc, 
    typename ParamStruct
>
struct RK4SDEHelper
{
    void operator()(ProcessFunc func, const Eigen::VectorXd & xdw, const Eigen::VectorXd & u, ParamStruct & param, double dt, Eigen::VectorXd &xnext)
    {
        // Check that dimension of augmented state is even
        assert(xdw.size()>0);
        assert(xdw.size() % 2 == 0);
        int nx  = xdw.size()/2;
        Eigen::VectorXd    x(nx), dw(nx), f1, f2, f3, f4;

        x       = xdw.head(nx);
        dw      = xdw.tail(nx);

        func(                   x, u, param, f1);

        // Check that the output works for the first instance
        assert(f1.size()>0);
        assert(f1.cols()==1);
        assert(f1.rows()==nx);

        func(  x + (f1*dt + dw)/2, u, param, f2);
        func(  x + (f2*dt + dw)/2, u, param, f3);
        func(      x + f3*dt + dw, u, param, f4);

        xnext   = x + (f1 + 2*f2 + 2*f3 + f4)*dt/6 + dw;
    }
    void operator()(ProcessFunc func, const Eigen::VectorXd & xdw, const Eigen::VectorXd & u, ParamStruct & param, double dt, Eigen::VectorXd &xnext, Eigen::MatrixXd  &SR)
    {
        assert(xdw.size() > 0);
        assert(xdw.size() % 2 == 0);

        int nx  = xdw.size()/2;
        SR      = Eigen::MatrixXd::Zero(nx, nx);
        operator()(func, xdw, u, param, dt, xnext);

    }
    void operator()(ProcessFunc func, const Eigen::VectorXd & xdw, const Eigen::VectorXd & u, ParamStruct & param, double dt, Eigen::VectorXd &xnext, Eigen::MatrixXd  &SR, Eigen::MatrixXd &J)
    {
        assert(xdw.size() > 0);
        assert(xdw.size() % 2 == 0);
        int nxdx    = xdw.size();
        int nx      = nxdx/2;
        Eigen::VectorXd    x(nx), dw(nx);

        x       = xdw.head(nx);
        dw      = xdw.tail(nx);

        typedef Eigen::MatrixXd Matrix;

        Matrix X(nx, nxdx+1),  dW(nx, nxdx+1);
        X   <<  x, Matrix::Identity(nx, nx),    Matrix::Zero(nx, nx);
        dW  << dw, Matrix::Zero(nx, nx),        Matrix::Identity(nx, nx);

        Matrix F1, F2, F3, F4, Xnext;
        augmentGradients(func,                  X, u, param, F1);
        augmentGradients(func, X + (F1*dt + dW)/2, u, param, F2);
        augmentGradients(func, X + (F2*dt + dW)/2, u, param, F3);
        augmentGradients(func,     X + F3*dt + dW, u, param, F4);

        Xnext       = X + (F1 + 2*F2 + 2*F3 + F4)*dt/6 + dW;
        xnext       = Xnext.col(0);
        J           = Xnext.block(0, 1, nx, 2*nx);
        SR          = Matrix::Zero(nx, nx);
    }
};

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// AugmentIdentityAdapter
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <
    typename Func, 
    typename ParamStruct
>
struct AugmentIdentityAdapter
{
    void operator()(Func h, const Eigen::VectorXd & x,  const Eigen::VectorXd & u, const ParamStruct param, Eigen::VectorXd & yx)
    {
        assert(x.size()>0);
        Eigen::VectorXd y;
        h(x, u, param, y);
        assert(y.size()>0);
        
        int nx  = x.size();
        int ny  = y.size();
        int nyx = ny + nx;
        
        yx.resize(nyx);

        yx.head(ny)                 = y;
        yx.tail(nx)                 = x;
    }

    void operator()(Func h, const Eigen::VectorXd & x,  const Eigen::VectorXd & u, const ParamStruct param, Eigen::VectorXd & yx, Eigen::MatrixXd  & SRR)
    {
        assert(x.size()>0);

        Eigen::VectorXd y;
        Eigen::MatrixXd SR;
        
        h(x, u, param, y, SR);
        assert(y.size()>0);
        assert(SR.size()>0);
        
        int nx  = x.size();
        int ny  = y.size();
        int nyx = nx + ny;

        SRR.resize(nyx, nyx);
        yx.resize(nyx);

        yx.head(ny)                 = y;
        yx.tail(nx)                 = x;

        SRR.fill(0);
        SRR.topLeftCorner(ny, ny)   = SR;
    }

    void operator()(Func h, const Eigen::VectorXd & x,  const Eigen::VectorXd & u, const ParamStruct param, Eigen::VectorXd & yx, Eigen::MatrixXd  & SRR, Eigen::MatrixXd & CI)
    {
        assert(x.size()>0);

        Eigen::VectorXd y;
        Eigen::MatrixXd SR;
        Eigen::MatrixXd C;

        h(x, u, param, y, SR, C);
        assert(y.size()>0);
        assert(SR.size()>0);
        assert(C.size()>0);
        
        int nx                      = x.size();
        int ny                      = y.size();
        int nyx                     = nx + ny;
            
        CI.resize(nyx, nx);
        SRR.resize(nyx, nyx);
        yx.resize(nyx);

        yx.head(ny)                 = y;
        yx.tail(nx)                 = x;

        SRR.fill(0);
        SRR.topLeftCorner(ny, ny)   = SR;

        CI.fill(0);
        CI.topLeftCorner(ny, nx)    = C;
        CI.bottomLeftCorner(nx, nx) = Eigen::MatrixXd::Identity(nx, nx);
    }
};


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// timeUpdateContinuous
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <typename ProcessFunc, typename ParamStruct>
void timeUpdateContinuous(
    const Eigen::VectorXd   & mukm1,    // Input
    const Eigen::MatrixXd    & Skm1,    // Input
    const Eigen::VectorXd       & u,    // Input
    ProcessFunc                  pm,    // Process model
    ParamStruct             & param,    // Model parameters
    double                 timestep,    // Time step
    Eigen::VectorXd           & muk,    // Output
    Eigen::MatrixXd            & Sk     // Output
    )
{

    // Noise mean
    assert(mukm1.size()>0);
    Eigen::VectorXd muxdw(2*mukm1.size());
    muxdw.fill(0.);
    muxdw.head(mukm1.size())    = mukm1;

    // Noise covariance
    Eigen::VectorXd  f;
    Eigen::MatrixXd  SQ;
    pm(mukm1, u, param, f, SQ);

    assert(f.size()>0);
    assert(SQ.size()>0);

    Eigen::MatrixXd  Sxdw(2*mukm1.size(), 2*mukm1.size());
    Sxdw.fill(0.);
    Sxdw.topLeftCorner(mukm1.size(), mukm1.size())      = Skm1;
    Sxdw.bottomRightCorner(mukm1.size(), mukm1.size())  = SQ*std::sqrt(timestep);
    
    // RK4SDEHelper::operator()(func, xdw, u, param, dt, f, SR, J)
    // https://www.cplusplus.com/reference/functional/bind/
    RK4SDEHelper<ProcessFunc, ParamStruct> func;
    auto h  = std::bind(func, pm, std::placeholders::_1, u, param, timestep, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    affineTransform(muxdw, Sxdw, h, muk, Sk);
}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// measurementUpdateEKF
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
template <typename Func, typename ParamStruct>
void measurementUpdateEKF(
    const Eigen::VectorXd       & mux,      // Input
    const Eigen::MatrixXd       & Sxx,      // Input
    const Eigen::VectorXd         & u,      // Input
    const Eigen::VectorXd         & y,      // Input
    Func             measurementModel,      // Model
    const ParamStruct         & param,      // Input
    Eigen::VectorXd           & muxGy,      // Output
    Eigen::MatrixXd           & SxxGy       // Output
    )
{
    assert(mux.size()>0);
    assert(Sxx.size()>0);
    assert(y.size()>0);

    AugmentIdentityAdapter<Func, ParamStruct>        aia;

    // Create joint function with the following prototype
    // jointFunc(x, h, SR, H)
    auto jointFunc      = std::bind(aia,
                            measurementModel,
                            std::placeholders::_1,
                            u,
                            param,
                            std::placeholders::_2,
                            std::placeholders::_3,
                            std::placeholders::_4);

    Eigen::VectorXd muyx;
    Eigen::MatrixXd Syx;
    affineTransform(mux, Sxx, jointFunc, muyx, Syx);
    conditionGaussianOnMarginal(muyx, Syx, y, muxGy, SxxGy);
}


#endif