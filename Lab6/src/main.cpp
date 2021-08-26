#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

#include <Eigen/Core>
#include "ballistic.h"
#include "gaussian.hpp"
#include "ballistic_plot.h"

#include "rosenbrock.h"
#include "fmin.hpp"

int main(int argc, char *argv[])
{
    // ------------------------------------------------------
    // TASK 1:
    // ------------------------------------------------------
    
    Eigen::VectorXd x(2);
    x << 10.0, 10.0;
    std::cout << "Initial x =\n" << x << "\n" << std::endl;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;

    RosenbrockAnalytical func;
    std::cout << "f =\n" << func(x,g,H) << "\n" << std::endl;
    std::cout << "g =\n" << g << "\n" << std::endl;
    std::cout << "H =\n" << H << "\n" << std::endl;
    // TODO: Call fminNewtonTrust

    fminNewtonTrust(func,x,g,H,3);
    std::cout << "Final x =" << std::endl;
    std::cout <<  x << std::endl;
    std::cout << "f = " << std::endl;
    std::cout << func(x) << std::endl;
    std::cout << "g = " << std::endl;
    std::cout << g << std::endl;
    std::cout << "H = " << std::endl;
    std::cout << H << std::endl;



    // Comment out the following line to move on to task 2
    // return 0;
    // ------------------------------------------------------
    // TASK 2:
    // ------------------------------------------------------

    std::string file_name   = "data/estimationdata.csv";
    
    // Model parameters
    BallisticParameters      param;
    BallisticProcessModel       pm;
    BallisticLogLikelihood      ll;

    // Simulation parameters
    double tmax, timestep;
    int nsteps = 0, nx, ny;


    nx              = 3;
    ny              = 1;

    Eigen::VectorXd x0(nx);
    Eigen::VectorXd u;
    Eigen::MatrixXd thist, xhist, hhist, yhist;

    // Read from CSV
    std::fstream inputfile;
    inputfile.open(file_name, std::fstream::in);
    if(!inputfile.is_open()){
        std::cout << "Could not open input file \"" << file_name << "\"! Exiting" << std::endl;
        return -1;
    }
    std::cout << "Reading data from " << file_name << std::endl;

    int rows = 0;
    std::string line;
    while(getline (inputfile, line)){
        rows ++;
    }
    std::cout << "Found " << rows<< " rows within " << file_name << std::endl << std::endl;

    nsteps = rows - 1;

    thist.resize(1, nsteps);
    xhist.resize(nx, nsteps);
    hhist.resize(ny, nsteps);
    yhist.resize(ny, nsteps);

    rows = 0;
    // Start at beginning of file again
    inputfile.clear();
    inputfile.seekg(0);
    std::vector<std::string> row;
    std::string csvElement;
    while(getline (inputfile, line)){
        if (rows>0){
            int i = rows -1;
            
            row.clear();

            std::stringstream s(line);
            while (getline(s, csvElement, ',')) {
                // add all the column data
                // of a row to a vector
                row.push_back(csvElement);
            }
            
            thist(0, i) = stof(row[0]);
            xhist(0, i) = stof(row[1]);
            xhist(1, i) = stof(row[2]);
            xhist(2, i) = stof(row[3]);
            hhist(0, i) = stof(row[4]);
            yhist(0, i) = stof(row[5]);

        }
        rows ++;
    }
    timestep        = thist(0,1) - thist(0,0);


    // Run Estimation
    Eigen::MatrixXd muhist(nx, nsteps);
    Eigen::MatrixXd sigmahist(nx, nsteps);

    Eigen::MatrixXd S(nx, nx);
    Eigen::VectorXd mu(nx);

    // Initialise filter
    S.fill(0);
    S.diagonal() << 2200, 100, 1e-3;

    mu <<        14000, // Initial height
                  -450, // Initial velocity
                0.0005; // Ballistic coefficient

    std::cout << "Initial state estimate" << std::endl;
    std::cout << "mu = \n" << mu << std::endl;
    std::cout << "S = \n" << S << std::endl;

    std::cout << "Run filter with " << nsteps << " steps. " << std::endl;
    for (int k = 0; k < nsteps; ++k)
    {
        Eigen::VectorXd xk, yk, muf, mup;
        Eigen::MatrixXd Sf, Sp;
        xk              = xhist.col(k);
        yk              = yhist.col(k);

        // Calculate prediction density
        timeUpdateContinuous(mu, S, u, pm, param, timestep, mup, Sp);
        
        // Calculate filtered density
        std::cout << "[k=" << k << "] Measurement update:";
        measurementUpdateIEKF(mup, Sp, u, yk, ll, param, muf, Sf);
        std::cout << "done" << std::endl;
        mu               = muf;
        S                = Sf;

        if (mu.hasNaN()){
            std::cout << "NaNs encountered in mu. mu = \n" << mu << std::endl;
            return -1;
        }
        if (S.hasNaN()){
            std::cout << "NaNs encountered in S. S = \n" << S << std::endl;
            return -1;
        }

        muhist.col(k)       = mu;
        sigmahist.col(k)    = (S.transpose()*S).diagonal().cwiseSqrt();
    }

    std::cout << std::endl;
    std::cout << "Final state estimate" << std::endl;
    std::cout << "mu = \n" << mu << std::endl;
    std::cout << "S = \n" << S << std::endl;

    // Do the plotting
    // 
    // 
    // 
    // 
    plot_simulation(thist, xhist, muhist, sigmahist, hhist, yhist);


    
    return 0;
}
