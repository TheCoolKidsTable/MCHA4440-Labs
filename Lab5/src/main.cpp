#include <cstdlib>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>



#include "ballistic.h"
#include "gaussian.hpp"
#include "ballistic_plot.h"


int main(int argc, char *argv[])
{
    
    std::string file_name   = "data/estimationdata.csv";
    
    // Model parameters
    BallisticParameters      param;
    BallisticProcessModel       pm;
    BallisticMeasurementModel   mm;

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

    Eigen::MatrixXd SEKF(nx, nx);
    Eigen::VectorXd muEKF(nx);

    // Initialise filter
    SEKF.fill(0);
    SEKF.diagonal() << 2200, 100, 1e-3;

    muEKF <<     14000, // Initial height
                  -450, // Initial velocity
                0.0005; // Ballistic coefficient

    std::cout << "Initial state estimate" << std::endl;
    std::cout << "muEKF = \n" << muEKF << std::endl;
    std::cout << "SEKF = \n" << SEKF << std::endl;

    std::cout << "Run filter with " << nsteps << " steps. " << std::endl;
    for (int k = 0; k < nsteps; ++k)
    {
        Eigen::VectorXd xk, yk, muf, mup;
        Eigen::MatrixXd Sf, Sp;
        xk              = xhist.col(k);
        yk              = yhist.col(k);

        // Calculate prediction density
        timeUpdateContinuous(muEKF, SEKF, u, pm, param, timestep, mup, Sp);
        
        // Calculate filtered density
        measurementUpdateEKF(mup, Sp, u, yk, mm, param, muf, Sf);
        muEKF               = muf;
        SEKF                = Sf;

        muhist.col(k)       = muEKF;
        sigmahist.col(k)    = (SEKF.transpose()*SEKF).diagonal().cwiseSqrt();
    }

    std::cout << std::endl;
    std::cout << "Final state estimate" << std::endl;
    std::cout << "muEKF = \n" << muEKF << std::endl;
    std::cout << "SEKF = \n" << SEKF << std::endl;

    // Do the plotting
    // 
    // 
    // 
    // 
    plot_simulation(thist, xhist, muhist, sigmahist, hhist, yhist);


    
    return 0;
}
