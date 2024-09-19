//
//  Parameters.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef Parameters_hpp
#define Parameters_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

// Data Parameters
static std::string fileName = "xor_data.txt";
static unsigned numberOfInstances = 2000;
static unsigned numberOfInputs = 2;
static unsigned numberOfOutputs = 1;

// Neural Network Parameters
enum TransferFunction {
    hyperbolicTangent,
    sigmoid
};
static TransferFunction transferFunction = hyperbolicTangent;
static std::vector<unsigned> topology = {numberOfInputs,2,numberOfOutputs};
static double eta = 0.15;
static double alpha = 0.5;
static double recentAverageSmoothingFactor = 100; // Number of training samples to average over

#endif /* Parameters_hpp */
