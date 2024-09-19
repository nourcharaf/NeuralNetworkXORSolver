//
//  Network.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef Network_hpp
#define Network_hpp

#include <stdio.h>

#include "Layer.hpp"

class Network{
    
public:
    
    std::vector<Layer *> layers;
    double rmsError;
    double recentAverageError;
    
    static void createLayers(Network *network,std::vector<unsigned> topology);
    static void setRandomEdgeWeights(Network *network);
    static void setEdgeWeights(Network *network, std::vector<double>weights);
    static unsigned getNumberOfEdges(std::vector<unsigned> topology);
    static double getRandomWeight();
    static void trainNetwork(Network *network, std::vector<std::vector<double>> instances);
    static void feedForward(Network *network,std::vector<double> inputValues);
    static double applyTransferFunction(double x, TransferFunction transferFunction);
    static double applyTransferFunctionDerivative(double x, TransferFunction transferFunction);
    static void backPropagate(Network *network,std::vector<double> targetValues);
    static void logResults(Network *network, unsigned instanceNumber, std::vector<double> inputValues, std::vector<double> targetValues);
};

#endif /* Network_hpp */
