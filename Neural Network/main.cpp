//
//  main.cpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#include "Parameters.hpp"
#include "DataManager.hpp"
#include "Network.hpp"

int main(int argc, const char * argv[]) {
    
    // Seed for Random Number Generator
    srand((int)time(NULL));
    
    // Generate XOR Data
    DataManager::generateXORData(fileName, numberOfInstances, numberOfInputs);
    
    // Get Instances from File
    std::vector<std::vector<double>> instances = DataManager::getInstances(fileName);
    
    // Create Network
    Network *network = new Network();
    
    // Create Network Layers
    Network::createLayers(network, topology);
    
    // Set Random Edge Weights
    Network::setRandomEdgeWeights(network);
    
    // Train Network with Instances
    Network::trainNetwork(network, instances);
    
    return 0;
}
