//
//  DataManager.cpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#include "DataManager.hpp"

void DataManager::generateXORData(std::string fileName,unsigned numberOfInstances,unsigned numberOfInputs){
    
    std::string output = "";
    
    for (unsigned i = 0; i < numberOfInstances; ++i){
        
        unsigned result = 0;
        
        for (unsigned j = 0; j < numberOfInputs; ++j){
            
            unsigned x = rand()/double(RAND_MAX) + 0.5;
            
            output = output + std::to_string(x) + ",";
            
            result = j == 0 ? x : result != x;
        }
        
        output = output + std::to_string(result) + "\n";
    }
    
    std::ofstream file(fileName);
    
    file << output;
    
    file.close();
}

std::vector<std::vector<double>> DataManager::getInstances(std::string fileName){
    
    std::vector<std::vector<double>> instances;
    
    std::ifstream file(fileName);
    
    std::string line;
    
    while(std::getline(file,line)){
        
        std::stringstream linestream(line);
        std::string value;
        
        std::vector<double> allLineValues;
        
        while(std::getline(linestream,value,',')){
            allLineValues.push_back(std::stod(value));
        }
        instances.push_back(allLineValues);
    }
    
    return instances;
}
