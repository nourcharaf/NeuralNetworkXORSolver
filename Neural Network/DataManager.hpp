//
//  DataManager.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef DataManager_hpp
#define DataManager_hpp

#include <stdio.h>

#include "Parameters.hpp"

class DataManager{
    
public:
    
    static void generateXORData(std::string filePath,unsigned numberOfInstances,unsigned numberOfInputs);
    static std::vector<std::vector<double>> getInstances(std::string fileName);
};

#endif /* DataManager_hpp */
