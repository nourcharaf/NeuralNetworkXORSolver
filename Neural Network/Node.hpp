//
//  Node.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef Node_hpp
#define Node_hpp

#include <stdio.h>

#include "Parameters.hpp"
#include "Edge.hpp"

class Node{
    
public:
    std::vector<Edge *> leftEdges;
    std::vector<Edge *> rightEdges;
    double value;
    double gradient;
};

#endif /* Node_hpp */
