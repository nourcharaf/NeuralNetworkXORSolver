//
//  Layer.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>

#include "Node.hpp"

class Layer{
    
public:
    
    std::vector<Node *> nodes;
};

#endif /* Layer_hpp */
