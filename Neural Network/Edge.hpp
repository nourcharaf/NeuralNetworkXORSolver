//
//  Edge.hpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#ifndef Edge_hpp
#define Edge_hpp

#include <stdio.h>

class Node;

class Edge{
    
public:
    
    Node *leftNode;
    Node *rightNode;
    double weight;
    double deltaWeight;
};

#endif /* Edge_hpp */
