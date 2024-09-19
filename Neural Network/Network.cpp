//
//  Network.cpp
//  Neural Network
//
//  Created by Nour Charaf on 1/22/20.
//  Copyright © 2020 Apple. All rights reserved.
//

#include "Network.hpp"
#include "assert.h"

void Network::createLayers(Network *network, std::vector<unsigned> topology){
    
    // Empty Layers
    network->layers.clear();
    
    // Add layers and nodes
    
    for (unsigned i = 0; i < topology.size(); ++i){
        
        Layer *layer = new Layer();
        network->layers.push_back(layer);
        
        unsigned numberOfNodes = topology[i];
        
        for (unsigned j = 0; j < numberOfNodes; ++j){
            
            Node *newNode = new Node();
            layer->nodes.push_back(newNode);
        }
        
        // Bias Node
        Node *biasNode = new Node();
        layer->nodes.push_back(biasNode);
        
        biasNode->value = 1;
    }
    
    // Add edges
    
    for (unsigned i = 0; i < network->layers.size() - 1; ++i){
        
        Layer *leftLayer = network->layers[i];
        Layer *rightLayer = network->layers[i + 1];
        
        for (unsigned j = 0; j < leftLayer->nodes.size(); ++j){
            
            Node *leftNode = leftLayer->nodes[j];
            
            for (unsigned k = 0; k < rightLayer->nodes.size(); ++k){
                
                Node *rightNode = rightLayer->nodes[k];
                
                Edge *edge = new Edge();
                
                leftNode->rightEdges.push_back(edge);
                rightNode->leftEdges.push_back(edge);
                
                edge->leftNode = leftNode;
                edge->rightNode = rightNode;
            }
        }
    }
}

void Network::setRandomEdgeWeights(Network *network){
    
    unsigned numberOfEdges = getNumberOfEdges(topology);
    
    std::vector<double> randomWeights;
    for (unsigned i = 0; i < numberOfEdges; ++i){
        randomWeights.push_back(getRandomWeight());
    }
    
    setEdgeWeights(network, randomWeights);
}

void Network::setEdgeWeights(Network *network, std::vector<double>weights){

    unsigned l = 0;
    
    for (unsigned i = 0; i < network->layers.size(); ++i){
        
        Layer *layer = network->layers[i];

        for (unsigned j = 0; j < layer->nodes.size(); ++j){

            Node *node = layer->nodes[j];

            for (unsigned k = 0; k < node->rightEdges.size(); ++k){

                Edge *rightEdge = node->rightEdges[k];
                rightEdge->weight = weights[l];
                l++;
            }
        }
    }
}

unsigned Network::getNumberOfEdges(std::vector<unsigned> topology){
    
    unsigned numberOfEdges = 0;
    for (unsigned i = 0; i < topology.size() - 1; ++i){
        unsigned leftLayerSize = topology[i] + 1;
        unsigned rightLayerSize = topology[i+1] + 1;
        numberOfEdges += leftLayerSize * rightLayerSize;
    }
    
    return numberOfEdges;
}

double Network::getRandomWeight(){
    return (rand()/double(RAND_MAX));
}

void Network::trainNetwork(Network *network, std::vector<std::vector<double>> instances){
    
    for (unsigned i = 0; i < instances.size(); ++i){
        
        std::vector<double> instance = instances[i];
        
        // Input Values
        std::vector<double> inputValues;
        for (unsigned i = 0; i < instance.size() - 1; ++i){
            inputValues.push_back(instance[i]);
        }
        
        // Target Values
        std::vector<double> targetValues;
        targetValues.push_back(instance.back());
        
        // Feed Forward
        feedForward(network,inputValues);
        
        // Back Propagate
        backPropagate(network,targetValues);
        
        // Log Results
        logResults(network,i + 1,inputValues,targetValues);
    }
}

void Network::feedForward(Network *network,std::vector<double> inputValues){
    
    // Get Input Layer
    Layer *inputLayer = network->layers[0];
    
    // Quick check
    assert(inputLayer->nodes.size() - 1 == inputValues.size());
    
    // Latch input values to input nodes
    for (unsigned i = 0; i < inputLayer->nodes.size() - 1; ++i){
        
        double inputValue = inputValues[i];
        
        Node *inputNode = inputLayer->nodes[i];
        inputNode->value = inputValue;
    }
    
    // Forward Propagate
    for (unsigned i = 1; i < network->layers.size(); ++i){
        
        Layer *layer = network->layers[i];
        
        for (unsigned j = 0; j < layer->nodes.size() - 1; ++j){
            
            Node *node = layer->nodes[j];
            
            double sum = 0;
            
            for (unsigned k = 0; k < node->leftEdges.size(); ++k){
                
                Edge *leftEdge = node->leftEdges[k];
                
                Node *leftNode = leftEdge->leftNode;
                
                sum += leftNode->value * leftEdge->weight;
            }
            
            node->value = applyTransferFunction(sum,transferFunction);
        }
    }
}

double Network::applyTransferFunction(double x, TransferFunction transferFunction){
    
    if (transferFunction == hyperbolicTangent){
        return tanh(x);
    }
    else if (transferFunction == sigmoid){
        return 1/(1+exp(-x));
    }
    else{
        return 0;
    }
}

double Network::applyTransferFunctionDerivative(double x, TransferFunction transferFunction){
    
    if (transferFunction == hyperbolicTangent){
        return 1 - (x * x);
    }
    else if (transferFunction == sigmoid){
        double f = applyTransferFunction(x,sigmoid);
        return f * (1 - f);
    }
    else{
        return 0;
    }
}


void Network::backPropagate(Network *network,std::vector<double> targetValues){
    
    // Root Mean Square (RMS) Error
    Layer *outputLayer = network->layers.back();
    
    unsigned numberOfOutputNodes = (int)outputLayer->nodes.size() - 1;
    double sumOfSquaredErrors = 0;
    
    for (unsigned i = 0; i < numberOfOutputNodes; ++i){
        
        Node *outputNode = outputLayer->nodes[i];
        double deltaError = targetValues[i] - outputNode->value;
        double delaErrorSquared = deltaError * deltaError;
        sumOfSquaredErrors += delaErrorSquared;
    }
    
    double averageOfSum = sumOfSquaredErrors / numberOfOutputNodes;
    
    network->rmsError = sqrt(averageOfSum);
    
    // Recent Average Error
    network->recentAverageError = (network->recentAverageError * recentAverageSmoothingFactor + network->rmsError) / (recentAverageSmoothingFactor + 1.0);
    
    // Output Layer Gradients
    for (unsigned i = 0; i < outputLayer->nodes.size() - 1; ++i){
        
        double targetValue = targetValues[i];
        
        Node *outputNode = outputLayer->nodes[i];
        
        double delta = targetValue - outputNode->value;
        
        outputNode->gradient = delta * applyTransferFunctionDerivative(outputNode->value,transferFunction);
    }
    
    // Hidden Layer Gradients
    for (unsigned i = (int)network->layers.size() - 2; i > 0; --i){
        
        Layer *hiddenLayer = network->layers[i];
        
        for (unsigned j = 0; j < hiddenLayer->nodes.size(); ++j){
            
            Node *hiddenNode = hiddenLayer->nodes[j];
            
            double dowSum = 0;
            
            for (unsigned k = 0; k < hiddenNode->rightEdges.size() - 1; ++k){
                
                Edge *rightEdge = hiddenNode->rightEdges[k];
                
                dowSum += rightEdge->weight * rightEdge->rightNode->gradient;
            }
            
            hiddenNode->gradient = dowSum * applyTransferFunctionDerivative(hiddenNode->value,transferFunction);
        }
    }
    
    // Update weights
    for (unsigned i = (int)network->layers.size() - 1; i > 0; --i){
        
        Layer *layer = network->layers[i];
        
        for (unsigned j = 0; j < layer->nodes.size() - 1; ++j){
            
            Node *node = layer->nodes[j];
            
            for (unsigned k = 0; k < node->leftEdges.size(); ++k){
                
                Edge *leftEdge = node->leftEdges[k];
                Node *leftNode = leftEdge->leftNode;
                
                double oldDeltaWeight = leftEdge->deltaWeight;
                
                double newDeltaWeight = (eta * leftNode->value * node->gradient) + (alpha * oldDeltaWeight);
                
                leftEdge->deltaWeight = newDeltaWeight;
                
                leftEdge->weight += newDeltaWeight;
            }
        }
    }
}

void Network::logResults(Network *network, unsigned instanceNumber, std::vector<double> inputValues, std::vector<double> targetValues){
    
    // Instance Number
    printf("Instance Number: %d",instanceNumber);
    printf("\n");
    
    // Inputs
    printf("Input Values: ");
    for (unsigned i = 0; i < inputValues.size(); ++i){
        printf("%f ",inputValues[i]);
    }
    printf("\n");
    
    // Targets
    printf("Target Values: ");
    for (unsigned i = 0; i < targetValues.size(); ++i){
        printf("%f ",targetValues[i]);
    }
    printf("\n");
    
    // Outputs
    printf("Output Values: ");
    Layer *outputLayer = network->layers.back();
    for (unsigned i = 0; i < outputLayer->nodes.size() - 1; ++i){
        Node *outputNode = outputLayer->nodes[i];
        printf("%f ",outputNode->value);
    }
    printf("\n");
    
    // Current RMS Error
    printf("Current RMS Error: %f",network->rmsError);
    printf("\n");
    
    // Recent Average Error
    printf("Recent Average Error: %f",network->recentAverageError);
    printf("\n");
    
    // Separator
    printf("**************************************************\n");
}
