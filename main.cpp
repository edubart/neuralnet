#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include "ann.h"

void ann_load_tain_sets(ANNet *net, const char *filename)
{
    nnreal in[100];
    nnreal out[100];
    int i;
    nnreal tmp;
    std::ifstream fin(filename);
    while(!fin.eof()) {
        for(i=0;i<net->inputLayer->numNeurons;++i) {
            fin >> tmp;
            in[i] = tmp;
        }
        for(i=0;i<net->outputLayer->numNeurons;++i) {
            fin >> tmp;
            out[i] = tmp;
        }
        ann_add_train_set(net, in, out);
    }
}

void xor_test()
{
    uint numInput = 2;
    uint numHidden = 3;
    uint numOutput = 1;
    nnreal learningRate = 0.7;
    nnreal momentum = 0;
    nnreal steepness = 1;
    ANNActivateFunction activateFunction = ANN_SIGMOID;
    ANNet net;

    printf("xor test\n");

    ann_init(&net);
    ann_add_layer(&net, numInput); // input layer
    ann_add_layer(&net, numHidden); // hidden layer
    ann_add_layer(&net, numOutput); // output layer
    ann_set_activate_function(&net, activateFunction, ANN_ALL_LAYERS);
    ann_set_activate_function(&net, ANN_LINEAR, ANN_OUTPUT_LAYER);
    ann_set_learning_rate(&net, learningRate, ANN_ALL_LAYERS);
    ann_set_momentum(&net, momentum, ANN_ALL_LAYERS);
    ann_set_steepness(&net, steepness, ANN_ALL_LAYERS);
    ann_load_tain_sets(&net, "datasets/xor.train");
    ann_train(&net, 100000, 1000, 0.01, 0.01);
}

void mushroom_test()
{
    uint numInput = 125;
    uint numHidden = 32;
    uint numOutput = 2;
    nnreal learningRate = 0.7;
    nnreal momentum = 0;
    nnreal steepness = 1;
    ANNActivateFunction activateFunction = ANN_SIGMOID;
    ANNet net;

    printf("mushroom room test\n");

    ann_init(&net);
    ann_add_layer(&net, numInput); // input layer
    ann_add_layer(&net, numHidden); // hidden layer
    ann_add_layer(&net, numOutput); // output layer
    ann_set_activate_function(&net, activateFunction, ANN_ALL_LAYERS);
    ann_set_learning_rate(&net, learningRate, ANN_ALL_LAYERS);
    ann_set_momentum(&net, momentum, ANN_ALL_LAYERS);
    ann_set_steepness(&net, steepness, ANN_ALL_LAYERS);
    ann_load_tain_sets(&net, "datasets/mushroom.train");
    ann_train(&net, 300, 10, 0.01, 0.1);
}

void robot_test()
{
    uint numInput = 48;
    uint numHidden = 96;
    uint numOutput = 3;
    nnreal learningRate = 0.7;
    nnreal momentum = 0.4;
    nnreal steepness = 1;
    ANNActivateFunction activateFunction = ANN_SIGMOID;
    ANNet net;

    printf("robot test\n");

    ann_init(&net);
    ann_add_layer(&net, numInput); // input layer
    ann_add_layer(&net, numHidden); // hidden layer
    ann_add_layer(&net, numOutput); // output layer
    ann_set_activate_function(&net, activateFunction, ANN_ALL_LAYERS);
    ann_set_learning_rate(&net, learningRate, ANN_ALL_LAYERS);
    ann_set_momentum(&net, momentum, ANN_ALL_LAYERS);
    ann_set_steepness(&net, steepness, ANN_ALL_LAYERS);
    ann_load_tain_sets(&net, "datasets/robot.train");
    ann_train(&net, 3000, 100, 0.01, 0.1);
}

int main(int argc, char **argv)
{
    xor_test();
    mushroom_test();
    robot_test();

    return 0;
}
