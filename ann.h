#ifndef ANN_H
#define ANN_H

#include <stdlib.h>

typedef unsigned short ushort;
typedef unsigned int uint;
typedef double nnreal;

typedef struct ANNSynapse ANNSynapse;
typedef struct ANNNeuron ANNNeuron;
typedef struct ANNLayer ANNLayer;
typedef struct ANNSet ANNSet;
typedef struct ANNet ANNet;

typedef enum
{
    ANN_LINEAR,
    ANN_SIGMOID,
    ANN_SIGMOID_SYMMETRIC
} ANNActivateFunction;

typedef enum {
    ANN_ALL_LAYERS = 0,
    ANN_HIDDEN_LAYERS = -1,
    ANN_OUTPUT_LAYER = -2,
    ANN_INPUT_LAYER = -3,
} ANNLayerGroup;

struct ANNSynapse {
    nnreal weight;
    nnreal weightChange;
    ANNNeuron *inputNeuron;
    ANNNeuron *outputNeuron;
};

struct ANNNeuron {
    nnreal delta;
    nnreal value;
    nnreal bias;
    nnreal biasChange;
    ushort numInputSynapses;
    ushort numOutputSynapses;
    ANNSynapse **inputSynapses;
    ANNSynapse **outputSynapses;
    ANNLayer *layer;
};

struct ANNLayer {
    ANNActivateFunction activateFunc;
    nnreal learningRate;
    nnreal momentum;
    nnreal steepness;
    ushort numNeurons;
    ANNNeuron **neurons;
    ANNLayer *prevLayer;
    ANNLayer *nextLayer;
};

struct ANNSet {
    nnreal *input;
    nnreal *output;
};

struct ANNet {
    ANNLayer *inputLayer;
    ANNLayer *outputLayer;
    ANNSet **trainSets;
    uint numTrainSets;
    uint randSeed;
    nnreal adptativeLearningRate;
};

#ifdef __cplusplus
extern "C" {
#endif

void ann_init(ANNet *net);

ANNLayer *ann_add_layer(ANNet *net, int numNeurons);
void ann_add_train_set(ANNet *net, nnreal *input, nnreal *output);

void ann_run(ANNet *net, nnreal *input, nnreal *output);

void ann_train_set(ANNet *net, nnreal *input, nnreal *output);
void ann_train_sets(ANNet *net);
void ann_train(ANNet *net, uint maxEpochs, uint epochsBetweenReports, nnreal minimumRMSE, nnreal bitFailLimit);

nnreal ann_calc_set_rmse(ANNet *net, nnreal *input, nnreal *output);
nnreal ann_calc_rmse(ANNet *net);

void ann_set_learning_rate(ANNet *net, nnreal learningRate, ANNLayerGroup layerGroup);
void ann_set_momentum(ANNet *net, nnreal momentum, ANNLayerGroup layerGroup);
void ann_set_steepness(ANNet *net, nnreal steepness, ANNLayerGroup layerGroup);
void ann_set_activate_function(ANNet *net, ANNActivateFunction func, ANNLayerGroup layerGroup);
void ann_randomize_weights(ANNet *net, nnreal min, nnreal max);

void ann_dump(ANNet *net);
void ann_dump_train_sets(ANNet *net);

inline nnreal ann_random_range(nnreal min, nnreal max)
{ return (min + (((max-min) * rand())/(RAND_MAX + 1.0))); }

inline nnreal ann_convert_range(nnreal v, nnreal from_min, nnreal from_max, nnreal to_min, nnreal to_max)
{ return ((((v - from_min) / (from_max - from_min)) * (to_max - to_min)) + to_min); }

inline nnreal ann_clip(nnreal v, nnreal min, nnreal max)
{ return ((v < max) ? ((v > min) ? v : min) : max); }

#ifdef __cplusplus
};
#endif

#endif
