#ifndef ANN_H
#define ANN_H

#include <stdlib.h>
#include <sys/timeb.h>

typedef unsigned short ushort;
typedef unsigned int uint;
typedef double annreal;

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
    ANN_INPUT_LAYER = -3
} ANNLayerGroup;

typedef enum {
    ANN_STOP_MAX_RMSE,
    ANN_STOP_NO_BITFAILS,
    ANN_DONT_STOP
} ANNStopMode;

struct ANNSynapse {
    annreal weight;
    annreal weight_delta;
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
};

struct ANNNeuron {
    annreal delta;
    annreal value;
    annreal bias;
    annreal bias_delta;
    ushort input_synapses_count;
    ushort output_synapses_count;
    ANNSynapse **input_synapses;
    ANNSynapse **output_synapses;
    ANNLayer *layer;
};

struct ANNLayer {
    ANNActivateFunction activate_func;
    annreal learning_rate;
    annreal momentum;
    annreal steepness;
    ushort neurons_count;
    ANNNeuron **neurons;
    ANNLayer *prev_layer;
    ANNLayer *next_layer;
};

struct ANNSet {
    annreal *input;
    annreal *output;
};

struct ANNet {
    ANNLayer *input_layer;
    ANNLayer *output_layer;
    ANNSet **train_sets;
    uint train_sets_count;
    uint seed;
};

#ifdef __cplusplus
extern "C" {
#endif

void ann_init(ANNet *net);

void ann_add_layer(ANNet *net, int neurons_count);
void ann_add_train_set(ANNet *net, annreal *input, annreal *output);
int ann_load_train_sets(ANNet *net, const char *filename);

void ann_run(ANNet *net, annreal *input, annreal *output);

void ann_train_set(ANNet *net, annreal *input, annreal *output);
void ann_train_sets(ANNet *net);
void ann_train(ANNet *net, annreal max_train_time, ANNStopMode stop_mode, annreal stop_param);

annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output);
annreal ann_calc_rmse(ANNet *net);

void ann_set_learning_rate(ANNet *net, annreal learning_rate, ANNLayerGroup layer_group);
void ann_set_momentum(ANNet *net, annreal momentum, ANNLayerGroup layer_group);
void ann_set_steepness(ANNet *net, annreal steepness, ANNLayerGroup layer_group);
void ann_set_activate_function(ANNet *net, ANNActivateFunction func, ANNLayerGroup layer_group);
void ann_randomize_weights(ANNet *net, annreal min, annreal max);

void ann_dump(ANNet *net);
void ann_dump_train_sets(ANNet *net);

/* utilities */
extern inline annreal ann_random_range(annreal min, annreal max)
{ return (min + (((max-min) * rand())/(RAND_MAX + 1.0))); }
extern inline annreal ann_convert_range(annreal v, annreal from_min, annreal from_max, annreal to_min, annreal to_max)
{ return ((((v - from_min) / (from_max - from_min)) * (to_max - to_min)) + to_min); }
extern inline annreal ann_clip(annreal v, annreal min, annreal max)
{ return ((v < max) ? ((v > min) ? v : min) : max); }
extern inline annreal ann_get_millis()
{ struct timeb t; ftime(&t); return (((annreal)t.millitm) + ((annreal)t.time) * 1000); }

#ifdef __cplusplus
};
#endif

#endif
