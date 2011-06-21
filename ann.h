#ifndef ANN_H
#define ANN_H

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
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
    ANN_ALL_LAYERS,
    ANN_HIDDEN_LAYERS,
    ANN_OUTPUT_LAYER
} ANNLayerGroup;

typedef enum {
    ANN_STOP_DESIRED_RMSE,
    ANN_STOP_NO_BITFAILS,
    ANN_DONT_STOP
} ANNStopMode;

typedef enum {
    ANN_TRAIN_STANDARD_EBP,
    ANN_TRAIN_RPROP
} ANNTrainAlgorithm;

struct ANNSynapse {
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
    annreal weight;

    annreal weight_delta;

    annreal rprop_weight_slope;
    annreal rprop_prev_weight_step;
    annreal rprop_prev_weight_slope;
};

struct ANNNeuron {
    ANNSynapse **input_synapses;
    ANNSynapse **output_synapses;
    ushort num_input_synapses;
    ushort num_output_synapses;
    annreal value;
    annreal bias;

    annreal delta;
    annreal bias_delta;

    annreal rprop_bias_slope;
    annreal rprop_prev_bias_slope;
    annreal rprop_prev_bias_step;
};

struct ANNLayer {
    ANNLayer *prev_layer;
    ANNLayer *next_layer;
    ANNActivateFunction activate_func;
    ANNNeuron **neurons;
    ushort num_neurons;
};

struct ANNSet {
    annreal *input;
    annreal *output;
};

struct ANNet {
    ANNLayer *input_layer;
    ANNLayer *output_layer;

    ANNSet **train_sets;
    uint num_train_sets;

    uint random_seed;

    ANNStopMode stop_mode;
    annreal bit_fail_limit;
    annreal desired_rmse;
    uint bit_fails;
    annreal mse;
    annreal prev_mse;

    annreal steepness;

    ANNTrainAlgorithm train_algorithm;

    /* used by backpropagation */
    annreal learning_rate;
    annreal momentum;

    /* used by rprop */
    annreal rprop_increase_factor;
    annreal rprop_decrease_factor;
    annreal rprop_min_step;
    annreal rprop_max_step;
};

#ifdef __cplusplus
extern "C" {
#endif

ANNet *ann_create();
void ann_destroy(ANNet *net);

void ann_add_layer(ANNet *net, int num_neurons);
void ann_add_train_set(ANNet *net, annreal *input, annreal *output);
int ann_load_train_sets(ANNet *net, const char *filename);

void ann_run(ANNet *net, annreal *input, annreal *output);

void ann_train_set(ANNet *net, annreal *input, annreal *output);
void ann_train_sets(ANNet *net);
void ann_train(ANNet *net, annreal max_train_time, annreal report_interval);
void ann_report(ANNet *net, uint epoch);

void ann_calc_errors(ANNet *net);
annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output);

void ann_dump_train_sets(ANNet *net);
void ann_randomize_weights(ANNet *net, annreal min, annreal max);

void ann_set_training_algorithm(ANNet *net, ANNTrainAlgorithm train_algorithm);
void ann_set_rprop_params(ANNet *net, annreal increase_factor, annreal decrease_factor, annreal min_step, annreal max_step);
void ann_set_desired_rmse(ANNet *net, annreal desired_rmse);
void ann_set_bit_fail_limit(ANNet *net, annreal bit_fail_limit);
void ann_set_stop_mode(ANNet *net, ANNStopMode stop_mode);
void ann_set_learning_rate(ANNet *net, annreal learning_rate);
void ann_set_momentum(ANNet *net, annreal momentum);
void ann_set_steepness(ANNet *net, annreal steepness);
void ann_set_activate_function(ANNet *net, ANNActivateFunction func, ANNLayerGroup layer_group);

/* utilities */
annreal ann_random_range(annreal min, annreal max);
annreal ann_get_seconds();

#define ann_convert_range(v, from_min, from_max, to_min, to_max) ((((annreal)(v - from_min) / (annreal)(from_max - from_min)) * (annreal)(to_max - to_min)) + to_min)
#define ann_clip(v,min,max) ((v > max) ? max : ((v < min) ? min : v))
#define ann_min(a,b) (a < b ? a : b)
#define ann_max(a,b) (a > b ? a : b)
#define ann_sign(a) (a > 0 ? 1 : (a < 0 ? -1 : 0))

#ifdef __cplusplus
};
#endif

#endif
