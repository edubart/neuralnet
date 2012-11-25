#ifndef ANN_H
#define ANN_H

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef double annreal;

struct ANNSynapse_s;
struct ANNNeuron_s;
struct ANNLayer_s;
struct ANNSet_s;
struct ANNet_s;

typedef struct ANNSynapse_s ANNSynapse;
typedef struct ANNNeuron_s ANNNeuron;
typedef struct ANNLayer_s ANNLayer;
typedef struct ANNSet_s ANNSet;
typedef struct ANNet_s ANNet;

typedef void (*ANNReportFunction)(ANNet *, uint, annreal);

typedef enum ANNActivateFunction_e {
    ANN_LINEAR,
    ANN_SIGMOID,
    ANN_SIGMOID_SYMMETRIC
} ANNActivateFunction;

typedef enum ANNLayerGroup_e {
    ANN_ALL_LAYERS,
    ANN_HIDDEN_LAYERS,
    ANN_OUTPUT_LAYER
} ANNLayerGroup;

typedef enum ANNStopMode_e {
    ANN_STOP_DESIRED_RMSE,
    ANN_STOP_NO_BITFAILS,
    ANN_DONT_STOP
} ANNStopMode;

typedef enum ANNTrainAlgorithm_e {
    ANN_TRAIN_STANDARD_EBP,
    ANN_TRAIN_RPROP
} ANNTrainAlgorithm;

struct ANNSynapse_s {
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
    annreal weight;

    annreal weight_delta;

    annreal rprop_weight_slope;
    annreal rprop_prev_weight_step;
    annreal rprop_prev_weight_slope;
};

struct ANNNeuron_s {
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

struct ANNLayer_s {
    ANNLayer *prev_layer;
    ANNLayer *next_layer;
    ANNActivateFunction activate_func;
    ANNNeuron **neurons;
    ushort num_neurons;
};

struct ANNSet_s {
    annreal *input;
    annreal *output;
};

struct ANNet_s {
    ANNLayer *input_layer;
    ANNLayer *output_layer;
    int num_layers;
    int layer_max_neurons;

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

    ANNReportFunction report_function;
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

void ann_dump_code(ANNet *net);
void ann_report(ANNet *net, uint epoch, annreal elapsed);

void ann_calc_errors(ANNet *net);
annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output);

void ann_dump_train_sets(ANNet *net);
void ann_randomize_weights(ANNet *net, annreal min, annreal max);

void ann_set_report_function(ANNet *net, ANNReportFunction report_function);
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

#ifdef __cplusplus
/* cpp binding */
class ANNetwork
{
public:
    ANNetwork() { net = ann_create(); }
    ~ANNetwork() { ann_destroy(net); }

    void addLayer(int numNeurons) { ann_add_layer(net, numNeurons); }
    void addTrainSet(annreal *input, annreal *output) { ann_add_train_set(net, input, output); }
    void loadTrainSets(const char *filename) { ann_load_train_sets(net, filename); }

    void run(annreal *input, annreal *output) { ann_run(net, input, output); }

    void trainSet(annreal *input, annreal *output) { ann_train_set(net, input, output); }
    void trainSets() { ann_train_sets(net); }
    void train(annreal maxTrainTime = 0, annreal reportInterval = 1) { ann_train(net, maxTrainTime, reportInterval); }

    void dumpCode() { ann_dump_code(net); }
    void report(uint epoch, annreal elapsed) { ann_report(net, epoch, elapsed); }

    void calcErrors() { ann_calc_errors(net); }
    annreal calcSetRMSE(annreal *input, annreal *output) { return ann_calc_set_rmse(net, input, output); }

    void randomizeWeights(annreal min = -1, annreal max = 1) { ann_randomize_weights(net, min, max); }

    void setReportFunction(ANNReportFunction reportFunction) { ann_set_report_function(net, reportFunction); }
    void setTrainingAlgorithm(ANNTrainAlgorithm trainAlgorithm) { ann_set_training_algorithm(net, trainAlgorithm); }
    void setRpropParams(annreal increaseFactor, annreal decreaseFactor, annreal minStep, annreal maxStep) { ann_set_rprop_params(net, increaseFactor, decreaseFactor, minStep, maxStep); }
    void setDesiredRMSE(annreal desiredRMSE) { ann_set_desired_rmse(net, desiredRMSE); }
    void setBitFailLimit(annreal bitFailLimit) { ann_set_bit_fail_limit(net, bitFailLimit); }
    void setStopMode(ANNStopMode stopMode) { ann_set_stop_mode(net, stopMode); }
    void setLearningRate(annreal learningRate) { ann_set_learning_rate(net, learningRate); }
    void setMomentum(annreal momentum) { ann_set_momentum(net, momentum); }
    void setSteepness(annreal steepness) { ann_set_steepness(net, steepness); }
    void setActivateFunction(ANNActivateFunction func, ANNLayerGroup layerGroup) { ann_set_activate_function(net, func, layerGroup); }

    static annreal randomRange(annreal min, annreal max) { return ann_random_range(min, max); }
    static annreal getSeconds() { return ann_get_seconds(); }

private:
    ANNet *net;
};
#endif

#endif
