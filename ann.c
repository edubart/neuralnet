#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "ann.h"

inline annreal activate_function(annreal value, annreal steepness, ANNActivateFunction func)
{
    value *= steepness;
    switch(func) {
        case ANN_LINEAR:
            return value;
        case ANN_SIGMOID:
            return (1.0/(1.0 + exp(-value)));
        case ANN_SIGMOID_SYMMETRIC:
            return (2.0/(1.0 + exp(-value)) - 1.0);
    }
    return 0;
}

inline annreal derive_activate_function(annreal value, annreal steepness, ANNActivateFunction func)
{
    switch(func) {
        case ANN_LINEAR:
            return steepness;
        case ANN_SIGMOID:
            value = ann_clip(value, 0, 1);
            return (steepness * value * (1.0 - value));
        case ANN_SIGMOID_SYMMETRIC:
            value = ann_clip(value, -1, 1);
            return (0.5 * steepness * (1.0 - (value*value)));
    }
    return 0;
}

inline void synapse_init(ANNSynapse *synapse, ANNNeuron *input_neuron, ANNNeuron *output_neuron)
{
    synapse->input_neuron = input_neuron;
    synapse->output_neuron = output_neuron;
    synapse->weight = ann_random_range(-0.1, 0.1);
    synapse->weight_delta = 0;
}

inline void neuron_init(ANNNeuron *neuron, ANNLayer *layer) {
    neuron->layer = layer;
    neuron->delta = 0;
    neuron->bias = 0;
    neuron->bias_delta = 0;
    neuron->input_synapses_count = 0;
    neuron->output_synapses_count = 0;
    neuron->input_synapses = NULL;
    neuron->output_synapses = NULL;
}

inline void layer_link(ANNLayer *input_layer, ANNLayer *output_layer)
{
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
    ANNSynapse *synapse;
    int i, j;
    int input_neurons_count = input_layer->neurons_count;
    int output_neurons_count = output_layer->neurons_count;

    input_layer->next_layer = output_layer;
    output_layer->prev_layer = input_layer;

    for(i=0;i<input_neurons_count;++i) {
        input_neuron = input_layer->neurons[i];
        input_neuron->output_synapses_count = output_neurons_count;
        input_neuron->output_synapses = malloc(output_neurons_count * sizeof(ANNSynapse*));

        for(j=0;j<output_neurons_count;++j) {
            output_neuron = output_layer->neurons[j];
            if(output_neuron->input_synapses_count == 0) {
                output_neuron->input_synapses_count = input_neurons_count;
                output_neuron->input_synapses = malloc(input_neurons_count * sizeof(ANNSynapse*));
            }

            synapse = malloc(sizeof(ANNSynapse));
            synapse_init(synapse, input_neuron, output_neuron);

            output_neuron->input_synapses[i] = synapse;
            input_neuron->output_synapses[j] = synapse;
        }
    }
}

inline void layer_init(ANNLayer *layer, int neurons_count, ANNLayer *prev_layer)
{
    int i;

    layer->next_layer = NULL;
    layer->prev_layer = NULL;
    layer->neurons_count = neurons_count;
    layer->steepness = 1.0;
    layer->learning_rate = 0.2;
    layer->momentum = 0.1;
    layer->activate_func = ANN_SIGMOID_SYMMETRIC;
    layer->neurons = malloc(neurons_count * sizeof(ANNNeuron *));

    for(i=0;i<neurons_count;++i) {
        layer->neurons[i] = malloc(sizeof(ANNNeuron));
        neuron_init(layer->neurons[i], layer);

        if(prev_layer)
            layer->neurons[i]->bias = ann_random_range(-0.1, 0.1);
    }

    if(prev_layer)
        layer_link(prev_layer, layer);
}

inline void layer_set_values(ANNLayer *layer, annreal *input)
{
    int i;
    for(i=0;i<layer->neurons_count;i++)
        layer->neurons[i]->value = input[i];
}

inline void layer_read_values(ANNLayer *layer, annreal *output)
{
    int i;
    for(i=0;i<layer->neurons_count;i++)
        output[i] = layer->neurons[i]->value;
}

void ann_init(ANNet *net)
{
    net->input_layer = NULL;
    net->output_layer = NULL;
    net->train_sets = NULL;
    net->train_sets_count = 0;
    net->seed = time(NULL);
    srand(time(NULL));
}

void ann_add_layer(ANNet *net, int neurons_count)
{
    ANNLayer *layer;

    if(neurons_count <= 0)
        return;

    layer = malloc(sizeof(ANNLayer));
    layer_init(layer, neurons_count, net->output_layer);

    if(!net->input_layer)
        net->input_layer = layer;

    net->output_layer = layer;
}

void ann_add_train_set(ANNet *net, annreal *input, annreal *output)
{
    ANNSet *set;
    int i;
    int input_size = net->input_layer->neurons_count;
    int output_size = net->output_layer->neurons_count;

    if(net->train_sets_count == 0)
        net->train_sets = (ANNSet**)malloc((net->train_sets_count+1) * sizeof(ANNSet*));
    else
        net->train_sets = (ANNSet**)realloc(net->train_sets, (net->train_sets_count+1) * sizeof(ANNSet*));

    set = (ANNSet*)malloc(sizeof(ANNSet));

    set->input = malloc(sizeof(annreal) * input_size);
    for(i=0;i<input_size;++i)
        set->input[i] = input[i];

    set->output = malloc(sizeof(annreal) * output_size);
    for(i=0;i<output_size;++i)
        set->output[i] = output[i];

    net->train_sets[net->train_sets_count++] = set;
}

int ann_load_train_sets(ANNet *net, const char *filename)
{
    annreal in[1000];
    annreal out[1000];
    int i;
    float tmp;
    FILE *fp;

    fp = fopen(filename, "r");
    if(!fp) {
        printf("ANN error: could not load dataset %s\n", filename);
        return 0;
    }

    while(!feof(fp)) {
        for(i=0;i<net->input_layer->neurons_count;++i) {
            fscanf(fp, "%f ", &tmp);
            in[i] = tmp;
        }
        for(i=0;i<net->output_layer->neurons_count;++i) {
            fscanf(fp, "%f ", &tmp);
            out[i] = tmp;
        }
        ann_add_train_set(net, in, out);
    }

    fclose(fp);
    return 1;
}

void ann_run(ANNet *net, annreal *input, annreal *output)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    annreal value;
    int i, j;

    layer_set_values(net->input_layer, input);

    /* feed forward */
    layer = net->input_layer->next_layer;
    while(layer) {
        for(i=0;i<layer->neurons_count;i++) {
            neuron = layer->neurons[i];
            value = 0;
            for(j=0;j<neuron->input_synapses_count;++j)
                value += neuron->input_synapses[j]->input_neuron->value * neuron->input_synapses[j]->weight;
            value += neuron->bias * 1.0;
            value = activate_function(value, layer->steepness, layer->activate_func);
            neuron->value = value;
        }
        layer = layer->next_layer;
    }

    if(output)
        layer_read_values(net->output_layer, output);
}

void ann_train_set(ANNet *net, annreal *input, annreal *desiredOutput)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    annreal delta, value;
    int i, j;

    ann_run(net, input, NULL);

    /* backpropagate */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->neurons_count;i++) {
            neuron = layer->neurons[i];

            if(layer == net->output_layer) {
                value = desiredOutput[i] - neuron->value;
                delta = value * derive_activate_function(neuron->value, layer->steepness, layer->activate_func);
            } else {
                delta = 0;
                for(j=0;j<neuron->output_synapses_count;++j)
                    delta += neuron->output_synapses[j]->weight * neuron->output_synapses[j]->output_neuron->delta;
                delta *= derive_activate_function(neuron->value, layer->steepness, layer->activate_func);
            }

            neuron->delta = delta;
        }
        layer = layer->prev_layer;
    }

    /* update weights */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->neurons_count;i++) {
            neuron = layer->neurons[i];

            for(j=0;j<neuron->input_synapses_count;++j) {
                synapse = neuron->input_synapses[j];
                value = (layer->learning_rate * neuron->delta * synapse->input_neuron->value) + (layer->momentum * synapse->weight_delta);
                synapse->weight += value;
                synapse->weight_delta = value;
            }

            value = (layer->learning_rate * neuron->delta * 1.0) + (layer->momentum * neuron->bias_delta);
            neuron->bias += value;
            neuron->bias_delta = value;
        }
        layer = layer->prev_layer;
    }
}

void ann_train_sets(ANNet *net)
{
    ANNSet *set;
    uint i, index;
    const uint sets_count = net->train_sets_count;

    /* random shuffle */
    for(i=0;i<sets_count-1;++i) {
        net->seed = (net->seed *1103515245)+12345;
        index = i+(net->seed%(sets_count-i));
        set = net->train_sets[i];
        net->train_sets[i] = net->train_sets[index];
        net->train_sets[index] = set;
    }

    for(i=0;i<sets_count;++i) {
        set = net->train_sets[i];
        ann_train_set(net, set->input, set->output);
    }
}

void ann_train(ANNet *net, annreal max_train_time, ANNStopMode stop_mode, annreal stop_param)
{
    int i, j;
    int must_stop = 0;
    uint epoch = 1;
    uint bit_fails = 0;
    annreal absoluteError;
    annreal rmse = 0;
    annreal bit_fail_limit = 0.035;
    annreal last_report_time = 0;
    annreal time_now;
    annreal stop_time = ann_get_millis() + (max_train_time * 1000);

    if(stop_mode == ANN_STOP_NO_BITFAILS)
        bit_fail_limit = stop_param;

    while(1) {
        rmse = 0;
        bit_fails = 0;
        for(i=0;i<net->train_sets_count;++i) {
            ann_run(net, net->train_sets[i]->input, NULL);
            for(j=0;j<net->output_layer->neurons_count;++j) {
                absoluteError = net->output_layer->neurons[j]->value - net->train_sets[i]->output[j];
                if(fabs(absoluteError) >= bit_fail_limit)
                    bit_fails++;
                rmse += absoluteError*absoluteError;
            }
        }
        rmse = sqrt(rmse/(net->train_sets_count * net->output_layer->neurons_count));

        time_now = ann_get_millis();

        if(stop_mode == ANN_STOP_NO_BITFAILS && bit_fails == 0)
            must_stop = 1;
        else if(stop_mode == ANN_STOP_MAX_RMSE && rmse < stop_param)
            must_stop = 1;
        else if(time_now >= stop_time && stop_time > 0)
            must_stop = 1;

        if(time_now - last_report_time >= 500 || must_stop) {
            printf("Epoch: %10u    Current RMSE: %.10f    Bit fails: %d\n", epoch, rmse, bit_fails);
            fflush(stdout);
            last_report_time = time_now;
        }

        if(must_stop)
            break;

        ++epoch;
        ann_train_sets(net);
    }
}

annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output)
{
    annreal absoluteError;
    annreal rmse = 0;
    int i;

    ann_run(net, input, NULL);
    for(i=0;i<net->output_layer->neurons_count;++i) {
        absoluteError = net->output_layer->neurons[i]->value - output[i];
        rmse += absoluteError*absoluteError;
    }
    rmse = sqrt(rmse/net->output_layer->neurons_count);
    return rmse;
}

annreal ann_calc_rmse(ANNet* net)
{
    annreal absoluteError;
    annreal rmse = 0;
    int i, j;

    for(i=0;i<net->train_sets_count;++i) {
        ann_run(net, net->train_sets[i]->input, NULL);
        for(j=0;j<net->output_layer->neurons_count;++j) {
            absoluteError = fabs(net->output_layer->neurons[j]->value - net->train_sets[i]->output[j]);
            rmse += absoluteError*absoluteError;
        }
    }
    rmse = sqrt(rmse/(net->train_sets_count * net->output_layer->neurons_count));
    return rmse;
}

void ann_set_learning_rate(ANNet *net, annreal learning_rate, ANNLayerGroup layer_group)
{
    ANNLayer *layer = net->input_layer;
    while(layer) {
        if(layer_group == ANN_ALL_LAYERS ||
          (layer_group == ANN_HIDDEN_LAYERS && layer != net->input_layer && layer != net->output_layer) ||
          (layer_group == ANN_OUTPUT_LAYER && layer == net->output_layer))
            layer->learning_rate = learning_rate;
        layer = layer->next_layer;
    }
}

void ann_set_momentum(ANNet* net, annreal momentum, ANNLayerGroup layer_group)
{
    ANNLayer *layer = net->input_layer;
    while(layer) {
        if(layer_group == ANN_ALL_LAYERS ||
          (layer_group == ANN_HIDDEN_LAYERS && layer != net->input_layer && layer != net->output_layer) ||
          (layer_group == ANN_OUTPUT_LAYER && layer == net->output_layer))
            layer->momentum = momentum;
        layer = layer->next_layer;
    }
}

void ann_set_steepness(ANNet* net, annreal steepness, ANNLayerGroup layer_group)
{
    ANNLayer *layer = net->input_layer;
    while(layer) {
        if(layer_group == ANN_ALL_LAYERS ||
          (layer_group == ANN_HIDDEN_LAYERS && layer != net->input_layer && layer != net->output_layer) ||
          (layer_group == ANN_OUTPUT_LAYER && layer == net->output_layer))
            layer->steepness = steepness;
        layer = layer->next_layer;
    }
}

void ann_set_activate_function(ANNet* net, ANNActivateFunction func, ANNLayerGroup layer_group)
{
    ANNLayer *layer = net->input_layer;
    while(layer) {
        if(layer_group == ANN_ALL_LAYERS ||
          (layer_group == ANN_HIDDEN_LAYERS && layer != net->input_layer && layer != net->output_layer) ||
          (layer_group == ANN_OUTPUT_LAYER && layer == net->output_layer))
            layer->activate_func = func;
        layer = layer->next_layer;
    }
}

void ann_randomize_weights(ANNet* net, annreal min, annreal max)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;

    layer = net->input_layer->next_layer;
    while(layer) {
        for(i=0;i<layer->neurons_count;i++) {
            neuron = layer->neurons[i];
            neuron->bias = ann_random_range(min, max);
            neuron->bias_delta = 0;
            neuron->delta = 0;

            for(j=0;j<neuron->input_synapses_count;++j) {
                synapse = neuron->input_synapses[j];
                synapse->weight = ann_random_range(min, max);
                synapse->weight_delta = 0;
            }
        }
        layer = layer->next_layer;
    }
}

void ann_dump(ANNet* net)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;
    int layerCount = 0;

    layer = net->input_layer;
    while(layer) {
        printf("layer[%d]\n", layerCount);
        for(i=0;i<layer->neurons_count;i++) {
            neuron = layer->neurons[i];
            printf("  neuron[%d] => (value = %f, delta = %f, bias = %f, bias_delta = %f)\n", i, neuron->value, neuron->delta, neuron->bias, neuron->bias_delta);

            for(j=0;j<neuron->input_synapses_count;++j) {
                synapse = neuron->input_synapses[j];
                printf("    synapse[%d] => (weight = %f, weight_delta = %f)\n", j, synapse->weight, synapse->weight_delta);
            }
        }
        layer = layer->next_layer;
        layerCount++;
    }
}

void ann_dump_train_sets(ANNet* net)
{
    int i, j;
    annreal absoluteError;
    ANNSet *set;
    annreal rmse;

    for(i=0;i<net->train_sets_count;++i) {
        rmse = 0;
        set = net->train_sets[i];
        ann_run(net, set->input, NULL);
        for(j=0;j<net->input_layer->neurons_count;++j)
            printf("%.2f ", set->input[j]);
        printf("=> ");
        for(j=0;j<net->output_layer->neurons_count;++j) {
            absoluteError = net->output_layer->neurons[j]->value - set->output[j];
            printf("%.2f ", net->output_layer->neurons[j]->value);
            rmse += absoluteError*absoluteError;
        }
        rmse = sqrt(rmse/net->output_layer->neurons_count);
        printf("[err: %.5f]\n", rmse);
    }
}
