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

    synapse->rprop_weight_slope = 0;
    synapse->rprop_prev_weight_slope = 0;
    synapse->rprop_prev_weight_step = 0.1;
}

inline void neuron_init(ANNNeuron *neuron, ANNLayer *layer)
{
    neuron->delta = 0;

    neuron->bias = 0;
    neuron->bias_delta = 0;

    neuron->rprop_bias_slope = 0;
    neuron->rprop_prev_bias_slope = 0;
    neuron->rprop_prev_bias_step = 0.1;

    neuron->input_synapses_num = 0;
    neuron->output_synapses_num = 0;
    neuron->input_synapses = NULL;
    neuron->output_synapses = NULL;
}

inline void layer_link(ANNLayer *input_layer, ANNLayer *output_layer)
{
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
    ANNSynapse *synapse;
    int i, j;
    int input_neurons_num = input_layer->neurons_num;
    int output_neurons_num = output_layer->neurons_num;

    input_layer->next_layer = output_layer;
    output_layer->prev_layer = input_layer;

    /* connect layers neurons */
    for(i=0;i<input_neurons_num;++i) {
        input_neuron = input_layer->neurons[i];
        input_neuron->output_synapses_num = output_neurons_num;
        input_neuron->output_synapses = malloc(output_neurons_num * sizeof(ANNSynapse*));

        for(j=0;j<output_neurons_num;++j) {
            output_neuron = output_layer->neurons[j];
            if(output_neuron->input_synapses_num == 0) {
                output_neuron->input_synapses_num = input_neurons_num;
                output_neuron->input_synapses = malloc(input_neurons_num * sizeof(ANNSynapse*));
            }

            synapse = malloc(sizeof(ANNSynapse));
            synapse_init(synapse, input_neuron, output_neuron);

            output_neuron->input_synapses[i] = synapse;
            input_neuron->output_synapses[j] = synapse;
        }
    }
}

inline void layer_init(ANNLayer *layer, int neurons_num, ANNLayer *prev_layer)
{
    int i;

    layer->next_layer = NULL;
    layer->prev_layer = NULL;
    layer->neurons_num = neurons_num;
    layer->activate_func = ANN_SIGMOID_SYMMETRIC;
    layer->neurons = malloc(neurons_num * sizeof(ANNNeuron *));

    /* initiaze layer neurons */
    for(i=0;i<neurons_num;++i) {
        layer->neurons[i] = malloc(sizeof(ANNNeuron));
        neuron_init(layer->neurons[i], layer);

        if(prev_layer)
            layer->neurons[i]->bias = ann_random_range(-0.1, 0.1);
    }

    /* link with the previous layer */
    if(prev_layer)
        layer_link(prev_layer, layer);
}

void ann_init(ANNet *net)
{
    net->steepness = 1;
    net->train_num_sets = 0;
    net->rmse = -1;
    net->desired_rmse = 0;
    net->bit_fail_limit = 0.01;
    net->bit_fails = -1;
    net->stop_mode = ANN_DONT_STOP;
    net->input_layer = NULL;
    net->output_layer = NULL;
    net->train_sets = NULL;
    net->random_seed = time(NULL);

    net->train_algorithm = ANN_TRAIN_RPROP;

    net->learning_rate = 0.7;
    net->momentum = 0;

    net->rprop_increase_factor = 1.2;
    net->rprop_decrease_factor = 0.5;
    net->rprop_min_step = 1e-6;
    net->rprop_max_step = 50;
}

void ann_add_layer(ANNet *net, int neurons_num)
{
    ANNLayer *layer;

    /* skip layers without neurons */
    if(neurons_num <= 0)
        return;

    layer = malloc(sizeof(ANNLayer));
    layer_init(layer, neurons_num, net->output_layer);

    /* first added layer is the input */
    if(!net->input_layer)
        net->input_layer = layer;

    /* last added layer is the output */
    net->output_layer = layer;
}

void ann_add_train_set(ANNet *net, annreal *input, annreal *output)
{
    ANNSet *set;
    int i;
    int input_size = net->input_layer->neurons_num;
    int output_size = net->output_layer->neurons_num;

    /* realloc train sets array */
    if(net->train_num_sets == 0)
        net->train_sets = (ANNSet**)malloc((net->train_num_sets+1) * sizeof(ANNSet*));
    else
        net->train_sets = (ANNSet**)realloc(net->train_sets, (net->train_num_sets+1) * sizeof(ANNSet*));

    set = (ANNSet*)malloc(sizeof(ANNSet));

    /* copy the input to the train set input buffer */
    set->input = malloc(sizeof(annreal) * input_size);
    for(i=0;i<input_size;++i)
        set->input[i] = input[i];

    /* copy the output to the train set output buffer */
    set->output = malloc(sizeof(annreal) * output_size);
    for(i=0;i<output_size;++i)
        set->output[i] = output[i];

    net->train_sets[net->train_num_sets++] = set;
}

int ann_load_train_sets(ANNet *net, const char *filename)
{
    annreal in[1000];
    annreal out[1000];
    int i;
    float tmp;
    FILE *fp;

    /* open dataset file */
    fp = fopen(filename, "r");
    if(!fp) {
        printf("ANN error: could not load dataset %s\n", filename);
        return 0;
    }

    while(!feof(fp)) {
        /* read inputs */
        for(i=0;i<net->input_layer->neurons_num;++i) {
            fscanf(fp, "%f ", &tmp);
            in[i] = tmp;
        }
        /* read desired outputs */
        for(i=0;i<net->output_layer->neurons_num;++i) {
            fscanf(fp, "%f ", &tmp);
            out[i] = tmp;
        }
        /* add the train set */
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

    /* copy the input values to input layer's neurons */
    for(i=0;i<net->input_layer->neurons_num;i++)
        net->input_layer->neurons[i]->value = input[i];

    /* feed forward algorithm, loops through all layers and feed the neurons  */
    layer = net->input_layer->next_layer;
    while(layer) {
        /* loop through all layer's neurons */
        for(i=0;i<layer->neurons_num;i++) {
            neuron = layer->neurons[i];

            /* sum values of input synapses's neurons */
            value = 0;
            for(j=0;j<neuron->input_synapses_num;++j)
                value += neuron->input_synapses[j]->input_neuron->value * neuron->input_synapses[j]->weight;

            /* sum the bias */
            value += neuron->bias * 1.0;

            /* apply the activate function */
            value = activate_function(value, net->steepness, layer->activate_func);

            /* update neuron value */
            neuron->value = value;
        }
        layer = layer->next_layer;
    }

    /* copy the output values if needed */
    if(output) {
        for(i=0;i<net->output_layer->neurons_num;i++)
            output[i] = net->output_layer->neurons[i]->value;
    }
}

inline void ann_backpropagate_mse(ANNet* net, annreal* desired_output)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    annreal delta, value;
    int i, j;

    /* backpropagate algorithm step 1, reverse loops through all layers and calculate neurons's deltas */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->neurons_num;i++) {
            neuron = layer->neurons[i];

            /* calculate output layer's neurons delta */
            if(layer == net->output_layer) {
                value = desired_output[i] - neuron->value;
                delta = -value * derive_activate_function(neuron->value, net->steepness, layer->activate_func);

                /* RPROP automatically calculates RMSE */
                if(net->train_algorithm == ANN_TRAIN_RPROP) {
                    if(fabs(value) >= net->bit_fail_limit)
                        net->bit_fails++;
                    net->rmse += value*value;
                }
            }
            /* calculate hidden layers's neurons delta */
            else {
                delta = 0;
                for(j=0;j<neuron->output_synapses_num;++j)
                    delta += neuron->output_synapses[j]->weight * neuron->output_synapses[j]->output_neuron->delta;
                delta *= derive_activate_function(neuron->value, net->steepness, layer->activate_func);
            }

            /* update slopes for RPROP */
            if(net->train_algorithm == ANN_TRAIN_RPROP) {
                for(j=0;j<neuron->input_synapses_num;++j) {
                    synapse = neuron->input_synapses[j];
                    synapse->rprop_weight_slope += synapse->input_neuron->value * delta;
                }
                neuron->rprop_bias_slope += 1 * delta;
            }

            /* set neuron delta */
            neuron->delta = delta;
        }
        layer = layer->prev_layer;
    }
}

void ann_rprop_update_weight(ANNet *net, annreal *slope, annreal *prev_slope, annreal *prev_step, annreal *weight, annreal *weight_delta)
{
    annreal slope_sign, step;

    slope_sign = (*slope) * (*prev_slope);

    if(slope_sign > 0) {
        step = ann_min((*prev_step) * net->rprop_increase_factor, net->rprop_max_step);
        *weight_delta = -ann_sign((*slope)) * step;
        *weight += (*weight_delta);
    } else if(slope_sign < 0) {
        step = ann_max((*prev_step) * net->rprop_decrease_factor, net->rprop_min_step);
        *weight -= (*weight_delta);
        *slope = 0;
    } else {
        step = (*prev_step);
        *weight_delta = -ann_sign((*slope)) * step;
        *weight += (*weight_delta);
    }

    *prev_slope = (*slope);
    *slope = 0;
    *prev_step = step;
}

inline void ann_rprop_update_weights(ANNet *net)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;

    /* RPROP algorithmn */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->neurons_num;i++) {
            neuron = layer->neurons[i];
            for(j=0;j<neuron->input_synapses_num;++j) {
                synapse = neuron->input_synapses[j];
                ann_rprop_update_weight(net,
                                        &synapse->rprop_weight_slope, &synapse->rprop_prev_weight_slope,
                                        &synapse->rprop_prev_weight_step,
                                        &synapse->weight, &synapse->weight_delta);
            }

            ann_rprop_update_weight(net,
                                    &neuron->rprop_bias_slope, &neuron->rprop_prev_bias_slope,
                                    &neuron->rprop_prev_bias_step,
                                    &neuron->bias, &neuron->bias_delta);
        }
        layer = layer->prev_layer;
    }
}

inline void ann_ebp_update_weight(ANNet *net, annreal delta, annreal value, annreal *weight, annreal *weight_delta)
{
    delta = (-net->learning_rate * delta * value) + (net->momentum * (*weight_delta));
    *weight += delta;
    *weight_delta = delta;
}

inline void ann_ebp_update_weights(ANNet *net, annreal *desired_output)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;

    /* standard error-backpropagation (EBP) algorithm */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->neurons_num;i++) {
            neuron = layer->neurons[i];

            for(j=0;j<neuron->input_synapses_num;++j) {
                synapse = neuron->input_synapses[j];

                /* calculate synapse's weight delta */
                ann_ebp_update_weight(net, neuron->delta, synapse->input_neuron->value, &synapse->weight, &synapse->weight_delta);
            }

            /* calculate neuron's bias delta */
            ann_ebp_update_weight(net, neuron->delta, 1, &neuron->bias, &neuron->bias_delta);
        }
        layer = layer->prev_layer;
    }
}

void ann_train_set(ANNet* net, annreal* input, annreal* output)
{
    ann_run(net, input, NULL);
    ann_backpropagate_mse(net, output);

    if(net->train_algorithm == ANN_TRAIN_STANDARD_EBP)
        ann_ebp_update_weights(net, output);
}

void ann_train_sets(ANNet *net)
{
    ANNSet *set;
    uint i, index;
    const uint num_sets = net->train_num_sets;

    if(net->train_algorithm == ANN_TRAIN_RPROP) {
        net->rmse = 0;
        net->bit_fails = 0;
    }
    /* random shuffle train sets*/
    else if(net->train_algorithm == ANN_TRAIN_STANDARD_EBP) {
        for(i=0;i<num_sets-1;++i) {
            /* generate our own random, it's faster than std's one */
            net->random_seed = (net->random_seed * 1103515245) + 12345;
            index = i + (net->random_seed % (num_sets-i));

            /* swap train sets */
            set = net->train_sets[i];
            net->train_sets[i] = net->train_sets[index];
            net->train_sets[index] = set;
        }
    }

    /* train each set */
    for(i=0;i<num_sets;++i) {
        set = net->train_sets[i];
        ann_train_set(net, set->input, set->output);
    }

    if(net->train_algorithm == ANN_TRAIN_RPROP) {
        net->rmse = sqrt(net->rmse/(net->output_layer->neurons_num * num_sets));

        /* update only if needed */
        if(!(net->stop_mode == ANN_STOP_NO_BITFAILS && net->bit_fails == 0) &&
           !(net->stop_mode == ANN_STOP_DESIRED_RMSE && net->rmse < net->desired_rmse))
            ann_rprop_update_weights(net);
    }
}

void ann_train(ANNet *net, annreal max_train_time, annreal report_interval)
{
    uint epoch;
    annreal last_report_time = 0;
    annreal time_now = 0;
    annreal stop_time = ann_get_seconds() + max_train_time;

    ann_calc_errors(net);

    for(epoch=1;;++epoch) {
        time_now = ann_get_seconds();
        if((net->stop_mode == ANN_STOP_NO_BITFAILS && net->bit_fails == 0) ||
           (net->stop_mode == ANN_STOP_DESIRED_RMSE && net->rmse < net->desired_rmse) ||
           (time_now >= stop_time && max_train_time > 0)) {
            break;
        } else if((time_now - last_report_time) >= report_interval) {
            ann_report(net, epoch);
            last_report_time = time_now;
        }

        ann_train_sets(net);

        /* standard EBP doesn't automatically calculates RMSE */
        if(net->train_algorithm == ANN_TRAIN_STANDARD_EBP)
            ann_calc_errors(net);
    }

    ann_report(net, epoch);
}

void ann_report(ANNet* net, uint epoch)
{
    printf("Epoch: %10u    Current RMSE: %.10f    Bit fails: %d\n", epoch, net->rmse, net->bit_fails);
    fflush(stdout);
}

void ann_calc_errors(ANNet* net)
{
    annreal error;
    annreal rmse = 0;
    uint bit_fails = 0;
    int i, j;

    /* loop through all sets and calculate errors */
    for(i=0;i<net->train_num_sets;++i) {
        ann_run(net, net->train_sets[i]->input, NULL);
        for(j=0;j<net->output_layer->neurons_num;++j) {
            error = net->output_layer->neurons[j]->value - net->train_sets[i]->output[j];
            if(fabs(error) >= net->bit_fail_limit)
                bit_fails++;
            rmse += error*error;
        }
    }
    rmse = sqrt(rmse/(net->train_num_sets * net->output_layer->neurons_num));

    net->rmse = rmse;
    net->bit_fails = bit_fails;
}

annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output)
{
    annreal error;
    annreal rmse = 0;
    int i;

    /* calculate set rmse */
    ann_run(net, input, NULL);
    for(i=0;i<net->output_layer->neurons_num;++i) {
        error = net->output_layer->neurons[i]->value - output[i];
        rmse += error*error;
    }
    rmse = sqrt(rmse/net->output_layer->neurons_num);
    return rmse;
}

void ann_dump_train_sets(ANNet* net)
{
    int i, j;
    annreal error;
    ANNSet *set;
    annreal rmse;

    for(i=0;i<net->train_num_sets;++i) {
        rmse = 0;
        set = net->train_sets[i];
        ann_run(net, set->input, NULL);
        for(j=0;j<net->input_layer->neurons_num;++j)
            printf("%.2f ", set->input[j]);
        printf("=> ");
        for(j=0;j<net->output_layer->neurons_num;++j) {
            error = net->output_layer->neurons[j]->value - set->output[j];
            printf("%.2f ", net->output_layer->neurons[j]->value);
            rmse += error*error;
        }
        rmse = sqrt(rmse/net->output_layer->neurons_num);
        printf("[err: %.5f]\n", rmse);
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
        for(i=0;i<layer->neurons_num;i++) {
            neuron = layer->neurons[i];
            neuron->bias = ann_random_range(min, max);
            neuron->bias_delta = 0;

            for(j=0;j<neuron->input_synapses_num;++j) {
                synapse = neuron->input_synapses[j];
                synapse->weight = ann_random_range(min, max);
                synapse->weight_delta = 0;
            }
        }
        layer = layer->next_layer;
    }
}


void ann_set_training_algorithm(ANNet *net, ANNTrainAlgorithm train_algorithm)
{
    net->train_algorithm = train_algorithm;
}

void ann_set_desired_rmse(ANNet* net, annreal desired_rmse)
{
    net->desired_rmse = desired_rmse;
}

void ann_set_bit_fail_limit(ANNet* net, annreal bit_fail_limit)
{
    net->bit_fail_limit = bit_fail_limit;
}

void ann_set_stop_mode(ANNet* net, ANNStopMode stop_mode)
{
    net->stop_mode = stop_mode;
}

void ann_set_learning_rate(ANNet *net, annreal learning_rate)
{
    net->learning_rate = learning_rate;
}

void ann_set_momentum(ANNet* net, annreal momentum)
{
    net->momentum = momentum;
}

void ann_set_steepness(ANNet* net, annreal steepness)
{
    net->steepness = steepness;
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
