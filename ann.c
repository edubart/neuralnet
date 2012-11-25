#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>

#include "ann.h"
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

    synapse->weight = ann_random_range(-1, 1);
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

    neuron->num_input_synapses = 0;
    neuron->num_output_synapses = 0;
    neuron->input_synapses = NULL;
    neuron->output_synapses = NULL;
}

inline void layer_link(ANNLayer *input_layer, ANNLayer *output_layer)
{
    ANNNeuron *input_neuron;
    ANNNeuron *output_neuron;
    ANNSynapse *synapse;
    int i, j;
    int input_num_neurons = input_layer->num_neurons;
    int output_num_neurons = output_layer->num_neurons;

    input_layer->next_layer = output_layer;
    output_layer->prev_layer = input_layer;

    /* connect layers neurons */
    for(i=0;i<input_num_neurons;++i) {
        input_neuron = input_layer->neurons[i];
        input_neuron->num_output_synapses = output_num_neurons;
        input_neuron->output_synapses = malloc(output_num_neurons * sizeof(ANNSynapse*));

        for(j=0;j<output_num_neurons;++j) {
            output_neuron = output_layer->neurons[j];
            if(output_neuron->num_input_synapses == 0) {
                output_neuron->num_input_synapses = input_num_neurons;
                output_neuron->input_synapses = malloc(input_num_neurons * sizeof(ANNSynapse*));
            }

            synapse = malloc(sizeof(ANNSynapse));
            synapse_init(synapse, input_neuron, output_neuron);

            output_neuron->input_synapses[i] = synapse;
            input_neuron->output_synapses[j] = synapse;
        }
    }
}

inline void layer_init(ANNLayer *layer, int num_neurons, ANNLayer *prev_layer)
{
    int i;

    layer->next_layer = NULL;
    layer->prev_layer = NULL;
    layer->num_neurons = num_neurons;
    layer->activate_func = ANN_SIGMOID_SYMMETRIC;
    layer->neurons = malloc(num_neurons * sizeof(ANNNeuron *));

    /* initiaze layer neurons */
    for(i=0;i<num_neurons;++i) {
        layer->neurons[i] = malloc(sizeof(ANNNeuron));
        neuron_init(layer->neurons[i], layer);

        if(prev_layer)
            layer->neurons[i]->bias = ann_random_range(-1, 1);
    }

    /* link with the previous layer */
    if(prev_layer)
        layer_link(prev_layer, layer);
}

ANNet *ann_create()
{
    ANNet *net = malloc(sizeof(ANNet));

    net->steepness = 1;
    net->num_train_sets = 0;
    net->prev_mse = 0;
    net->mse = 1e100;
    net->desired_rmse = 0;
    net->bit_fail_limit = 0.01;
    net->bit_fails = -1;
    net->stop_mode = ANN_DONT_STOP;
    net->input_layer = NULL;
    net->output_layer = NULL;
    net->train_sets = NULL;
    net->random_seed = time(NULL);
    net->num_layers = 0;
    net->layer_max_neurons = 0;

    net->train_algorithm = ANN_TRAIN_RPROP;

    net->learning_rate = 0.7;
    net->momentum = 0;

    net->rprop_increase_factor = 1.2;
    net->rprop_decrease_factor = 0.5;
    net->rprop_min_step = 1e-6;
    net->rprop_max_step = 50;
    net->report_function = ann_report;

    srand(time(NULL));

    return net;
}

void ann_destroy(ANNet* net)
{
    ANNLayer *layer;
    ANNSynapse *synapse;
    ANNNeuron *neuron;
    ANNSet *set;
    int i, j;

    /* free synapses */
    layer = net->output_layer;
    while(layer != net->input_layer) {
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];
            for(j=0;j<neuron->num_input_synapses;++j) {
                synapse = neuron->input_synapses[j];
                free(synapse);
            }
        }
        layer = layer->prev_layer;
    }

    /* free neurons */
    layer = net->output_layer;
    while(layer) {
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];
            free(neuron->input_synapses);
            free(neuron->output_synapses);
            free(neuron);
        }
        free(layer->neurons);
        layer = layer->prev_layer;
    }

    /* free layers */
    layer = net->output_layer->prev_layer;
    while(layer) {
        free(layer->next_layer);
        layer = layer->prev_layer;
    }
    free(net->input_layer);

    /* free train sets */
    for(i=0;i<net->num_train_sets;++i) {
        set = net->train_sets[i];
        free(set->input);
        free(set->output);
        free(set);
    }
    free(net->train_sets);

    /* free net */
    free(net);
}

void ann_add_layer(ANNet *net, int num_neurons)
{
    ANNLayer *layer;

    /* skip layers without neurons */
    if(num_neurons <= 0)
        return;

    layer = malloc(sizeof(ANNLayer));
    layer_init(layer, num_neurons, net->output_layer);

    /* first added layer is the input */
    if(!net->input_layer)
        net->input_layer = layer;

    /* last added layer is the output */
    net->output_layer = layer;
    net->num_layers++;
    if(num_neurons > net->layer_max_neurons)
        net->layer_max_neurons = num_neurons;
}

void ann_add_train_set(ANNet *net, annreal *input, annreal *output)
{
    ANNSet *set;
    int i;
    int input_size = net->input_layer->num_neurons;
    int output_size = net->output_layer->num_neurons;

    /* realloc train sets array */
    if(net->num_train_sets == 0)
        net->train_sets = malloc((net->num_train_sets+1) * sizeof(ANNSet*));
    else
        net->train_sets = realloc(net->train_sets, (net->num_train_sets+1) * sizeof(ANNSet*));

    set = malloc(sizeof(ANNSet));

    /* copy the input to the train set input buffer */
    set->input = malloc(sizeof(annreal) * input_size);
    for(i=0;i<input_size;++i)
        set->input[i] = input[i];

    /* copy the output to the train set output buffer */
    set->output = malloc(sizeof(annreal) * output_size);
    for(i=0;i<output_size;++i)
        set->output[i] = output[i];

    net->train_sets[net->num_train_sets++] = set;
}

int ann_load_train_sets(ANNet *net, const char *filename)
{
    int i;
    float tmp;
    FILE *fp;
    annreal *in = malloc(net->input_layer->num_neurons * sizeof(annreal));
    annreal *out = malloc(net->output_layer->num_neurons * sizeof(annreal));

    /* open dataset file */
    fp = fopen(filename, "r");
    if(!fp) {
        printf("ANN error: could not load dataset %s\n", filename);
        return 0;
    }

    while(!feof(fp)) {
        /* read inputs */
        for(i=0;i<net->input_layer->num_neurons;++i) {
            if(fscanf(fp, "%f", &tmp) != 1) {
                printf("ANN error: failed to read dataset input\n");
                fclose(fp);
                free(in);
                free(out);
                return 0;
            }
            in[i] = tmp;
        }
        /* read desired outputs */
        for(i=0;i<net->output_layer->num_neurons;++i) {
            if(fscanf(fp, "%f", &tmp) != 1) {
                printf("ANN error: failed to read dataset input\n");
                fclose(fp);
                free(in);
                free(out);
                return 0;
            }
            out[i] = tmp;
        }
        /* add the train set */
        ann_add_train_set(net, in, out);
    }

    fclose(fp);
    free(in);
    free(out);
    return 1;
}

void ann_run(ANNet *net, annreal *input, annreal *output)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    annreal value;
    int i, j;

    /* copy the input values to input layer's neurons */
    for(i=0;i<net->input_layer->num_neurons;i++)
        net->input_layer->neurons[i]->value = input[i];

    /* feed forward algorithm, loops through all layers and feed the neurons  */
    layer = net->input_layer->next_layer;
    while(layer) {
        /* loop through all layer's neurons */
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];

            /* sum values of input synapses's neurons */
            value = 0;
            for(j=0;j<neuron->num_input_synapses;++j)
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
        for(i=0;i<net->output_layer->num_neurons;i++)
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
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];

            /* calculate output layer's neurons delta */
            if(layer == net->output_layer) {
                value = desired_output[i] - neuron->value;
                delta = -value * derive_activate_function(neuron->value, net->steepness, layer->activate_func);

                /* RPROP automatically calculates MSE */
                if(net->train_algorithm == ANN_TRAIN_RPROP) {
                    if(fabs(value) >= net->bit_fail_limit)
                        net->bit_fails++;
                    net->mse += value*value;
                }
            }
            /* calculate hidden layers's neurons delta */
            else {
                delta = 0;
                for(j=0;j<neuron->num_output_synapses;++j)
                    delta += neuron->output_synapses[j]->weight * neuron->output_synapses[j]->output_neuron->delta;
                delta *= derive_activate_function(neuron->value, net->steepness, layer->activate_func);
            }

            /* update slopes for RPROP */
            if(net->train_algorithm == ANN_TRAIN_RPROP) {
                for(j=0;j<neuron->num_input_synapses;++j) {
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
    annreal slope_sign;

    slope_sign = (*slope) * (*prev_slope);

    /* iRPROP+
     * C. Igel and M. Husken, "Empirical evaluation of the improved Rprop learning algorithms," Neurocomputing, vol. 50, pp. 105–123, Jan. 2003
     */
    if(slope_sign > 0) {
        (*prev_step) = ann_min(net->rprop_increase_factor * (*prev_step), net->rprop_max_step);
        *weight_delta = -ann_sign((*slope)) * (*prev_step);
        *weight += (*weight_delta);
        *prev_slope = (*slope);
    } else if(slope_sign < 0) {
        (*prev_step) = ann_max(net->rprop_decrease_factor * (*prev_step), net->rprop_min_step);
        if(net->mse > net->prev_mse)
            *weight -= (*weight_delta);
        *prev_slope = 0;
    } else {
        *weight_delta = -ann_sign((*slope)) * (*prev_step);
        *weight += (*weight_delta);
        *prev_slope = (*slope);
    }

    *slope = 0;
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
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];
            for(j=0;j<neuron->num_input_synapses;++j) {
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
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];

            for(j=0;j<neuron->num_input_synapses;++j) {
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
    const uint num_sets = net->num_train_sets;

    if(net->train_algorithm == ANN_TRAIN_RPROP) {
        net->prev_mse = net->mse;
        net->mse = 0;
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
        net->mse = net->mse/(net->output_layer->num_neurons * num_sets);

        /* update only if needed */
        if(!(net->stop_mode == ANN_STOP_NO_BITFAILS && net->bit_fails == 0) &&
           !(net->stop_mode == ANN_STOP_DESIRED_RMSE && sqrt(net->mse) < net->desired_rmse))
            ann_rprop_update_weights(net);
    }
}

void ann_train(ANNet *net, annreal max_train_time, annreal report_interval)
{
    uint epoch;
    annreal elapsed;
    annreal last_report_time = 0;
    annreal time_now = 0;
    annreal stop_time = ann_get_seconds() + max_train_time;

    ann_calc_errors(net);

    for(epoch=1;;++epoch) {
        time_now = ann_get_seconds();
        elapsed = time_now - last_report_time;
        if((net->stop_mode == ANN_STOP_NO_BITFAILS && net->bit_fails == 0) ||
           (net->stop_mode == ANN_STOP_DESIRED_RMSE && sqrt(net->mse) < net->desired_rmse) ||
           (time_now >= stop_time && max_train_time > 0)) {
            break;
        } else if((time_now - last_report_time) >= report_interval) {
            net->report_function(net, epoch, elapsed);
            last_report_time = time_now;
        }

        ann_train_sets(net);

        /* standard EBP doesn't automatically calculates RMSE */
        if(net->train_algorithm == ANN_TRAIN_STANDARD_EBP)
            ann_calc_errors(net);
    }

    net->report_function(net, epoch, elapsed);
}

void ann_compact_run(double *input, double *output, unsigned char *netbuffer, int netbuffer_len)
{
    FILE *stream;
    double value;
    double prev_values[1024], values[1024];
    int i, j, k, prev_size;
    struct {
        unsigned short num_layers;
        unsigned short max_layer_neurons;
        double steepness;
    } mnet;
    struct {
        unsigned short num_neurons;
        unsigned char activate_function;
    } mlayer;
    struct { double bias;} mneuron;
    struct { double weight; } msynapse;
    if(!netbuffer || !input || !output)
        return;
    stream = fmemopen((void*)netbuffer, netbuffer_len, "rb");
    if(!stream)
        return;
    fread(&mnet, 1, sizeof(mnet), stream);
    fread(&mlayer, 1, sizeof(mlayer), stream);
    for(i=0;i<mlayer.num_neurons;i++)
        values[i]=input[i];
    for(i=1;i<mnet.num_layers;++i) {
        prev_size = mlayer.num_neurons;
        for(j=0;j<mlayer.num_neurons;++j)
            prev_values[j] = values[j];
        fread(&mlayer, 1, sizeof(mlayer), stream);
        for(j=0;j<mlayer.num_neurons;++j) {
            fread(&mneuron, 1, sizeof(mneuron), stream);
            value = 0;
            for(k=0;k<prev_size;++k) {
                fread(&msynapse, 1, sizeof(msynapse), stream);
                value += prev_values[k]*msynapse.weight;
            }
            value = (value + mneuron.bias) * mnet.steepness;
            if(mlayer.activate_function == 1)
                value = (1.0 / (1.0 + exp(-value)));
            else if(mlayer.activate_function == 2)
                value = (2.0 / (1.0 + exp(-value)) - 1.0);
            values[j] = value;
        }
    }
    for(i=0;i<mlayer.num_neurons;i++)
        output[i]=values[i];
    fclose(stream);
}

void ann_dump_code(ANNet* net)
{
    ANNLayer *layer;
    int i, j, k, max_size;
    unsigned char *buffer;
    FILE *stream;
    struct {
        unsigned short num_layers;
        unsigned short max_layer_neurons;
        double steepness;
    } mnet;
    struct {
        unsigned short num_neurons;
        unsigned char activate_function;
    } mlayer;
    struct { double bias;} mneuron;
    struct { double weight; } msynapse;

    max_size = sizeof(mnet) + (sizeof(mlayer) + (sizeof(mneuron)+sizeof(msynapse)*net->layer_max_neurons)*net->layer_max_neurons) * net->num_layers;
    buffer = malloc(max_size);
    if(!buffer) {
        printf("allocation failed\n");
        return;
    }

    stream = fmemopen((void*)buffer, max_size, "w");
    mnet.max_layer_neurons = net->layer_max_neurons;
    mnet.steepness = net->steepness;
    mnet.num_layers = net->num_layers;
    fwrite(&mnet, 1, sizeof(mnet), stream);
    layer=net->input_layer;
    mlayer.num_neurons = layer->num_neurons;
    mlayer.activate_function = 0;
    fwrite(&mlayer, 1, sizeof(mlayer), stream);
    for(i=1,layer=layer->next_layer;i<net->num_layers;++i,layer=layer->next_layer) {
        mlayer.activate_function = layer->activate_func;
        mlayer.num_neurons = layer->num_neurons;
        fwrite(&mlayer, 1, sizeof(mlayer), stream);
        for(j=0;j<layer->num_neurons;++j) {
            mneuron.bias = layer->neurons[j]->bias;
            fwrite(&mneuron, 1, sizeof(mneuron), stream);
            for(k=0;k<layer->prev_layer->num_neurons;++k) {
                msynapse.weight = layer->neurons[j]->input_synapses[k]->weight;
                fwrite(&msynapse, 1, sizeof(msynapse), stream);
            }
        }
    }
    max_size = ftell(stream);
    fclose(stream);

    printf("unsigned char mnet[] = \n\t\"");
    for(i=0;i<max_size;++i) {
        printf("\\x%02X",buffer[i]);
        if(i > 0 && (i+1)%32 == 0)
            printf("\"\n\t\"");
    }
    free(buffer);

    printf("\";\n");
    printf("void ann_compact_run(double *input,double *output,unsigned char *netbuffer,int netbuffer_len){\n");
    printf("FILE *a;double v;double pvs[1024],vs[1024];int i,j,k,p;\n");
    printf("struct{short nl;short mln;double e;} t;struct{short nn;char a;} ml;struct{double bias;} mn;struct {double w;} ms;\n");
    printf("if(!netbuffer||!input||!output)return;a=fmemopen((void*)netbuffer,netbuffer_len,\"rb\");if(!a)return;\n");
    printf("fread(&t,1,sizeof(t),a);fread(&ml,1,sizeof(ml),a);for(i=0;i<ml.nn;i++)vs[i]=input[i];\n");
    printf("for(i=1;i<t.nl;++i){p=ml.nn;for(j=0;j<ml.nn;++j)pvs[j]=vs[j];fread(&ml,1,sizeof(ml),a);for(j=0;j<ml.nn;++j){\n");
    printf("fread(&mn,1,sizeof(mn),a);v=0;for(k=0;k<p;++k){fread(&ms,1,sizeof(ms),a);v += pvs[k]*ms.w;}\n");
    printf("v=(v+mn.bias)*t.e;if(ml.a==1)v=(1.0/(1.0+exp(-v)));else if(ml.a==2)v=(2.0/(1.0+exp(-v))- 1.0);vs[j]=v;}}\n");
    printf("for(i=0;i<ml.nn;i++)output[i]=vs[i];fclose(a);}\n");
    fflush(stdout);
}

void ann_report(ANNet* net, uint epoch, annreal elapsed)
{
    printf("Epoch: %10u    Current RMSE: %.10f    Bit fails: %d\n", epoch, sqrt(net->mse), net->bit_fails);
    fflush(stdout);
}

void ann_calc_errors(ANNet* net)
{
    annreal error;
    annreal mse = 0;
    uint bit_fails = 0;
    int i, j;

    /* loop through all sets and calculate errors */
    for(i=0;i<net->num_train_sets;++i) {
        ann_run(net, net->train_sets[i]->input, NULL);
        for(j=0;j<net->output_layer->num_neurons;++j) {
            error = net->output_layer->neurons[j]->value - net->train_sets[i]->output[j];
            if(fabs(error) >= net->bit_fail_limit)
                bit_fails++;
            mse += error*error;
        }
    }
    mse = mse/(net->num_train_sets * net->output_layer->num_neurons);

    net->mse = mse;
    net->bit_fails = bit_fails;
}

annreal ann_calc_set_rmse(ANNet *net, annreal *input, annreal *output)
{
    annreal error;
    annreal rmse = 0;
    int i;

    /* calculate set rmse */
    ann_run(net, input, NULL);
    for(i=0;i<net->output_layer->num_neurons;++i) {
        error = net->output_layer->neurons[i]->value - output[i];
        rmse += error*error;
    }
    rmse = sqrt(rmse/net->output_layer->num_neurons);
    return rmse;
}

void ann_dump_train_sets(ANNet* net)
{
    int i, j;
    annreal error;
    ANNSet *set;
    annreal rmse;

    for(i=0;i<net->num_train_sets;++i) {
        rmse = 0;
        set = net->train_sets[i];
        ann_run(net, set->input, NULL);
        for(j=0;j<net->input_layer->num_neurons;++j)
            printf("%.2f ", set->input[j]);
        printf("=> ");
        for(j=0;j<net->output_layer->num_neurons;++j) {
            error = net->output_layer->neurons[j]->value - set->output[j];
            printf("%.2f ", net->output_layer->neurons[j]->value);
            rmse += error*error;
        }
        rmse = sqrt(rmse/net->output_layer->num_neurons);
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
        for(i=0;i<layer->num_neurons;i++) {
            neuron = layer->neurons[i];
            neuron->bias = ann_random_range(min, max);
            neuron->bias_delta = 0;

            for(j=0;j<neuron->num_input_synapses;++j) {
                synapse = neuron->input_synapses[j];
                synapse->weight = ann_random_range(min, max);
                synapse->weight_delta = 0;
            }
        }
        layer = layer->next_layer;
    }
}

void ann_set_report_function(ANNet* net, ANNReportFunction report_function)
{
    net->report_function = report_function;
}

void ann_set_training_algorithm(ANNet *net, ANNTrainAlgorithm train_algorithm)
{
    net->train_algorithm = train_algorithm;
}

void ann_set_rprop_params(ANNet* net, annreal increase_factor, annreal decrease_factor, annreal min_step, annreal max_step)
{
    net->rprop_increase_factor = increase_factor;
    net->rprop_decrease_factor = decrease_factor;
    net->rprop_min_step = min_step;
    net->rprop_max_step = max_step;
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

annreal ann_random_range(annreal min, annreal max)
{
    return (min + (((max-min) * rand())/(RAND_MAX + 1.0)));
}

annreal ann_get_seconds()
{
    static time_t firstTick = 0;
    annreal ret;

    struct timeb t;
    ftime(&t);

    if(firstTick == 0)
        firstTick = t.time;

    ret = t.time - firstTick;
    ret += ((annreal)t.millitm)/1000.0;
    return ret;
}
