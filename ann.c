#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "ann.h"

inline nnreal activate_function(nnreal value, nnreal steepness, ANNActivateFunction func)
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

inline nnreal derive_activate_function(nnreal value, nnreal steepness, ANNActivateFunction func)
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

inline void synapse_init(ANNSynapse *synapse, ANNNeuron *inputNeuron, ANNNeuron *outputNeuron)
{
    synapse->inputNeuron = inputNeuron;
    synapse->outputNeuron = outputNeuron;
    synapse->weight = ann_random_range(-0.1, 0.1);
    synapse->weightChange = 0;
}

inline void neuron_init(ANNNeuron *neuron, ANNLayer *layer) {
    neuron->layer = layer;
    neuron->delta = 0;
    neuron->bias = 0;
    neuron->biasChange = 0;
    neuron->numInputSynapses = 0;
    neuron->numOutputSynapses = 0;
    neuron->inputSynapses = NULL;
    neuron->outputSynapses = NULL;
}

inline void layer_link(ANNLayer *inputLayer, ANNLayer *outputLayer)
{
    int numInputNeurons = inputLayer->numNeurons;
    int numOutputNeurons = outputLayer->numNeurons;
    int i, j;
    ANNNeuron *inputNeuron;
    ANNNeuron *outputNeuron;
    ANNSynapse *synapse;

    inputLayer->nextLayer = outputLayer;
    outputLayer->prevLayer = inputLayer;

    for(i=0;i<numInputNeurons;++i) {
        inputNeuron = inputLayer->neurons[i];
        inputNeuron->numOutputSynapses = numOutputNeurons;
        inputNeuron->outputSynapses = malloc(numOutputNeurons * sizeof(ANNSynapse*));

        for(j=0;j<numOutputNeurons;++j) {
            outputNeuron = outputLayer->neurons[j];
            if(outputNeuron->numInputSynapses == 0) {
                outputNeuron->numInputSynapses = numInputNeurons;
                outputNeuron->inputSynapses = malloc(numInputNeurons * sizeof(ANNSynapse*));
            }

            synapse = malloc(sizeof(ANNSynapse));
            synapse_init(synapse, inputNeuron, outputNeuron);

            outputNeuron->inputSynapses[i] = synapse;
            inputNeuron->outputSynapses[j] = synapse;
        }
    }
}

inline void layer_init(ANNLayer *layer, int numNeurons, ANNLayer *prevLayer)
{
    int i;

    layer->nextLayer = NULL;
    layer->prevLayer = NULL;
    layer->numNeurons = numNeurons;
    layer->steepness = 1.0;
    layer->learningRate = 0.2;
    layer->momentum = 0.1;
    layer->activateFunc = ANN_SIGMOID_SYMMETRIC;
    layer->neurons = malloc(numNeurons * sizeof(ANNNeuron *));

    for(i=0;i<numNeurons;++i) {
        layer->neurons[i] = malloc(sizeof(ANNNeuron));
        neuron_init(layer->neurons[i], layer);

        if(prevLayer)
            layer->neurons[i]->bias = ann_random_range(-0.1, 0.1);
    }

    if(prevLayer)
        layer_link(prevLayer, layer);
}

inline void layer_set_values(ANNLayer *layer, nnreal *input)
{
    int i;
    for(i=0;i<layer->numNeurons;i++)
        layer->neurons[i]->value = input[i];
}

inline void layer_read_values(ANNLayer *layer, nnreal *output)
{
    int i;
    for(i=0;i<layer->numNeurons;i++)
        output[i] = layer->neurons[i]->value;
}

void ann_init(ANNet *net)
{
    net->inputLayer = NULL;
    net->outputLayer = NULL;
    net->trainSets = NULL;
    net->numTrainSets = 0;
    net->randSeed = time(NULL);
    srand(time(NULL));
}

void ann_add_layer(ANNet *net, int numNeurons)
{
    ANNLayer *layer;

    if(numNeurons <= 0)
        return;

    layer = malloc(sizeof(ANNLayer));
    layer_init(layer, numNeurons, net->outputLayer);

    if(!net->inputLayer)
        net->inputLayer = layer;

    net->outputLayer = layer;
}

void ann_add_train_set(ANNet *net, nnreal *input, nnreal *output)
{
    int numInputs = net->inputLayer->numNeurons;
    int numOutputs = net->outputLayer->numNeurons;
    int i;
    ANNSet *set;

    if(net->numTrainSets == 0)
        net->trainSets = (ANNSet**)malloc((net->numTrainSets+1) * sizeof(ANNSet*));
    else
        net->trainSets = (ANNSet**)realloc(net->trainSets, (net->numTrainSets+1) * sizeof(ANNSet*));

    set = (ANNSet*)malloc(sizeof(ANNSet));

    set->input = malloc(sizeof(nnreal) * numInputs);
    for(i=0;i<numInputs;++i)
        set->input[i] = input[i];

    set->output = malloc(sizeof(nnreal) * numOutputs);
    for(i=0;i<numOutputs;++i)
        set->output[i] = output[i];

    net->trainSets[net->numTrainSets++] = set;
}

int ann_load_train_sets(ANNet *net, const char *filename)
{
    nnreal in[100];
    nnreal out[100];
    int i;
    float tmp;
    FILE *fp;

    fp = fopen(filename, "r");
    if(!fp) {
        printf("ANN error: could not load dataset %s\n", filename);
        return 0;
    }

    while(!feof(fp)) {
        for(i=0;i<net->inputLayer->numNeurons;++i) {
            fscanf(fp, "%f ", &tmp);
            in[i] = tmp;
        }
        for(i=0;i<net->outputLayer->numNeurons;++i) {
            fscanf(fp, "%f ", &tmp);
            out[i] = tmp;
        }
        ann_add_train_set(net, in, out);
    }

    fclose(fp);
    return 1;
}

void ann_run(ANNet *net, nnreal *input, nnreal *output)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    nnreal value;
    int i, j;

    layer_set_values(net->inputLayer, input);

    /* feed forward */
    layer = net->inputLayer->nextLayer;
    while(layer) {
        for(i=0;i<layer->numNeurons;i++) {
            neuron = layer->neurons[i];
            value = 0;
            for(j=0;j<neuron->numInputSynapses;++j)
                value += neuron->inputSynapses[j]->inputNeuron->value * neuron->inputSynapses[j]->weight;
            value += neuron->bias * 1.0;
            value = activate_function(value, layer->steepness, layer->activateFunc);
            neuron->value = value;
        }
        layer = layer->nextLayer;
    }

    if(output)
        layer_read_values(net->outputLayer, output);
}

void ann_train_set(ANNet *net, nnreal *input, nnreal *desiredOutput)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    nnreal delta, value;
    int i, j;

    ann_run(net, input, NULL);

    /* backpropagate */
    layer = net->outputLayer;
    while(layer != net->inputLayer) {
        for(i=0;i<layer->numNeurons;i++) {
            neuron = layer->neurons[i];

            if(layer == net->outputLayer) {
                value = desiredOutput[i] - neuron->value;
                delta = value * derive_activate_function(neuron->value, layer->steepness, layer->activateFunc);
            } else {
                delta = 0;
                for(j=0;j<neuron->numOutputSynapses;++j)
                    delta += neuron->outputSynapses[j]->weight * neuron->outputSynapses[j]->outputNeuron->delta;
                delta *= derive_activate_function(neuron->value, layer->steepness, layer->activateFunc);
            }

            neuron->delta = delta;
        }
        layer = layer->prevLayer;
    }

    /* update weights */
    layer = net->outputLayer;
    while(layer != net->inputLayer) {
        for(i=0;i<layer->numNeurons;i++) {
            neuron = layer->neurons[i];

            for(j=0;j<neuron->numInputSynapses;++j) {
                synapse = neuron->inputSynapses[j];
                value = (layer->learningRate * neuron->delta * synapse->inputNeuron->value) + (layer->momentum * synapse->weightChange);
                synapse->weight += value;
                synapse->weightChange = value;
            }

            value = (layer->learningRate * neuron->delta * 1.0) + (layer->momentum * neuron->biasChange);
            neuron->bias += value;
            neuron->biasChange = value;
        }
        layer = layer->prevLayer;
    }
}

void ann_train_sets(ANNet *net)
{
    int i, index;
    ANNSet *set;

    for(i=0;i<net->numTrainSets;++i) {
        net->randSeed = net->randSeed * 1103515245 + 12345;
        index = (net->randSeed/65536) % net->numTrainSets;

        set = net->trainSets[index];
        ann_train_set(net, set->input, set->output);
    }
}

void ann_train(ANNet *net, uint maxEpochs, uint epochsBetweenReports, nnreal minimumRMSE, nnreal bitFailLimit)
{
    int i, j;
    nnreal absoluteError;
    uint epoch = 0;
    nnreal rmse = 0;
    uint bitFails = 0;
    while(1) {
        if(epoch % epochsBetweenReports == 0) {
            rmse = 0;
            bitFails = 0;
            for(i=0;i<net->numTrainSets;++i) {
                ann_run(net, net->trainSets[i]->input, NULL);
                for(j=0;j<net->outputLayer->numNeurons;++j) {
                    absoluteError = net->outputLayer->neurons[j]->value - net->trainSets[i]->output[j];
                    if(fabs(absoluteError) >= bitFailLimit)
                        bitFails++;
                    rmse += absoluteError*absoluteError;
                }
            }
            rmse = sqrt(rmse/(net->numTrainSets * net->outputLayer->numNeurons));

            printf("Epoch: %10u    Current Error: %.10f    Bit fails: %d\n", epoch, rmse, bitFails);
            fflush(stdout);
        }

        ++epoch;
        if(epoch >= maxEpochs || (rmse < minimumRMSE && bitFails == 0))
            break;

        ann_train_sets(net);
    }
}

nnreal ann_calc_set_rmse(ANNet *net, nnreal *input, nnreal *output)
{
    nnreal absoluteError;
    nnreal rmse = 0;
    int i;
    ann_run(net, input, NULL);
    for(i=0;i<net->outputLayer->numNeurons;++i) {
        absoluteError = net->outputLayer->neurons[i]->value - output[i];
        rmse += absoluteError*absoluteError;
    }
    rmse = sqrt(rmse/net->outputLayer->numNeurons);
    return rmse;
}

nnreal ann_calc_rmse(ANNet* net)
{
    nnreal absoluteError;
    nnreal rmse = 0;
    int i, j;
    for(i=0;i<net->numTrainSets;++i) {
        ann_run(net, net->trainSets[i]->input, NULL);
        for(j=0;j<net->outputLayer->numNeurons;++j) {
            absoluteError = fabs(net->outputLayer->neurons[j]->value - net->trainSets[i]->output[j]);
            rmse += absoluteError*absoluteError;
        }
    }
    rmse = sqrt(rmse/(net->numTrainSets * net->outputLayer->numNeurons));
    return rmse;
}

void ann_set_learning_rate(ANNet *net, nnreal learningRate, ANNLayerGroup layerGroup)
{
    ANNLayer *layer = net->inputLayer;
    while(layer) {
        if(layerGroup == ANN_ALL_LAYERS ||
          (layerGroup == ANN_HIDDEN_LAYERS && layer != net->inputLayer && layer != net->outputLayer) ||
          (layerGroup == ANN_OUTPUT_LAYER && layer == net->outputLayer))
            layer->learningRate = learningRate;
        layer = layer->nextLayer;
    }
}

void ann_set_momentum(ANNet* net, nnreal momentum, ANNLayerGroup layerGroup)
{
    ANNLayer *layer = net->inputLayer;
    while(layer) {
        if(layerGroup == ANN_ALL_LAYERS ||
          (layerGroup == ANN_HIDDEN_LAYERS && layer != net->inputLayer && layer != net->outputLayer) ||
          (layerGroup == ANN_OUTPUT_LAYER && layer == net->outputLayer))
            layer->momentum = momentum;
        layer = layer->nextLayer;
    }
}

void ann_set_steepness(ANNet* net, nnreal steepness, ANNLayerGroup layerGroup)
{
    ANNLayer *layer = net->inputLayer;
    while(layer) {
        if(layerGroup == ANN_ALL_LAYERS ||
          (layerGroup == ANN_HIDDEN_LAYERS && layer != net->inputLayer && layer != net->outputLayer) ||
          (layerGroup == ANN_OUTPUT_LAYER && layer == net->outputLayer))
            layer->steepness = steepness;
        layer = layer->nextLayer;
    }
}

void ann_set_activate_function(ANNet* net, ANNActivateFunction func, ANNLayerGroup layerGroup)
{
    ANNLayer *layer = net->inputLayer;
    while(layer) {
        if(layerGroup == ANN_ALL_LAYERS ||
          (layerGroup == ANN_HIDDEN_LAYERS && layer != net->inputLayer && layer != net->outputLayer) ||
          (layerGroup == ANN_OUTPUT_LAYER && layer == net->outputLayer))
            layer->activateFunc = func;
        layer = layer->nextLayer;
    }
}

void ann_randomize_weights(ANNet* net, nnreal min, nnreal max)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;

    layer = net->inputLayer->nextLayer;
    while(layer) {
        for(i=0;i<layer->numNeurons;i++) {
            neuron = layer->neurons[i];
            neuron->bias = ann_random_range(min, max);
            neuron->biasChange = 0;
            neuron->delta = 0;

            for(j=0;j<neuron->numInputSynapses;++j) {
                synapse = neuron->inputSynapses[j];
                synapse->weight = ann_random_range(min, max);
                synapse->weightChange = 0;
            }
        }
        layer = layer->nextLayer;
    }
}

void ann_dump(ANNet* net)
{
    ANNLayer *layer;
    ANNNeuron *neuron;
    ANNSynapse *synapse;
    int i, j;
    int layerCount = 0;

    layer = net->inputLayer;
    while(layer) {
        printf("layer[%d]\n", layerCount);
        for(i=0;i<layer->numNeurons;i++) {
            neuron = layer->neurons[i];
            printf("  neuron[%d] => (value = %f, delta = %f, bias = %f, biasChange = %f)\n", i, neuron->value, neuron->delta, neuron->bias, neuron->biasChange);

            for(j=0;j<neuron->numInputSynapses;++j) {
                synapse = neuron->inputSynapses[j];
                printf("    synapse[%d] => (weight = %f, weightChange = %f)\n", j, synapse->weight, synapse->weightChange);
            }
        }
        layer = layer->nextLayer;
        layerCount++;
    }
}

void ann_dump_train_sets(ANNet* net)
{
    int i, j;
    nnreal absoluteError;
    ANNSet *set;
    nnreal rmse;
    for(i=0;i<net->numTrainSets;++i) {
        rmse = 0;
        set = net->trainSets[i];
        ann_run(net, set->input, NULL);
        for(j=0;j<net->inputLayer->numNeurons;++j)
            printf("%.2f ", set->input[j]);
        printf("=> ");
        for(j=0;j<net->outputLayer->numNeurons;++j) {
            absoluteError = net->outputLayer->neurons[j]->value - set->output[j];
            printf("%.2f ", net->outputLayer->neurons[j]->value);
            rmse += absoluteError*absoluteError;
        }
        rmse = sqrt(rmse/net->outputLayer->numNeurons);
        printf("[err: %.5f]\n", rmse);
    }
}
