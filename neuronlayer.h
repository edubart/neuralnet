#ifndef NEURONLAYER_H
#define NEURONLAYER_H

#include "neuron.h"

class NeuronLayer
{
public:
    NeuronLayer(NeuronType type, int numNeurons, NeuralNetwork *net);

    void link(NeuronLayer *inputLayer);
    void feedForward();
    void reset();

    int size() const { return m_neurons.size(); }
    Neuron *neuron(int i) { return m_neurons[i]; }
    const std::vector<Neuron*>& neurons() const { return m_neurons; }

private:
    NeuralNetwork *m_net;
    std::vector<Neuron*> m_neurons;
};

#endif // NEURONLAYER_H
