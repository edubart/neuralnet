#include "neuronlayer.h"
#include "neuron.h"
#include "util.h"
#include "synapse.h"

NeuronLayer::NeuronLayer(NeuronType type, int numNeurons, NeuralNetwork *net) :
    m_net(net)
{
    for(int i=0;i<numNeurons;++i) {
        Neuron *neuron = new Neuron(type, m_net);
        m_neurons.push_back(neuron);
    }
}

void NeuronLayer::link(NeuronLayer* inputLayer)
{
    foreach(Neuron *inputNeuron, inputLayer->m_neurons) {
        foreach(Neuron *outputNeuron, m_neurons) {
            Synapse *synapse = new Synapse(inputNeuron, outputNeuron, m_net);
            inputNeuron->addOutputSynapse(synapse);
            outputNeuron->addInputSynapse(synapse);
        }
    }
}

void NeuronLayer::feedForward()
{
    foreach(Neuron *neuron, m_neurons)
        neuron->feedForward();
}

void NeuronLayer::reset()
{
    foreach(Neuron *neuron, m_neurons)
        neuron->reset();
}
