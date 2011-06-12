#include "neuron.h"
#include "synapse.h"
#include "util.h"
#include "neuralnetwork.h"

Neuron::Neuron(NeuronType type, NeuralNetwork *net) : m_net(net), m_type(type), m_value(0), m_bias(0), m_delta(0)
{
}

void Neuron::feed(double value)
{
    m_value += value;
}

void Neuron::feedForward()
{
    if(m_type != INPUT_NEURON)
        m_value = sigmoid((m_value + m_bias) * m_net->stepness());

    foreach(Synapse *synapse, m_outputSynapses)
        synapse->fire(m_value);
}

void Neuron::reset()
{
    m_value = 0;
    m_delta = 0;
}

void Neuron::learn(double desired)
{
    double delta = m_value * (desired - m_value) * (1 - m_value);
    internalLearn(delta);
}

void Neuron::learn()
{
    double delta = 0;
    foreach(Synapse *synapse, m_outputSynapses)
        delta += synapse->weight() * synapse->outputNeuron()->delta();
    delta *= m_value * (1 - m_value);

    internalLearn(delta);
}

void Neuron::internalLearn(double delta)
{
    m_bias += (m_net->learningRate() * delta * 1.0) + (m_net->momentum() * m_delta);
    m_delta = delta;
    foreach(Synapse *synapse, m_inputSynapses)
        synapse->learn(delta);
}


