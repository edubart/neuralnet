#include "synapse.h"
#include "util.h"
#include "neuron.h"
#include "neuralnetwork.h"

Synapse::Synapse(Neuron *inputNeuron, Neuron *outputNeuron, NeuralNetwork *net) :
    m_net(net), m_inputNeuron(inputNeuron), m_outputNeuron(outputNeuron), m_lastWeightChange(0)
{
    m_weight = m_net->generateRandomWeight();
    outputNeuron->setBias(m_net->generateRandomWeight());
}

void Synapse::fire(double value)
{
    m_outputNeuron->feed(value * m_weight);
}

void Synapse::learn(double delta)
{
    double value = m_inputNeuron->value() * delta;
    m_weight += (m_net->learningRate() * value) + (m_net->momentum() * m_lastWeightChange);
    m_lastWeightChange = value;
}
