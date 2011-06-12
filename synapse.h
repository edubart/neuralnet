#ifndef SYNAPSE_H
#define SYNAPSE_H

class Neuron;
class NeuralNetwork;

class Synapse
{
public:
    Synapse(Neuron *inputNeuron, Neuron *outputNeuron, NeuralNetwork *net);

    void fire(double value);
    void learn(double delta);

    Neuron *inputNeuron() const { return m_inputNeuron; }
    Neuron *outputNeuron() const { return m_outputNeuron; }
    double weight() const { return m_weight; }

private:
    NeuralNetwork *m_net;

    Neuron *m_inputNeuron;
    Neuron *m_outputNeuron;

    double m_weight;
    double m_lastWeightChange;
};

#endif // SYNAPSE_H
