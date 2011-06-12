#ifndef NEURON_H
#define NEURON_H

#include <vector>

class NeuralNetwork;
class Synapse;

enum NeuronType {
    INPUT_NEURON,
    HIDDEN_NEURON,
    OUTPUT_NEURON
};

class Neuron
{
public:
    Neuron(NeuronType type, NeuralNetwork *net);

    void feed(double value);
    void feedForward();
    void learn(double desired);
    void learn();
    void reset();

    void setBias(double bias) { m_bias = bias; }
    double bias() const { return m_bias; }

    void setValue(double value) { m_value = value; }
    double value() const { return m_value; }

    double delta() { return m_delta; }

    void addOutputSynapse(Synapse *synapse) { m_outputSynapses.push_back(synapse); }
    void addInputSynapse(Synapse *synapse) { m_inputSynapses.push_back(synapse); }


private:
    void internalLearn(double delta);

    NeuralNetwork *m_net;

    std::vector<Synapse*> m_outputSynapses;
    std::vector<Synapse*> m_inputSynapses;

    NeuronType m_type;
    double m_value;
    double m_bias;
    double m_delta;
};

#endif // NEURON_H
