#include "neuralnetwork.h"
#include "neuronlayer.h"
#include "neuron.h"
#include "util.h"
#include "synapse.h"

NeuralNetwork::NeuralNetwork()
{
    seedRandom();
    m_numInputs = 1;
    m_numOutputs = 1;
    m_learningRate = 0.7;
    m_momentum = 0.1;
    m_minRandomWeight = -0.2;
    m_maxRandomWeight = 0.2;
    m_stepness = 0.5;
    m_maxIterations = 10000;
    m_reportRate = 1000;
    m_stopFunc = STOP_NEVER;
    m_stopParam = 0;
}

void NeuralNetwork::createLayers(const std::vector< int >& hiddenLayers)
{
    m_inputLayer = new NeuronLayer(INPUT_NEURON, m_numInputs, this);
    m_layers.push_back(m_inputLayer);

    for(int i=0;i<(int)hiddenLayers.size(); ++i) {
        NeuronLayer *layer = new NeuronLayer(HIDDEN_NEURON, hiddenLayers[i], this);
        layer->link(m_layers.back());
        m_layers.push_back(layer);
    }

    m_outputLayer = new NeuronLayer(OUTPUT_NEURON, m_numOutputs, this);
    m_outputLayer->link(m_layers.back());
    m_layers.push_back(m_outputLayer);
}

void NeuralNetwork::setIO(int numInputs, int numOutputs)
{
    m_numInputs = numInputs;
    m_numOutputs = numOutputs;
}

std::vector<double> NeuralNetwork::execute(const std::vector<double>& input)
{
    std::vector<double> output;

    if((int)input.size() != m_numInputs) {
        std::cout << "input size doesnt match" << std::endl;
        return output;
    }

    foreach(NeuronLayer *layer, m_layers)
        layer->reset();

    for(int i=0;i<(int)input.size();++i)
        m_inputLayer->neuron(i)->feed(input[i]);

    foreach(NeuronLayer *layer, m_layers)
        layer->feedForward();

    foreach(Neuron *neuron, m_outputLayer->neurons())
        output.push_back(neuron->value());

    return output;
}

double NeuralNetwork::test(const std::vector< double >& input, const std::vector< double >& output)
{
    std::vector<double> processedOutput = execute(input);
    double mse = 0;

    for(int i=0;i<m_numOutputs;++i) {
        double err = processedOutput[i] - output[i];
        mse += err*err;
    }

    return mse;
}

void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& output)
{
    execute(input);

    for(auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        NeuronLayer *layer = (*it);

        if(layer == m_inputLayer)
            break;

        for(int i=0;i<layer->size();++i) {
            Neuron *neuron = layer->neuron(i);
            double desired = output[i];
            if(layer == m_outputLayer)
                neuron->learn(desired);
            else
                neuron->learn();
        }
    }
}

void NeuralNetwork::train()
{
    if(m_maxIterations > 0) {
        std::cout << "training for " << m_maxIterations << " iterations";
    }

    if(m_stopFunc == STOP_MSE) {
        std::cout << " stoping with mse " << m_stopParam;
    }

    std::cout << std::endl;
    unsigned int it = 0;
    while(it < m_maxIterations || m_maxIterations == 0) {
        foreach(const TrainSet& set, m_trainSets) {
            train(set.first, set.second);
        }
        ++it;

        bool mustReport = false;
        if(m_reportRate > 0 && it % m_reportRate == 0)
            mustReport = true;

        if(mustReport) {
            double mse = 0;
            foreach(const TrainSet& set, m_trainSets) {
                mse += test(set.first, set.second);
            }

            printf("iteration: %d  mse: %f\n", it, mse);
            fflush(stdout);

            if(m_stopFunc == STOP_MSE && mse <= m_stopParam)
                break;
        }
    }
    m_trainSets.clear();
}
