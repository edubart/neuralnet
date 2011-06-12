#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "util.h"
#include <vector>
#include <list>

class NeuronLayer;

enum StopFunc {
    STOP_MSE,
    STOP_NEVER
};

typedef std::pair<std::vector<double>, std::vector<double> > TrainSet;

class NeuralNetwork
{
public:
    NeuralNetwork();

    std::vector<double> execute(const std::vector<double>& input);
    double test(const std::vector<double>& input, const std::vector<double>& output);
    void train(const std::vector<double>& input, const std::vector<double>& output);

    void addTrainSet(const TrainSet& trainSet) { m_trainSets.push_back(trainSet); }
    void train();

    void setIO(int numInputs, int numOutputs);
    void createLayers(const std::vector<int>& hiddenLayers = std::vector<int>());

    void setLearningRate(double rate) { m_learningRate = rate; }
    double learningRate() const { return m_learningRate; }

    void setMomentum(double momentum) { m_momentum = momentum; }
    double momentum() const  { return m_momentum; }

    void setRandomWeightLimits(double min, double max) { m_minRandomWeight = min; m_maxRandomWeight = max; }
    double generateRandomWeight() const { return randomRange(m_minRandomWeight, m_maxRandomWeight); }

    void setStepness(double stepness) { m_stepness = stepness; }
    double stepness() { return m_stepness; }

    void setMaxIterations(unsigned int maxIterations) { m_maxIterations = maxIterations; }
    void setReportRate(unsigned int iterations) { m_reportRate = iterations; }
    void setStopFunc(StopFunc stopFunc, double param = 0) { m_stopFunc = stopFunc; m_stopParam = param; }

private:
    NeuronLayer *m_inputLayer;
    NeuronLayer *m_outputLayer;
    std::vector<NeuronLayer*> m_layers;
    int m_numInputs;
    int m_numOutputs;
    double m_minRandomWeight, m_maxRandomWeight;
    double m_learningRate;
    double m_momentum;
    double m_stepness;
    unsigned int m_maxIterations;
    unsigned int m_reportRate;
    StopFunc m_stopFunc;
    double m_stopParam;

    std::list<TrainSet> m_trainSets;
};

#endif // NEURALNETWORK_H
