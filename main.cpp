#include <iostream>

#include "ann.h"

void runAnnBenchmark(const char *datasetFilename,
                       annreal maxTrainTime,
                       ANNStopMode stopMode,
                       annreal stopParam,
                       uint numInput,
                       uint numHidden,
                       uint numHidden2,
                       uint numOutput,
                       ANNActivateFunction hiddenActivateFunction,
                       ANNActivateFunction outputActivateFunction)
{
    annreal elapsed;

    std::cout << ">> training dataset '" << datasetFilename << "'" << std::endl;

    ANNetwork net;
    net.addLayer(numInput);
    net.addLayer(numHidden);
    net.addLayer(numHidden2);
    net.addLayer(numOutput);

    net.setActivateFunction(hiddenActivateFunction, ANN_HIDDEN_LAYERS);
    net.setActivateFunction(outputActivateFunction, ANN_OUTPUT_LAYER);
    net.setStopMode(stopMode);
    net.setTrainingAlgorithm(ANN_TRAIN_RPROP);
    net.setRpropParams(1.2, 0.5, 1e-6, 50);
    net.setStopMode(stopMode);
    if(stopMode == ANN_STOP_NO_BITFAILS)
        net.setBitFailLimit(stopParam);
    else if(stopMode == ANN_STOP_DESIRED_RMSE)
        net.setDesiredRMSE(stopParam);

    net.loadTrainSets(datasetFilename);

    elapsed = ANNetwork::getSeconds();
    net.train(maxTrainTime, 0.3);
    elapsed = ANNetwork::getSeconds() - elapsed;

    std::cout << ">> train completed in " << elapsed << " seconds" << std::endl << std::endl;
}

int main(int argc, char **argv)
{
    runAnnBenchmark("datasets/xor.train",
                      300, ANN_STOP_NO_BITFAILS, 0.035,
                      2, 3, 0, 1,
                      ANN_SIGMOID_SYMMETRIC,
                      ANN_LINEAR);

    runAnnBenchmark("datasets/mushroom.train",
                      300, ANN_STOP_NO_BITFAILS, 0.035,
                      125, 32, 0, 2,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    runAnnBenchmark("datasets/robot.train",
                      300, ANN_STOP_NO_BITFAILS, 0.1,
                      48, 96, 0, 3,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    runAnnBenchmark("datasets/parity8.train",
                      300, ANN_STOP_NO_BITFAILS, 0.1,
                      8, 16, 0, 1,
                      ANN_SIGMOID_SYMMETRIC,
                      ANN_LINEAR);
    return 0;
}