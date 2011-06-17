#include <stdio.h>
#include "ann.h"

void run_ann_benckmark(const char *datasetFilename,
              uint maxEphocs,
              uint epochsBetweenReports,
              nnreal desiredRMSE,
              nnreal bitFailLimit,
              uint numInput,
              uint numHidden,
              uint numHidden2,
              uint numOutput,
              nnreal learningRate,
              nnreal momentum,
              ANNActivateFunction hiddenActivateFunction,
              ANNActivateFunction outputActivateFunction)
{
    ANNet net;
    nnreal elapsed;

    printf(">> training dataset '%s' with desired RMSE %f\n", datasetFilename, desiredRMSE);

    ann_init(&net);
    ann_add_layer(&net, numInput);
    ann_add_layer(&net, numHidden);
    ann_add_layer(&net, numHidden2);
    ann_add_layer(&net, numOutput);
    ann_set_activate_function(&net, hiddenActivateFunction, ANN_HIDDEN_LAYERS);
    ann_set_activate_function(&net, outputActivateFunction, ANN_OUTPUT_LAYER);
    ann_set_learning_rate(&net, learningRate, ANN_ALL_LAYERS);
    ann_set_momentum(&net, momentum, ANN_ALL_LAYERS);
    ann_load_train_sets(&net, datasetFilename);

    elapsed = ann_get_millis();
    ann_train(&net, maxEphocs, epochsBetweenReports, desiredRMSE, bitFailLimit);
    elapsed = (ann_get_millis() - elapsed)/1000.0;

    printf(">> train completed in %.3f seconds\n\n", elapsed);
}

int main(int argc, char **argv)
{
    run_ann_benckmark("datasets/xor.train",
                      100000, 1000, 0.01, 0.01,
                      2, 3, 0, 1,
                      0.7, 0,
                      ANN_SIGMOID,
                      ANN_LINEAR);

    run_ann_benckmark("datasets/mushroom.train",
                      300, 10, 0.01, 0.01,
                      125, 32, 0, 2,
                      0.7, 0,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    run_ann_benckmark("datasets/robot.train",
                      3000, 100, 0.01, 0.1,
                      48, 96, 0, 3,
                      0.7, 0.4,
                      ANN_SIGMOID,
                      ANN_SIGMOID);
    return 0;
}
