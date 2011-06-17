#include <stdio.h>
#include "ann.h"

#include <time.h>

void run_ann_benckmark(const char *datasetFilename,
                       nnreal maxTrainTime,
                       ANNStopMode stopMode,
              nnreal stopParam,
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

    printf(">> training dataset '%s'\n", datasetFilename);

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
    ann_train(&net, maxTrainTime, stopMode, stopParam);
    elapsed = (ann_get_millis() - elapsed)/1000.0;

    printf(">> train completed in %.3f seconds\n\n", elapsed);
}

int main(int argc, char **argv)
{
    run_ann_benckmark("datasets/xor.train",
                      300, ANN_STOP_NO_BITFAILS, 0.035,
                      2, 3, 0, 1,
                      0.7, 0,
                      ANN_SIGMOID,
                      ANN_LINEAR);

    run_ann_benckmark("datasets/mushroom.train",
                      300, ANN_STOP_NO_BITFAILS, 0.035,
                      125, 32, 0, 2,
                      0.7, 0,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    run_ann_benckmark("datasets/robot.train",
                      300, ANN_STOP_NO_BITFAILS, 0.1,
                      48, 96, 0, 3,
                      0.7, 0.4,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    /*
    run_ann_benckmark("datasets/parity8.train",
                      300, ANN_STOP_NO_BITFAILS, 0.1,
                      8, 16, 0, 1,
                      0.7, 0.8,
                      ANN_SIGMOID,
                      ANN_SIGMOID);

    run_ann_benckmark("datasets/building.train",
                      300, ANN_STOP_NO_BITFAILS, 0.1,
                      14, 16, 0, 3,
                      0.7, 0,
                      ANN_SIGMOID,
                      ANN_SIGMOID);
    */
    return 0;
}
