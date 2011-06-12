#include "util.h"
#include "neuralnetwork.h"

int main(int argc, char **argv)
{
    NeuralNetwork *net = new NeuralNetwork;
    net->setIO(2, 1);
    net->createLayers({ 4, 4, 8 });
    net->setRandomWeightLimits(-0.1, 0.1);
    net->setLearningRate(0.7);
    net->setMomentum(0.4);
    net->setStepness(0.5);
    net->setStopFunc(STOP_MSE, pow(0.001, 2));
    net->setMaxIterations(100000);
    net->setReportRate(10000);

    for(int x=0;x<9;x++) {
        for(int y=0;y<9;y++) {
            net->addTrainSet(TrainSet( { (double)x, (double)y }, { (double)(x+y)/100.0 } ));
        }
    }
    net->train();

    for(int x=0;x<9;x++) {
        for(int y=0;y<9;y++) {
            double res = x + y;
            double got = net->execute({ (double)x, (double)y })[0] * 100.0;
            double err = res - got;
            printf("%d + %d = %f  err: %f\n", x, y, got, err*err);
        }
    }
    return 0;
}
