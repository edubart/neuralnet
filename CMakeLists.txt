project(neuralnet)
set(CMAKE_CXX_FLAGS "-Wall -std=gnu++0x -O2")
add_executable(neuralnet neuralnetwork.cpp neuronlayer.cpp synapse.cpp neuron.cpp main.cpp)
target_link_libraries(neuralnet fann)

