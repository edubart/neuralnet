CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(neuralnet)

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_C_FLAGS "-Wall")
SET(CMAKE_C_FLAGS_DEBUG "-O1 -g -ggdb")
SET(CMAKE_C_FLAGS_RELEASE "-O3 -fomit-frame-pointer -ffast-math")
SET(CMAKE_CXX_FLAGS "-Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fomit-frame-pointer -ffast-math")

ADD_EXECUTABLE(neuralnet main.cpp ann.c)
TARGET_LINK_LIBRARIES(neuralnet m)
