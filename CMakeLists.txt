CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(neuralnet)

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_C_FLAGS "-Wall")
SET(CMAKE_CXX_FLAGS "-Wall ")
SET(CMAKE_C_FLAGS_DEBUG "-O1 -g -ggdb -fno-inline")
SET(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -ggdb -fno-inline")
SET(CMAKE_C_FLAGS_RELEASE "-O3 -fomit-frame-pointer -march=nocona")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fomit-frame-pointer -march=nocona")

FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
ADD_EXECUTABLE(neuralnet ann.c main.cpp)
TARGET_LINK_LIBRARIES(neuralnet ${QT_LIBRARIES} -lqwt)

