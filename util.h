#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <sstream>
#include <vector>

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cmath>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

inline void seedRandom() { srand(time(NULL)); }

template<typename T>
inline T randomRange(T min, T max) { return (min + ((T)(((abs((double)(max - min)) + 1.0) * rand())/(RAND_MAX+1.0)))); }

inline double sigmoid(double x) { return (1.0/(1.0 + exp(-x))); }

struct Dump {
public:
    Dump() { }
    ~Dump() { std::cout << m_buf.str() << std::endl; }
    template<class T>  Dump &operator<<(const T &x) { m_buf << x << " "; return *this; }
private:
    std::ostringstream m_buf;
};

#define dump Dump()

#endif // UTIL_H
