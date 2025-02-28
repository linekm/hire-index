#ifndef __LIPP_BASE_H__
#define __LIPP_BASE_H__

#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// Linear regression model
template <class T>
class LippLinearModel
{
public:
    double a = 0; // slope
    long double b = 0; // intercept

    LippLinearModel() = default;
    LippLinearModel(double a, long double b) : a(a), b(b) {}
    explicit LippLinearModel(const LippLinearModel &other) : a(other.a), b(other.b) {}

    inline int predict(T key) const
    {
        return std::floor(a * static_cast<long double>(key) + b);
    }

    inline double predict_double(T key) const
    {
        return a * static_cast<long double>(key) + b;
    }
};

#endif // __LIPP_BASE_H__
