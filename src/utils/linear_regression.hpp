//
// Created by xinyi on 25-3-10.
//

#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

using namespace std;

template<typename T>
using LargeSignedT = typename std::conditional_t<std::is_floating_point_v<T>,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128>>;

template <typename keyType, typename indexType>
class LinearRegression {
private:
    struct Point {
        keyType x;
        indexType y;
    };

    LargeSignedT<keyType> sum_x;
    LargeSignedT<indexType> sum_y;
    long double sum_xx;
    long double sum_xy;
    uint64_t count;

    vector<Point> points;

public:
    LinearRegression()
        : sum_x(0.0L), sum_y(0.0L), sum_xx(0.0L), sum_xy(0.0L), count(0) {}

    void addDataPoint(keyType x, indexType y) {
        count++;
        sum_x  += x;
        sum_y  += y;
        sum_xx += static_cast<long double>(x) * x;
        sum_xy += static_cast<long double>(x) * y;
        points.push_back({x, y});
    }

    std::pair<long double, LargeSignedT<indexType>> computeModel() const {
        if (count < 2) return false;
        long double denominator = count * sum_xx - sum_x * sum_x;
        if (fabsl(denominator) < 1e-10L) return false;
        long double slope = (count * sum_xy - sum_x * sum_y) / denominator;
        LargeSignedT<indexType> intercept = (sum_y - slope * sum_x) / count;
        return true;
    }

    indexType computeMaxError(long double slope, LargeSignedT<indexType> intercept) const {
        double max_error = 0.0L;
        for (const auto &p : points) {
            long double predicted = slope * p.x + intercept;
            double error = fabsl(p.y - predicted);
            if (error > max_error)
                max_error = error;
        }
        return std::ceil(max_error);
    }

    void printModel() const {
        long double slope, intercept;
        if (computeModel(slope, intercept)) {
            cout << "\nmodel: y = " << slope << " * x + " << intercept << "\n";
        }
    }
};


#endif //LINEAR_REGRESSION_HPP
