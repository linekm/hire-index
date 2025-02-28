#ifndef LEARNEDINDEX_MODEL_HPP
#define LEARNEDINDEX_MODEL_HPP

#include "pgm/piecewise_linear_model.hpp"

using namespace pgm::internal;
// X: key type, Y: position type
// Real epsilon is eps + 1
template <class X, class Y> class LinearModel {
  X max_key = 0;

public:
  OptimalPiecewiseLinearModel<X, Y> *opt;
  LinearModel(int eps) {
    this->opt = new OptimalPiecewiseLinearModel<X, Y>(eps);
  }
  LinearModel() { opt = nullptr;}
  ~LinearModel() {
    delete opt;
  }

  bool add_point(const X &key, Y position);

  void initialize_opt(int eps) {
    this->opt = new OptimalPiecewiseLinearModel<X, Y>(eps);
  }

  void delete_opt() {
    delete opt;
    opt = nullptr;
  }
};

template <class X, class Y>
bool LinearModel<X, Y>::add_point(const X &key, Y position) {
  if (opt) {
    if (max_key > key) {
      return false;
    }

    max_key = key;
    bool is_added = this->opt->add_point(key, position);
    return is_added;
  } else {
    return false;
  }
}

#endif // LEARNEDINDEX_MODEL_HPP
