// Copyright (c) 2021 Everything's Reduced authors
// SPDX-License-Identifier: MIT

#include <memory>
#include <tuple>

struct complex_sum_soa {

  // Problem size and data arrays
  // Data arrays use C++ PIMPL because different models store data with very different types
  long N = 1024*1024*1024;
  struct data;
  std::unique_ptr<data> pdata;

  // Constructor: set up any model initialisation (not data)
  complex_sum_soa();

  // Deconstructor: set any model finalisation
  ~complex_sum_soa();

  // Allocate and initalise benchmark data
  // C will be set to (2 * 1024)/N + i (2*1024/N)
  // Scaling the input data is helpful to keep the reduction in range
  void setup();

  // Run the benchmark once
  std::tuple<double, double> run();

  // Finalise, clearing any benchmark data
  void teardown();

  // Return expected result
  std::tuple<double, double> expect() {

    double v = 2.0 * 1024.0;
    return {v, v};
  }

};


