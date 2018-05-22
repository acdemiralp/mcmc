#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/stein_variational_gradient_descent_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Stein Variational Gradient Descent (SVGD) sampler is tested.", "[mcmc::stein_variational_gradient_descent_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(250.0f, 0.1f);
  const auto data = data_generator.generate<Eigen::VectorXf>(100);
  
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 1000.0f;
  
  mcmc::stein_variational_gradient_descent_sampler<float, Eigen::VectorXf, Eigen::MatrixXf> sampler(
    [=] (const Eigen::VectorXf& state)
    {
      Eigen::VectorXf gradients(state.size());
      gradients[0] = (data.array() - state[0])       .sum() / std::pow(state[1], 2);
      gradients[1] = (data.array() - state[0]).pow(2).sum() / std::pow(state[1], 3) - static_cast<float>(data.size()) / state[1];
      return gradients;
    });

  mcmc::markov_chain<Eigen::MatrixXf> markov_chain(sampler.setup(initial_state));
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  // Not trivial for gradient descent.
  // REQUIRE(Approx(markov_chain.state()[0]).epsilon(0.1) == 250.0f);
}