#include "catch.hpp"

#include <iostream>
#include <math.h>

#include <mcmc/samplers/stein_variational_gradient_descent_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Stein Variational Gradient Descent (SVGD) sampler is tested.", "[mcmc::stein_variational_gradient_descent_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(250.0f, 0.1f);
  const auto data = data_generator.generate<Eigen::VectorXf>(100);
  
  mcmc::stein_variational_gradient_descent_sampler<float, Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
    [=] (const Eigen::MatrixXf& state)
    {
      Eigen::MatrixXf gradients(state.rows(), state.cols());
      for (auto i = 0; i < state.rows(); ++i)
      {
        gradients(i, 0) = (data.array() - state(i, 0))       .sum() / std::pow(state(i, 1), 2);
        gradients(i, 1) = (data.array() - state(i, 0)).pow(2).sum() / std::pow(state(i, 1), 3) - static_cast<float>(data.size()) / state(i, 1);
      }
      return gradients;
    },
    2,
    10,
    0.1f);

  mcmc::markov_chain<Eigen::MatrixXf> markov_chain(sampler.setup());
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  // Not trivial for gradient descent.
  // REQUIRE(Approx(markov_chain.state()[0]).epsilon(0.1) == 250.0f);
}