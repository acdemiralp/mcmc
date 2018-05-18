#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <limits>
#include <math.h>
#include <iostream>

#include <mcmc/samplers/differential_evolution_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Differential evolution sampler is tested.", "[mcmc::differential_evolution_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(250.0f, 0.1f);
  const auto data = data_generator.generate<Eigen::VectorXf>(100);
  
  Eigen::VectorXf lower_bounds (1);
  Eigen::VectorXf upper_bounds (1);
  Eigen::VectorXf initial_state(1);
  lower_bounds [0] = -10000000.0f;
  upper_bounds [0] =  10000000.0f;
  initial_state[0] =  100000.0f  ;
  
  auto log_likelihood_density = [ ] (const Eigen::VectorXf& state, const Eigen::VectorXf& data, const float sigma = 1.0f)
  {
    return -static_cast<float>(data.size()) * (0.5f * std::log(2.0f * M_PI) + std::log(sigma)) - ((data.array() - state[0]).pow(2) / (2.0f * std::pow(sigma, 2))).sum();
  };
  auto log_prior_density      = [ ] (const Eigen::VectorXf& state, const float mu = 0.0f, const float sigma = 1.0f)
  {
    return -0.5f * std::log(2.0f * M_PI) - std::log(sigma) - std::pow(state[0] - mu, 2) / (2.0f * std::pow(sigma, 2));
  };

  mcmc::differential_evolution_sampler<float, Eigen::VectorXf, Eigen::MatrixXf> sampler(
    [=] (const Eigen::VectorXf& state)
    {
      return log_likelihood_density(state, data, 0.1f) + log_prior_density(state, 0.0f, 1.0f);
    },
    100,
    lower_bounds,
    upper_bounds);

  mcmc::markov_chain<Eigen::MatrixXf> markov_chain(sampler.setup(initial_state));
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()(0, 0)).epsilon(0.1) == 250.0f);
}