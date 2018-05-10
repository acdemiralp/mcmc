#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>
#include "mcmc/samplers/metropolis_hastings_sampler.hpp"

TEST_CASE("Metropolis-Hastings sampler is tested.", "[mcmc::metropolis_hastings_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(250.0f, 0.1f);
  const auto data = data_generator.generate<Eigen::VectorXf>(100);
  
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 1000.0f;
  
  Eigen::MatrixXf covariance_matrix(1, 1);
  covariance_matrix.setIdentity();

  auto log_likelihood_density = [ ] (const Eigen::VectorXf& state, const Eigen::VectorXf& data, const float sigma = 1.0f)
  {
    return -static_cast<float>(data.size()) * (0.5f * std::log(2.0f * M_PI) + std::log(sigma)) - ((data.array() - state[0]).pow(2) / (2.0f * std::pow(sigma, 2))).sum();
  };
  auto log_prior_density      = [ ] (const Eigen::VectorXf& state, const float mu = 0.0f, const float sigma = 1.0f)
  {
    return -0.5f * std::log(2.0f * M_PI) - std::log(sigma) - std::pow(state[0] - mu, 2) / (2.0f * std::pow(sigma, 2));
  };

  mcmc::metropolis_hastings_sampler<Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
    [=] (const Eigen::VectorXf& state)
    {
      return log_likelihood_density(state, data, 0.1f) + log_prior_density(state, 0.0f, 1.0f);
    },
    std::normal_distribution<float>(0.0f, 1.0f),
    [ ] (const float x, const float mu)
    {
      return -0.5f * std::log(2.0f * M_PI) - std::log(1.0f) - std::pow(x - mu, 2) / (2.0f * std::pow(1.0f, 2));
    },
    covariance_matrix);
  sampler.setup(initial_state);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(0.1) == 250.0f);
}