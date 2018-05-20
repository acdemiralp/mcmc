#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/adaptive_equi_energy_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Adaptive equi-energy sampler is tested.", "[mcmc::adaptive_equi_energy_sampler]")
{  
  auto log_mixture_of_gaussians_density = [ ] (const Eigen::VectorXf& x, const Eigen::VectorXf& weights, const Eigen::MatrixXf& mean, const Eigen::VectorXf& variance)
  {
    auto density = 0.0f;
    for (auto i = 0; i < weights.size(); i++)
      density += weights[i] * std::exp(-0.5f * (x - mean.col(0)).array().pow(2).sum() / variance[i]) / std::pow(2.0f * M_PI * variance[i], static_cast<float>(x.size()) / 2.0f);
    return std::log(density);
  };

  Eigen::VectorXf weights(2);
  weights[0]       = 0.5f;
  weights[1]       = 0.5f;
  
  Eigen::MatrixXf mean(2, 2);
  mean(0, 0)       = 2.0f;
  mean(1, 0)       = 2.0f;
  mean(0, 1)       = 1.0f;
  mean(1, 1)       = 1.0f;

  Eigen::VectorXf variance(2);
  variance[0]      = 0.1f;
  variance[1]      = 0.1f;

  Eigen::VectorXf initial_state(2);
  initial_state[0] = 2.0f;
  initial_state[1] = 2.0f;

  Eigen::VectorXf temperatures(2);
  temperatures[0]  = 50.0f;
  temperatures[1]  = 10.0f;

  Eigen::MatrixXf covariance_matrix(2, 2);
  covariance_matrix.setIdentity();
  covariance_matrix *= 0.35f;

  mcmc::adaptive_equi_energy_sampler<float, Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
    std::bind(log_mixture_of_gaussians_density, std::placeholders::_1, weights, mean, variance),
    covariance_matrix,
    1.0f,
    0.1f,
    10  );

  mcmc::markov_chain<Eigen::MatrixXf> markov_chain(sampler.setup(initial_state, temperatures, 1000, 10000));
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  // Not trivial for mixed distributions.
  // REQUIRE(Approx(markov_chain.state()[0]).epsilon(0.1) == 250.0f);
}