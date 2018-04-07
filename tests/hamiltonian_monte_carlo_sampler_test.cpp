#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/hamiltonian_monte_carlo_sampler.hpp>
#include <mcmc/markov_chain.hpp>

float normal_distribution_density(const float mu, const float sigma, const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
{
  if (gradients) 
  {
    (*gradients)[0] = (state.array() - mu).sum() / std::pow(sigma, 2);
    (*gradients)[1] = (state.array() - mu).pow(2).sum() / std::pow(sigma, 3) - 2.0f / sigma;
  }
  return -2.0f * (0.5f * std::log(2.0f * M_PI) + std::log(sigma)) - ((state.array() - mu).pow(2) / (2.0f * std::pow(sigma, 2))).sum();
}

TEST_CASE("Hamiltonian Monte Carlo sampler is tested.", "[mcmc::hamiltonian_monte_carlo_sampler]")
{
  Eigen::VectorXf initial_state(2);
  initial_state[0] = 2.0f; // Mean.
  initial_state[1] = 1.0f; // Standard deviation.

  Eigen::MatrixXf precondition_matrix(2, 2);
  precondition_matrix.setIdentity();

  mcmc::hamiltonian_monte_carlo_sampler<Eigen::VectorXf> sampler(
    [ ] (const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
    {
      return normal_distribution_density(4.0f, 1.0f, state, gradients);
    },
    precondition_matrix,
    1u,
    1.0f);
  sampler.setup(initial_state);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(1.0f) == 4.0f);
}