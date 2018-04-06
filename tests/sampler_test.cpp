#include "catch.hpp"

#include <fstream>
#include <iostream>

#include <mcmc/samplers/adaptive_equi_energy_sampler.hpp>
#include <mcmc/samplers/differential_evolution_sampler.hpp>
#include <mcmc/samplers/gibbs_sampler.hpp>
#include <mcmc/samplers/hamiltonian_monte_carlo_sampler.hpp>
#include <mcmc/samplers/metropolis_adjusted_langevin_sampler.hpp>
#include <mcmc/samplers/random_walk_metropolis_hastings_sampler.hpp>
#include <mcmc/samplers/riemannian_manifold_hamiltonian_monte_carlo_sampler.hpp>
#include <mcmc/markov_chain.hpp>

double normal_distribution_density(
  double       x           , 
  const double mu          = 0.0  , 
  const double sigma       = 1.0  ,
  const int    logarithmic = false)
{
  x = fabs((x - mu) / sigma);
  return logarithmic ? -(0.918938533204672741780329736406 + 0.5 * x * x + log(sigma)) : 0.398942280401432677939946059934 * exp(-0.5 * x * x) / sigma;
}

/* TEST_CASE("Random Walk Metropolis-Hastings sampler is tested.", "[mcmc::random_walk_metropolis_hastings_sampler]")
{
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 450.0f;
  
  Eigen::MatrixXf covariance_matrix(1, 1);
  covariance_matrix.setIdentity();

  mcmc::random_walk_metropolis_hastings_sampler<Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> metropolis_hastings_sampler(
    [ ] (const Eigen::VectorXf state)
    {
      return normal_distribution_density(state[0], 500.0f, 1.0f, true);
    },
    covariance_matrix, 
    100.0f);

  const auto iterations = 100000;

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < iterations; ++i)
  {
    markov_chain.update(metropolis_hastings_sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n"; // In-situ.
  }

  std::ofstream file("output_rwmh.csv");
  for (auto state : markov_chain.state_history())
    file << state.format(Eigen::IOFormat()) << "\n";
} */
TEST_CASE("Hamiltonian Monte Carlo sampler is tested."        , "[mcmc::hamiltonian_monte_carlo_sampler]")
{
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 500.0f;

  Eigen::VectorXf momentum(1);
  initial_state[0] = 0.0f;

  Eigen::MatrixXf precondition_matrix(1, 1);
  precondition_matrix.setIdentity();

  mcmc::hamiltonian_monte_carlo_sampler<Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> hamiltonian_monte_carlo_sampler(
    [ ] (const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
    {
      return normal_distribution_density(state[0], 500.0f, 1.0f, true);
    },
    precondition_matrix,
    20,
    100.0f);

  const auto iterations = 100000;

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < iterations; ++i)
  {
    markov_chain.update(hamiltonian_monte_carlo_sampler, &momentum);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n"; // In-situ.
  }

  std::ofstream file("output_hmc.csv");
  for (auto state : markov_chain.state_history())
    file << state.format(Eigen::IOFormat()) << "\n";
}
