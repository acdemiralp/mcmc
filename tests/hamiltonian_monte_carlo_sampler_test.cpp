#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/hamiltonian_monte_carlo_sampler.hpp>
#include <mcmc/markov_chain.hpp>

TEST_CASE("Hamiltonian Monte Carlo sampler is tested.", "[mcmc::hamiltonian_monte_carlo_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(10.0f, 5.0f);
  const auto data = data_generator.generate<Eigen::VectorXf>(1000);
  
  Eigen::VectorXf initial_state(2);
  initial_state[0] = 100.0f;
  initial_state[1] = 100.0f;
  
  Eigen::MatrixXf precondition_matrix(2, 2);
  precondition_matrix.setIdentity();

  auto log_likelihood_density = [ ] (const Eigen::VectorXf& state, const Eigen::VectorXf& data, Eigen::VectorXf* gradients)
  {
    if(gradients)
    {
      (*gradients)[0] = (data.array() - state[0])       .sum() / std::pow(state[1], 2);
      (*gradients)[1] = (data.array() - state[0]).pow(2).sum() / std::pow(state[1], 3) - static_cast<float>(data.size()) / state[1];
    }
    return -static_cast<float>(data.size()) * (0.5f * std::log(2.0f * M_PI) + std::log(state[1])) - ((data.array() - state[0]).pow(2) / (2.0f * std::pow(state[1], 2))).sum();
  };

  mcmc::hamiltonian_monte_carlo_sampler<Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
    [=] (const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
    {
      return log_likelihood_density(state, data, gradients);
    },
    precondition_matrix, 
    20,
    0.1f);
  sampler.setup(initial_state);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 1000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(1.0f) == 10.0f);
  REQUIRE(Approx(markov_chain.state()[1]).epsilon(1.0f) ==  5.0f);
}