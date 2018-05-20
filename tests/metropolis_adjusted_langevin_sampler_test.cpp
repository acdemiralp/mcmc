#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include <mcmc/samplers/metropolis_adjusted_langevin_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Metropolis adjusted Langevin sampler is tested.", "[mcmc::metropolis_adjusted_langevin_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(10.0f, 5.0f);
  const auto data = data_generator.generate<Eigen::VectorXf>(1000);
  
  Eigen::VectorXf initial_state(2);
  initial_state[0] = 100.0f;
  initial_state[1] = 100.0f;
  
  Eigen::MatrixXf precondition_matrix(2, 2);
  precondition_matrix.setIdentity();

  mcmc::metropolis_adjusted_langevin_sampler<float, Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
    [=] (const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
    {
      if(gradients)
      {
        (*gradients)[0] = (data.array() - state[0])       .sum() / std::pow(state[1], 2);
        (*gradients)[1] = (data.array() - state[0]).pow(2).sum() / std::pow(state[1], 3) - static_cast<float>(data.size()) / state[1];
      }
      return -static_cast<float>(data.size()) * (0.5f * std::log(2.0f * M_PI) + std::log(state[1])) - ((data.array() - state[0]).pow(2) / (2.0f * std::pow(state[1], 2))).sum();
    },
    [=] (const Eigen::VectorXf& state, const Eigen::VectorXf& mu, const Eigen::MatrixXf& sigma)
    {
      return 
        - 0.5f * state.size() * std::log(2.0f * M_PI) 
        - 0.5f * (std::log(sigma.determinant()) + (state - mu).transpose() * sigma.inverse() * (state - mu));
    },
    precondition_matrix, 
    0.1f);
  sampler.setup(initial_state);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(0.1) == 250.0f);
}