#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/riemannian_manifold_hamiltonian_monte_carlo_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Riemannian manifold Hamiltonian Monte Carlo sampler is tested.", "[mcmc::riemannian_manifold_hamiltonian_monte_carlo_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(10.0f, 5.0f);
  const auto data = data_generator.generate<Eigen::VectorXf>(1000);
  
  Eigen::VectorXf initial_state(2);
  initial_state[0] = 15.0f;
  initial_state[1] = 10.0f;
  
  mcmc::riemannian_manifold_hamiltonian_monte_carlo_sampler<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::Tensor<float, 3>, std::normal_distribution<float>> sampler(
    [=] (const Eigen::VectorXf& state, Eigen::VectorXf* gradients)
    {
      if(gradients)
      {
        (*gradients)[0] = (data.array() - state[0])       .sum() / std::pow(state[1], 2);
        (*gradients)[1] = (data.array() - state[0]).pow(2).sum() / std::pow(state[1], 3) - static_cast<float>(data.size()) / state[1];
      }
      return -static_cast<float>(data.size()) * (0.5f * std::log(2.0f * M_PI) + std::log(state[1])) - ((data.array() - state[0]).pow(2) / (2.0f * std::pow(state[1], 2))).sum();
    },
    [=] (const Eigen::VectorXf& state, Eigen::Tensor<float, 3>* tensor)
    {
      auto value = static_cast<float>(data.size()) / std::pow(state[1], 2);

      Eigen::MatrixXf matrix(2, 2);
      matrix.setIdentity();
      matrix(0, 0) =        value;
      matrix(1, 1) = 2.0f * value;
    
      if(tensor)
      {
        auto coefficients = -2.0f * matrix / state[1];
        tensor->resize  (2, 2, 2);
        tensor->setZero ();
        tensor->coeffRef(0, 0, 1) = coefficients(0, 0);
        tensor->coeffRef(0, 1, 1) = coefficients(0, 1);
        tensor->coeffRef(1, 0, 1) = coefficients(1, 0);
        tensor->coeffRef(1, 1, 1) = coefficients(1, 1);
      }
      return matrix;
    },
    4,
    4,
    0.02f);
  sampler.setup(initial_state);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(1.0f) == 10.0f);
  REQUIRE(Approx(markov_chain.state()[1]).epsilon(1.0f) ==  5.0f);
}