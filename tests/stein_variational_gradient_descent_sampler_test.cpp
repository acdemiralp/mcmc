#include "catch.hpp"

#include <iostream>

#include <mcmc/samplers/stein_variational_gradient_descent_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Stein Variational Gradient Descent (SVGD) sampler is tested.", "[mcmc::stein_variational_gradient_descent_sampler]")
{
  mcmc::random_number_generator<std::normal_distribution<float>> data_generator(10.0f, 1.0f);
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
    100,
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

TEST_CASE("Matrix squared Euclidean distance is tested.", "[mcmc::stein_variational_gradient_descent_sampler]")
{
  Eigen::MatrixXf input(4, 2);
  for (auto i = 0; i < input.rows(); ++i)
    input.row(i) = Eigen::Vector2f {i, i};

  Eigen::MatrixXf output = 
    ((input * input.transpose() * -2).rowwise() + input.rowwise().squaredNorm().transpose()).colwise() + input.rowwise().squaredNorm();

  for (auto i = 0; i < input.rows(); ++i)
    for (auto j = 0; j < input.cols(); ++j)
      REQUIRE(output(i, j) == (input.row(i) - input.row(j)).array().pow(2).sum());
}
