#include "catch.hpp"

#include <iostream>

#include <mcmc/samplers/adaptive_equi_energy_sampler.hpp>
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

TEST_CASE("Adaptive equi-energy sampler is tested.", "[mcmc::adaptive_equi_energy_sampler]")
{
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 450.0f;
  
  Eigen::MatrixXf covariance_matrix(1, 1);
  covariance_matrix.setIdentity();

  mcmc::adaptive_equi_energy_sampler<Eigen::VectorXf> sampler(
    [ ] (const Eigen::VectorXf state)
    {
      return normal_distribution_density(state[0], 500.0f, 1.0f, true);
    },
    covariance_matrix, 
    100.0f);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  REQUIRE(Approx(markov_chain.state()[0]).epsilon(1.0) == 500.0f);
}