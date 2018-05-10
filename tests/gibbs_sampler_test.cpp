#include "catch.hpp"

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include <mcmc/samplers/gibbs_sampler.hpp>
#include <mcmc/markov_chain.hpp>
#include <mcmc/random_number_generator.hpp>

TEST_CASE("Gibbs sampler is tested.", "[mcmc::gibbs_sampler]")
{
  // Three unit vectors parametrized as (theta, phi). Assume they are placed horizontally: v1 v2 v3.
  Eigen::VectorXcf data(3);
  data[0] = {30.0f, 60.0f}; 
  data[1] = {15.0f, 45.0f};
  data[2] = {45.0f, 90.0f};

  mcmc::gibbs_sampler<std::complex<float>, Eigen::VectorXcf, std::normal_distribution<float>> sampler(
    [=] (const Eigen::VectorXcf& state, std::size_t index)
    {
      // TODO: Draw a sample from the conditional distribution (state[index] | state[!index]...).
      // The problem: Create a connected line segment from the three vectors without significantly deviating from the original data.
      // Model the conditional distribution in a way that is both dependent on the distance of the vector from the original data as well as its suitability to the current state of neighbours.
      // Note that v1 and v3 are dependent on v2 and v2 is dependent on v1 and v3 due to their horizontal placement (assuming a maximum neighbour distance of one).
      return std::complex<float>{0.0f, 0.0f};
    });

  mcmc::markov_chain<Eigen::VectorXcf> markov_chain(data);
  for(auto i = 0; i < 10000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }

  for (auto i = 0; i < markov_chain.state().size(); ++i)
  {
    REQUIRE(Approx(markov_chain.state()[i].real()).epsilon(1.0) == 0.0f);
    REQUIRE(Approx(markov_chain.state()[i].imag()).epsilon(1.0) == 0.0f);
  }
}