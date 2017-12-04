#include "catch.hpp"

#include <mcmc/monte_carlo.hpp>

TEST_CASE("Monte Carlo estimator is tested.", "[mcmc::monte_carlo]") 
{
  mcmc::monte_carlo<double> pi_estimator(
  [ ] (const std::function<double()>& rng)
  {
    return sqrtf(pow(rng(), 2) + pow(rng(), 2)) <= 1.0 ? 1.0 : 0.0;
  });

  for(auto i = 0; i < 100; ++i)
    REQUIRE(Approx(4.0 * pi_estimator.simulate(20000)).epsilon(0.01) == 3.14);
}