#ifndef MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <algorithm>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class hamiltonian_monte_carlo_sampler
{
public:
  hamiltonian_monte_carlo_sampler           ()                                             = default;
  hamiltonian_monte_carlo_sampler           (const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler           (      hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~hamiltonian_monte_carlo_sampler  ()                                             = default;
  hamiltonian_monte_carlo_sampler& operator=(const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler& operator=(      hamiltonian_monte_carlo_sampler&& temp) = default;

  state_type apply(const state_type& state, state_type& gradients)
  {
    return state;
  }

protected:

};
}

#endif