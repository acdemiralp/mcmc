#ifndef MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class riemannian_manifold_hamiltonian_monte_carlo_sampler // New personal high score.
{
public:
  riemannian_manifold_hamiltonian_monte_carlo_sampler           ()                                                                 = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~riemannian_manifold_hamiltonian_monte_carlo_sampler  ()                                                                 = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;
  
  void       setup(const state_type& state)
  {

  }
  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif