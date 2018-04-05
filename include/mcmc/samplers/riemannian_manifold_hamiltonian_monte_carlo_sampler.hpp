#ifndef MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <external/Eigen/Core>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class riemannian_manifold_hamiltonian_monte_carlo_sampler // New personal high score.
{
public:
  riemannian_manifold_hamiltonian_monte_carlo_sampler           ()                                                                 = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~riemannian_manifold_hamiltonian_monte_carlo_sampler  ()                                                                 = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;

  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif