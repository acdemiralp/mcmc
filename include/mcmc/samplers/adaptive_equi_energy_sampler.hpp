#ifndef MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_
#define MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_

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
class adaptive_equi_energy_sampler
{
public:
  adaptive_equi_energy_sampler           ()                                          = default;
  adaptive_equi_energy_sampler           (const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler           (      adaptive_equi_energy_sampler&& temp) = default;
  virtual ~adaptive_equi_energy_sampler  ()                                          = default;
  adaptive_equi_energy_sampler& operator=(const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler& operator=(      adaptive_equi_energy_sampler&& temp) = default;

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