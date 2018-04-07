#ifndef MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_
#define MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_

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
class metropolis_adjusted_langevin_sampler
{
public:
  metropolis_adjusted_langevin_sampler           ()                                                  = default;
  metropolis_adjusted_langevin_sampler           (const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler           (      metropolis_adjusted_langevin_sampler&& temp) = default;
  virtual ~metropolis_adjusted_langevin_sampler  ()                                                  = default;
  metropolis_adjusted_langevin_sampler& operator=(const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler& operator=(      metropolis_adjusted_langevin_sampler&& temp) = default;
  
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