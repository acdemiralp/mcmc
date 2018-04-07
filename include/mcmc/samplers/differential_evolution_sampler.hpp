#ifndef MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_
#define MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_

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
class differential_evolution_sampler
{
public:
  differential_evolution_sampler           ()                                            = default;
  differential_evolution_sampler           (const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler           (      differential_evolution_sampler&& temp) = default;
  virtual ~differential_evolution_sampler  ()                                            = default;
  differential_evolution_sampler& operator=(const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler& operator=(      differential_evolution_sampler&& temp) = default;
  
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