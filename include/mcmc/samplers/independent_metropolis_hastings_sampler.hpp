#ifndef MCMC_INDEPENDENT_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_INDEPENDENT_METROPOLIS_HASTINGS_SAMPLER_HPP_

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
class independent_metropolis_hastings_sampler
{
public:
  independent_metropolis_hastings_sampler           ()                                                     = default;
  independent_metropolis_hastings_sampler           (const independent_metropolis_hastings_sampler&  that) = default;
  independent_metropolis_hastings_sampler           (      independent_metropolis_hastings_sampler&& temp) = default;
  virtual ~independent_metropolis_hastings_sampler  ()                                                     = default;
  independent_metropolis_hastings_sampler& operator=(const independent_metropolis_hastings_sampler&  that) = default;
  independent_metropolis_hastings_sampler& operator=(      independent_metropolis_hastings_sampler&& temp) = default;
  
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