#ifndef MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_
#define MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_

#include <external/Eigen/Core>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class metropolis_adjusted_langevin_sampler
{
public:
  metropolis_adjusted_langevin_sampler           ()                                                  = default;
  metropolis_adjusted_langevin_sampler           (const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler           (      metropolis_adjusted_langevin_sampler&& temp) = default;
  virtual ~metropolis_adjusted_langevin_sampler  ()                                                  = default;
  metropolis_adjusted_langevin_sampler& operator=(const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler& operator=(      metropolis_adjusted_langevin_sampler&& temp) = default;

  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif