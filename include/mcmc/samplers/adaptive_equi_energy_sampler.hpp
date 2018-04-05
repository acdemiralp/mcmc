#ifndef MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_
#define MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_

#include <external/Eigen/Core>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class adaptive_equi_energy_sampler
{
public:
  adaptive_equi_energy_sampler           ()                                          = default;
  adaptive_equi_energy_sampler           (const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler           (      adaptive_equi_energy_sampler&& temp) = default;
  virtual ~adaptive_equi_energy_sampler  ()                                          = default;
  adaptive_equi_energy_sampler& operator=(const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler& operator=(      adaptive_equi_energy_sampler&& temp) = default;

  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif