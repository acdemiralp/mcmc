#ifndef MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_
#define MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_

#include <external/Eigen/Core>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class differential_evolution_sampler
{
public:
  differential_evolution_sampler           ()                                            = default;
  differential_evolution_sampler           (const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler           (      differential_evolution_sampler&& temp) = default;
  virtual ~differential_evolution_sampler  ()                                            = default;
  differential_evolution_sampler& operator=(const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler& operator=(      differential_evolution_sampler&& temp) = default;

  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif