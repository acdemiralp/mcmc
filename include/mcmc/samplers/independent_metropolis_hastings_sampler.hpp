#ifndef MCMC_INDEPENDENT_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_INDEPENDENT_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <external/Eigen/Core>

namespace mcmc
{
template<typename state_type = Eigen::VectorXf>
class independent_metropolis_hastings_sampler
{
public:
  independent_metropolis_hastings_sampler           ()                                                     = default;
  independent_metropolis_hastings_sampler           (const independent_metropolis_hastings_sampler&  that) = default;
  independent_metropolis_hastings_sampler           (      independent_metropolis_hastings_sampler&& temp) = default;
  virtual ~independent_metropolis_hastings_sampler  ()                                                     = default;
  independent_metropolis_hastings_sampler& operator=(const independent_metropolis_hastings_sampler&  that) = default;
  independent_metropolis_hastings_sampler& operator=(      independent_metropolis_hastings_sampler&& temp) = default;

  state_type apply(const state_type& state)
  {
    return state;
  }

protected:

};
}

#endif