#ifndef MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <random>

namespace mcmc
{
template<typename type, typename prior_distribution = std::gamma_distribution<type>, typename likelihood_distribution = std::normal_distribution<type>>
class metropolis_hastings_sampler
{
public:
  metropolis_hastings_sampler           ()                                         = default;
  metropolis_hastings_sampler           (const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler           (      metropolis_hastings_sampler&& temp) = default;
  virtual ~metropolis_hastings_sampler  ()                                         = default;
  metropolis_hastings_sampler& operator=(const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler& operator=(      metropolis_hastings_sampler&& temp) = default;

protected:
  
};
}

#endif
