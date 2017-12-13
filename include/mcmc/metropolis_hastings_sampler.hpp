#ifndef MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <random>
#include <type_traits>

namespace mcmc
{
template<
  typename type, 
  typename prior_distribution_type      = std::gamma_distribution <type>, 
  typename likelihood_distribution_type = std::normal_distribution<type>>
class metropolis_hastings_sampler
{
public:
  metropolis_hastings_sampler(prior_distribution_type prior_distribution, likelihood_distribution_type likelihood_distribution) 
  : prior_distribution_(prior_distribution), likelihood_distribution_(likelihood_distribution)
  {
    static_assert(!std::is_function<prior_distribution_type>::value     , "Prior distribution is not a function."     );
    static_assert(!std::is_function<likelihood_distribution_type>::value, "Likelihood distribution is not a function.");
  }
  metropolis_hastings_sampler           (const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler           (      metropolis_hastings_sampler&& temp) = default;
  virtual ~metropolis_hastings_sampler  ()                                         = default;
  metropolis_hastings_sampler& operator=(const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler& operator=(      metropolis_hastings_sampler&& temp) = default;

protected:
  prior_distribution_type      prior_distribution_     ;
  likelihood_distribution_type likelihood_distribution_;
};
}

#endif
