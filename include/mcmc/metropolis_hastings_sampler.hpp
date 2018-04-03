#ifndef MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <algorithm>
#include <random>
#include <type_traits>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename data_type                    ,
  typename parameter_type               ,
  typename model_function_type          ,
  typename prior_distribution_type      = std::gamma_distribution <double>, 
  typename likelihood_distribution_type = std::normal_distribution<double>>
class metropolis_hastings_sampler
{
public:
  explicit metropolis_hastings_sampler  (
    prior_distribution_type      prior_distribution      = prior_distribution_type     (), 
    likelihood_distribution_type likelihood_distribution = likelihood_distribution_type()) 
  : prior_distribution_     (prior_distribution     )
  , likelihood_distribution_(likelihood_distribution)
  {
    static_assert(!std::is_function<prior_distribution_type>::value     , "Prior distribution is not a function."     );
    static_assert(!std::is_function<likelihood_distribution_type>::value, "Likelihood distribution is not a function.");
  }
  metropolis_hastings_sampler           (const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler           (      metropolis_hastings_sampler&& temp) = default;
  virtual ~metropolis_hastings_sampler  ()                                         = default;
  metropolis_hastings_sampler& operator=(const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler& operator=(      metropolis_hastings_sampler&& temp) = default;

  template<typename state_type>
  state_type apply(const state_type& state) const
  {
    // Pick from posterior distribution g(x' | x_t).
    const auto xp = 0.0F;

    // Calculate acceptance ratio.
    const auto a  = 0.0F;
    
    // Accept or reject step.
    return uniform_rng_.generate() <= a ? xp : state;
  }

protected:
  random_number_generator<type, prior_distribution_type>      uniform_rng_            ;
  random_number_generator<type, likelihood_distribution_type> normal_rng_             ;
  model_function_type                                         model_function_         ;
  prior_distribution_type                                     prior_distribution_     ;
  likelihood_distribution_type                                likelihood_distribution_;
};
}

#endif
