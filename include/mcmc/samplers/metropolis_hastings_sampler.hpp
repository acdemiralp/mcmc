#ifndef MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <functional>
#include <math.h>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class metropolis_hastings_sampler
{
public:
  template<typename... arguments_type>
  explicit metropolis_hastings_sampler(
    const std::function<density_type(const state_type&)>&                    log_target_density_function  ,
    const covariance_matrix_type&                                            covariance_matrix            ,
    const proposal_distribution_type&                                        proposal_distribution        ,
    const std::function<density_type(const state_type&, const state_type&)>& log_proposal_density_function,
    const density_type                                                       scale                        = density_type(1))
  : log_target_density_function_  (log_target_density_function)
  , covariance_matrix_            ((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_                 (proposal_distribution)
  , log_proposal_density_function_(log_proposal_density_function)
  , acceptance_rng_               (0, 1)
  , current_density_              (0)
  {

  }
  metropolis_hastings_sampler           (const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler           (      metropolis_hastings_sampler&& temp) = default;
  virtual ~metropolis_hastings_sampler  ()                                         = default;
  metropolis_hastings_sampler& operator=(const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler& operator=(      metropolis_hastings_sampler&& temp) = default;
  
  void       setup (const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
  }
  state_type apply (const state_type& state)
  {
    state_type random                 = proposal_rng_.template generate<state_type>(state.size());
    state_type next_state             = state + covariance_matrix_ * random;
    const auto density                = log_target_density_function_  (next_state);
    const auto proposal_given_current = log_proposal_density_function_(next_state, state     );
    const auto current_given_proposal = log_proposal_density_function_(state     , next_state);
    if (std::exp(std::min(density_type(0), density + proposal_given_current - current_density_ - current_given_proposal)) < acceptance_rng_.generate()) 
      return state;
    current_density_ = density;
    return next_state;
  }

protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_  ;
  covariance_matrix_type                                                covariance_matrix_            ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_                 ;
  std::function<density_type(const state_type&, const state_type&)>     log_proposal_density_function_;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_               ;
  density_type                                                          current_density_              ;
};
}

#endif