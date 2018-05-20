#ifndef MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_
#define MCMC_METROPOLIS_ADJUSTED_LANGEVIN_SAMPLER_HPP_

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
  typename matrix_type                = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class metropolis_adjusted_langevin_sampler
{
public:
  explicit metropolis_adjusted_langevin_sampler(
    const std::function<density_type(const state_type&, state_type*)>&                           log_target_density_function  ,
    const std::function<density_type(const state_type&, const state_type&, const matrix_type&)>& log_proposal_density_function,
    const matrix_type&                                                                           precondition_matrix          ,
    const density_type                                                                           step_size                    = density_type(1),
    const proposal_distribution_type&                                                            proposal_distribution        = proposal_distribution_type())
  : log_proposal_density_function_(log_proposal_density_function)
  , precondition_matrix_          (precondition_matrix)
  , llt_precondition_matrix_      (precondition_matrix.llt().matrixLLT())
  , step_size_                    (step_size)
  , proposal_rng_                 (proposal_distribution)
  , acceptance_rng_               (0, 1)
  , current_density_              (0)
  {
    log_target_density_function_ = [=] (const state_type& state) -> density_type
    {
      return log_target_density_function(state, nullptr);
    };
    log_mean_function_           = [=] (const state_type& state) -> state_type
    {
      state_type gradients(state.size());
      log_target_density_function(state, &gradients);
      return state + std::pow(step_size_, 2) * precondition_matrix_ * gradients / density_type(2);
    };
  }
  metropolis_adjusted_langevin_sampler           (const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler           (      metropolis_adjusted_langevin_sampler&& temp) = default;
  virtual ~metropolis_adjusted_langevin_sampler  ()                                                  = default;
  metropolis_adjusted_langevin_sampler& operator=(const metropolis_adjusted_langevin_sampler&  that) = default;
  metropolis_adjusted_langevin_sampler& operator=(      metropolis_adjusted_langevin_sampler&& temp) = default;
  
  void         setup (const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
  }
  state_type   apply (const state_type& state)
  {
    const state_type   random     = proposal_rng_.template generate<state_type>(state.size());
    const state_type   next_state = log_mean_function_(state) + step_size_ * llt_precondition_matrix_ * random;
    const density_type density    = log_target_density_function_(next_state);
    if (std::exp(std::min(density_type(0), density - current_density_ + adjust(state, next_state))) < acceptance_rng_.generate())
      return state;
    current_density_ = density;
    return next_state;
  }

protected:
  density_type adjust(const state_type& state, const state_type& next_state)
  {
    matrix_type matrix = std::pow(step_size_, 2) * precondition_matrix_;
    return log_proposal_density_function_(state     , log_mean_function_(next_state), matrix) - 
           log_proposal_density_function_(next_state, log_mean_function_(state     ), matrix);
  }

  std::function<density_type(const state_type&)>                                        log_target_density_function_  ;
  std::function<state_type  (const state_type&)>                                        log_mean_function_            ;
  std::function<density_type(const state_type&, const state_type&, const matrix_type&)> log_proposal_density_function_;
  matrix_type                                                                           precondition_matrix_          ;
  matrix_type                                                                           llt_precondition_matrix_      ;
  density_type                                                                          step_size_                    ;
  random_number_generator<proposal_distribution_type>                                   proposal_rng_                 ;
  random_number_generator<std::uniform_real_distribution<density_type>>                 acceptance_rng_               ;
  density_type                                                                          current_density_              ;
};
}

#endif