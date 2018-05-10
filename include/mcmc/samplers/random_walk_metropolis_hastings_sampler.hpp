#ifndef MCMC_RANDOM_WALK_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_RANDOM_WALK_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <functional>
#include <math.h>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class random_walk_metropolis_hastings_sampler
{
public:
  template<typename... arguments_type>
  explicit random_walk_metropolis_hastings_sampler  (
    const std::function<float(const state_type&)>& log_kernel_function   ,
    const covariance_matrix_type&                  covariance_matrix     ,
    const float                                    scale                 = 1.0f,
    const proposal_distribution_type&              proposal_distribution = proposal_distribution_type())
  : log_kernel_function_(log_kernel_function)
  , covariance_matrix_  ((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_       (proposal_distribution)
  , acceptance_rng_     (0.0f, 1.0f)
  , current_density_    (0.0f)
  {

  }
  random_walk_metropolis_hastings_sampler           (const random_walk_metropolis_hastings_sampler&  that) = default;
  random_walk_metropolis_hastings_sampler           (      random_walk_metropolis_hastings_sampler&& temp) = default;
  virtual ~random_walk_metropolis_hastings_sampler  ()                                                     = default;
  random_walk_metropolis_hastings_sampler& operator=(const random_walk_metropolis_hastings_sampler&  that) = default;
  random_walk_metropolis_hastings_sampler& operator=(      random_walk_metropolis_hastings_sampler&& temp) = default;

  void       setup (const state_type& state)
  {
    current_density_ = log_kernel_function_(state);
  }
  state_type apply (const state_type& state)
  {
    state_type random     = proposal_rng_.template generate<state_type>(state.size());
    state_type next_state = state + covariance_matrix_ * random;
    const auto density    = log_kernel_function_(next_state);
    if (std::exp(std::min(0.0f, density - current_density_)) < acceptance_rng_.generate()) 
      return state;
    current_density_ = density;
    return next_state;
  }

protected:
  std::function<float(const state_type&)>                        log_kernel_function_;
  covariance_matrix_type                                         covariance_matrix_  ;
  random_number_generator<proposal_distribution_type>            proposal_rng_       ;
  random_number_generator<std::uniform_real_distribution<float>> acceptance_rng_     ;
  float                                                          current_density_    ;
};
}

#endif
