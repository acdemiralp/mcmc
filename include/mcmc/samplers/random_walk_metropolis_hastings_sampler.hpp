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
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class random_walk_metropolis_hastings_sampler
{
public:
  explicit random_walk_metropolis_hastings_sampler  (
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const covariance_matrix_type&                         covariance_matrix          ,
    const density_type                                    scale                      = density_type(1),
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , covariance_matrix_          ((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  , current_density_            (0)
  {

  }
  random_walk_metropolis_hastings_sampler           (const random_walk_metropolis_hastings_sampler&  that) = default;
  random_walk_metropolis_hastings_sampler           (      random_walk_metropolis_hastings_sampler&& temp) = default;
  virtual ~random_walk_metropolis_hastings_sampler  ()                                                     = default;
  random_walk_metropolis_hastings_sampler& operator=(const random_walk_metropolis_hastings_sampler&  that) = default;
  random_walk_metropolis_hastings_sampler& operator=(      random_walk_metropolis_hastings_sampler&& temp) = default;

  void       setup (const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
  }
  state_type apply (const state_type& state)
  {
    state_type random     = proposal_rng_.template generate<state_type>(state.size());
    state_type next_state = state + covariance_matrix_ * random;
    const auto density    = log_target_density_function_(next_state);
    if (std::exp(std::min(density_type(0), density - current_density_)) < acceptance_rng_.generate()) 
      return state;
    current_density_ = density;
    return next_state;
  }

protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_;
  covariance_matrix_type                                                covariance_matrix_          ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
  density_type                                                          current_density_            ;
};
}

#endif
