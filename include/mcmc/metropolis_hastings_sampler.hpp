#ifndef MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_
#define MCMC_METROPOLIS_HASTINGS_SAMPLER_HPP_

#include <algorithm>
#include <math.h>
#include <random>
#include <type_traits>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Dense>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class metropolis_hastings_sampler
{
public:
  template<typename... arguments_type>
  explicit metropolis_hastings_sampler  (
    const std::function<float(const state_type&)>& kernel_function       ,
    const covariance_matrix_type&                  covariance_matrix     ,
    const float                                    scale                 = 1.0f,
    const proposal_distribution_type&              proposal_distribution = proposal_distribution_type())
  : kernel_function_  (kernel_function      )
  , covariance_matrix_((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_     (proposal_distribution)
  , accept_rng_       (0.0f, 1.0f           )
  , last_density_     (0.0f                 )
  {

  }
  metropolis_hastings_sampler           (const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler           (      metropolis_hastings_sampler&& temp) = default;
  virtual ~metropolis_hastings_sampler  ()                                         = default;
  metropolis_hastings_sampler& operator=(const metropolis_hastings_sampler&  that) = default;
  metropolis_hastings_sampler& operator=(      metropolis_hastings_sampler&& temp) = default;

  state_type apply (const state_type& state)
  {
    state_type  random_vector(state.size());
    std::generate_n(&random_vector[0], random_vector.size(), proposal_rng_.function());

    state_type  next_state = state + covariance_matrix_ * random_vector;
    const float density    = kernel_function_(next_state);

    if (std::exp(std::min(0.0f, density - last_density_)) < accept_rng_.generate()) 
      return state;
    
    last_density_ = density;
    return next_state;
  }

protected:
  std::function<float(const state_type&)>                        kernel_function_  ;
  covariance_matrix_type                                         covariance_matrix_;
  random_number_generator<proposal_distribution_type>            proposal_rng_     ;
  random_number_generator<std::uniform_real_distribution<float>> accept_rng_       ;
  float                                                          last_density_     ;
};
}

#endif
