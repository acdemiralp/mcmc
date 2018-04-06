#ifndef MCMC_GIBBS_SAMPLER_HPP_
#define MCMC_GIBBS_SAMPLER_HPP_

#include <algorithm>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class gibbs_sampler
{
public:
  template<typename... arguments_type>
  explicit gibbs_sampler  (
    const std::function<float(const state_type&)>& kernel_function       , // Expects logarithmic return values.
    const covariance_matrix_type&                  covariance_matrix     ,
    const float                                    scale                 = 1.0f,
    const proposal_distribution_type&              proposal_distribution = proposal_distribution_type())
  : kernel_function_  (kernel_function)
  , covariance_matrix_((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_     (proposal_distribution)
  , acceptance_rng_   (0.0f, 1.0f)
  , last_log_density_ (0.0f)
  {

  }
  gibbs_sampler           (const gibbs_sampler&  that) = default;
  gibbs_sampler           (      gibbs_sampler&& temp) = default;
  virtual ~gibbs_sampler  ()                           = default;
  gibbs_sampler& operator=(const gibbs_sampler&  that) = default;
  gibbs_sampler& operator=(      gibbs_sampler&& temp) = default;

  state_type apply (const state_type& state)
  {
    auto selection = selection_rng_.generate();
    

    state_type random_vector(state.size());
    std::generate_n(&random_vector[0], random_vector.size(), proposal_rng_.function());

    state_type next_state = state + covariance_matrix_ * random_vector;
    const auto density    = kernel_function_(next_state);

    if (std::exp(std::min(0.0f, density - last_log_density_)) < acceptance_rng_.generate()) 
      return state;
    
    last_log_density_ = density;
    return next_state;
  }

protected:
  std::function<float(const state_type&)>                              kernel_function_  ;
  covariance_matrix_type                                               covariance_matrix_;
  random_number_generator<std::uniform_real_distribution<std::size_t>> selection_rng_    ;
  random_number_generator<proposal_distribution_type>                  proposal_rng_     ;
  random_number_generator<std::uniform_real_distribution<float>>       acceptance_rng_   ;
  float                                                                last_log_density_ ;
};
}

#endif
