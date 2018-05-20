#ifndef MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_
#define MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_

#include <cstddef>
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
class stein_variational_gradient_descent_sampler
{
public:
  explicit stein_variational_gradient_descent_sampler  (
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const std::size_t                                     particles                  = std::size_t (1000),
    const density_type                                    step_size                  = density_type(0.1 ),
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  {

  }
  stein_variational_gradient_descent_sampler           (const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler           (      stein_variational_gradient_descent_sampler&& temp) = default;
  virtual ~stein_variational_gradient_descent_sampler  ()                                                     = default;
  stein_variational_gradient_descent_sampler& operator=(const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler& operator=(      stein_variational_gradient_descent_sampler&& temp) = default;

  void       setup (const state_type& state)
  {

  }
  state_type apply (const state_type& state)
  {
    return state;
  }

protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
};
}

#endif
