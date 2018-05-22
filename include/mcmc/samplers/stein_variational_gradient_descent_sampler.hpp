#ifndef MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_
#define MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_

#include <array>
#include <cstddef>
#include <functional>
#include <random>

#include <external/unsupported/Eigen/MatrixFunctions>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename scalar_type               = float,
  typename vector_type               = Eigen::VectorXf,
  typename matrix_type               = Eigen::MatrixXf,
  typename initial_distribution_type = std::normal_distribution<scalar_type>>
class stein_variational_gradient_descent_sampler
{
public:
  explicit stein_variational_gradient_descent_sampler  (
    const std::function<matrix_type(const matrix_type&)>                 log_target_gradient_function,
    const std::size_t                                                    states                      = std::size_t(1)             ,
    const std::size_t                                                    particles                   = std::size_t(1000)          ,
    const scalar_type                                                    step_size                   = scalar_type(0.1)           ,
    const std::function<std::array<matrix_type, 2>(const matrix_type&)>  kernel_function             = default_kernel_function    ,
    const initial_distribution_type&                                     initial_distribution        = initial_distribution_type())
  : log_target_gradient_function_(log_target_gradient_function)
  , kernel_function_             (kernel_function             )
  , particles_                   (particles                   )
  , states_                      (states                      )
  , step_size_                   (step_size                   )
  , initialization_rng_          (initial_distribution        )
  {

  }
  stein_variational_gradient_descent_sampler           (const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler           (      stein_variational_gradient_descent_sampler&& temp) = default;
  virtual ~stein_variational_gradient_descent_sampler  ()                                                        = default;
  stein_variational_gradient_descent_sampler& operator=(const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler& operator=(      stein_variational_gradient_descent_sampler&& temp) = default;

  matrix_type setup()
  {
    return initialization_rng_.template generate<matrix_type>({particles_, states_});
  }
  matrix_type apply(const matrix_type& state)
  {
    auto log_gradient = log_target_gradient_function_(state);
    auto kernel       = kernel_function_             (state);
    return state + (step_size_ / particles_) * std::get<0>(kernel) * log_gradient + std::get<1>(kernel);
  }

protected:
  static std::array<matrix_type, 2> default_kernel_function(const matrix_type& state)
  {
    matrix_type k         = (((state * state.transpose() * -2).rowwise() + state.rowwise().squaredNorm().transpose()).colwise() + state.rowwise().squaredNorm());
    scalar_type bandwidth = scalar_type(0.1); // median(k) / std::log(k.rows());
    k = (scalar_type(-0.5) / bandwidth * k).exp();

    matrix_type dk (state.rows(), state.cols());
    for (auto i = 0; i < state.rows(); ++i)
      dk.row(i) = scalar_type(1) / bandwidth * k.row(i).dot(state.row(i) - state.rowwise());
   
    return {k, dk};
  }

  std::function<matrix_type               (const matrix_type&)> log_target_gradient_function_;
  std::function<std::array<matrix_type, 2>(const matrix_type&)> kernel_function_             ;
  std::size_t                                                   particles_                   ;
  std::size_t                                                   states_                      ;
  scalar_type                                                   step_size_                   ;
  random_number_generator<initial_distribution_type>            initialization_rng_          ;
};
}

#endif
