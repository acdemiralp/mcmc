#ifndef MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_
#define MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <random>

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
    const std::function<matrix_type(const matrix_type&)>                log_target_gradient_function,
    const std::size_t                                                   states                      = std::size_t(1   )          ,
    const std::size_t                                                   particles                   = std::size_t(1000)          ,
    const scalar_type                                                   step_size                   = scalar_type(0.1 )          ,
    const scalar_type                                                   fudge_factor                = scalar_type(1e-6)          ,
    const scalar_type                                                   alpha                       = scalar_type(0.9 )          ,
    const std::function<std::array<matrix_type, 2>(const matrix_type&)> kernel_function             = default_kernel_function    ,
    const initial_distribution_type&                                    initial_distribution        = initial_distribution_type())
  : log_target_gradient_function_(log_target_gradient_function)
  , kernel_function_             (kernel_function             )
  , particles_                   (particles                   )
  , states_                      (states                      )
  , step_size_                   (step_size                   )
  , fudge_factor_                (fudge_factor                )
  , alpha_                       (alpha                       )
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
    perturbation_history_ = matrix_type();
    return initialization_rng_.template generate<matrix_type>({particles_, states_});
  }
  matrix_type apply(const matrix_type& state)
  {
    auto        target_gradient = log_target_gradient_function_(state);
    auto        kernel          = kernel_function_             (state);
    matrix_type perturbation    = (std::get<0>(kernel) * target_gradient + std::get<1>(kernel)) / particles_;
    perturbation_history_       = perturbation_history_.rows() == 0 
      ? perturbation.array().pow(2) 
      : matrix_type(alpha_ * perturbation_history_.array() + (scalar_type(1) - alpha_) * perturbation.array().pow(2));
    perturbation                = perturbation.array() / (fudge_factor_ + perturbation_history_.array().sqrt());
    return state + step_size_ * perturbation;
  }

protected:
  static scalar_type                median                 (const matrix_type& state)
  {
    std::vector<scalar_type> values(state.size());
    std::copy_n     (state .data (), state .size()                     , values.data());
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end ());
    return values[values.size() / 2];
  }
  static std::array<matrix_type, 2> default_kernel_function(const matrix_type& state)
  {
    matrix_type k         = ((state * state.transpose() * -2).rowwise() + state.rowwise().squaredNorm().transpose()).colwise() + state.rowwise().squaredNorm();
    scalar_type bandwidth = scalar_type(0.5) * median(k) / std::log(k.rows() + 1);
    k = (scalar_type(-0.5) / bandwidth * k).array().exp();

    matrix_type dk  = -k * state;
    vector_type sum = k.colwise().sum();
    for (auto i = 0; i < dk.cols(); ++i)
      dk.col(i) = dk.col(i).array() + state.col(i).array() * sum.array();
    dk /= bandwidth;

    return {k, dk};
  }

  std::function<matrix_type               (const matrix_type&)> log_target_gradient_function_;
  std::function<std::array<matrix_type, 2>(const matrix_type&)> kernel_function_             ;
  std::size_t                                                   particles_                   ;
  std::size_t                                                   states_                      ;
  scalar_type                                                   step_size_                   ;
  scalar_type                                                   fudge_factor_                ;
  scalar_type                                                   alpha_                       ;
  random_number_generator<initial_distribution_type>            initialization_rng_          ;
  matrix_type                                                   perturbation_history_        ;
};
}

#endif
