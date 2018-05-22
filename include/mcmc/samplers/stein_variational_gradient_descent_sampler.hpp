#ifndef MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_
#define MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_

#include <cstddef>
#include <functional>

#include <external/Eigen/Core>

namespace mcmc
{
template<
  typename scalar_type = float,
  typename vector_type = Eigen::VectorXf,
  typename matrix_type = Eigen::MatrixXf>
class stein_variational_gradient_descent_sampler
{
public:
  explicit stein_variational_gradient_descent_sampler  (
    const std::function<vector_type(const vector_type&)>& log_target_gradient_function,
    const std::size_t                                     particles                   = std::size_t(1000),
    const scalar_type                                     step_size                   = scalar_type(0.1 ))
  : log_target_gradient_function_(log_target_gradient_function)
  , particles_                   (particles)
  , step_size_                   (step_size)
  {

  }
  stein_variational_gradient_descent_sampler           (const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler           (      stein_variational_gradient_descent_sampler&& temp) = default;
  virtual ~stein_variational_gradient_descent_sampler  ()                                                        = default;
  stein_variational_gradient_descent_sampler& operator=(const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler& operator=(      stein_variational_gradient_descent_sampler&& temp) = default;

  matrix_type setup (const vector_type& state)
  {
    matrix_type next_state(state.size(), particles_);
    next_state.setZeros();
    return next_state;
  }
  matrix_type apply (const matrix_type& state)
  {
    matrix_type next_state = state;
    for (auto i = 0; i < particles_; ++i) // Parallelize.
    {
      vector_type particle_i = next_state.col(i);
      vector_type sum(particle_i.size());
      for (auto j = 0; j < particles_; ++j) // Parallelize.
      {
        vector_type particle_j          = next_state.col(j);
        vector_type log_target_gradient = log_target_gradient_function_(particle_j);
        // sum += kernel(particle_j, particle_i) * log_target_gradient + kernel_gradient(particle_j, particle_i)
      }  
      next_state.col(i) += (step_size_ / particles_) * sum;
    }
    return next_state;
  }

protected:
  std::function<vector_type(const vector_type&)>                     log_target_gradient_function_;
  std::function<matrix_type(const vector_type&, const vector_type&)> kernel_function_             ;
  std::size_t                                                        particles_                   ;
  scalar_type                                                        step_size_                   ;
};
}

#endif
