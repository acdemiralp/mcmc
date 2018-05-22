#ifndef MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_
#define MCMC_STEIN_VARIATIONAL_GRADIENT_DESCENT_SAMPLER_HPP_

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
    const std::function<vector_type(const vector_type&)>& log_target_gradient_function,
    const std::size_t                                     particles                   = std::size_t(1000),
    const scalar_type                                     step_size                   = scalar_type(0.1),
    const initial_distribution_type&                      initial_distribution        = initial_distribution_type())
  : log_target_gradient_function_(log_target_gradient_function)
  , particles_                   (particles)
  , step_size_                   (step_size)
  , initialization_rng_          (initial_distribution)
  {

  }
  stein_variational_gradient_descent_sampler           (const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler           (      stein_variational_gradient_descent_sampler&& temp) = default;
  virtual ~stein_variational_gradient_descent_sampler  ()                                                        = default;
  stein_variational_gradient_descent_sampler& operator=(const stein_variational_gradient_descent_sampler&  that) = default;
  stein_variational_gradient_descent_sampler& operator=(      stein_variational_gradient_descent_sampler&& temp) = default;

  matrix_type setup (const vector_type& state)
  {
    return initialization_rng_.template generate<matrix_type>(particles_, state.size());
  }
  matrix_type apply (const matrix_type& state)
  {
    matrix_type next_state = state;

    matrix_type k = 
      (((state.transpose() * state * -2).colwise() + 
         state.colwise().squaredNorm().transpose()).rowwise() + 
         state.colwise().squaredNorm()).exp();

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
  std::function<vector_type(const vector_type&)>     log_target_gradient_function_;
  std::size_t                                        particles_                   ;
  scalar_type                                        step_size_                   ;
  random_number_generator<initial_distribution_type> initialization_rng_          ;
};
}

#endif
