#ifndef MCMC_ADAPTIVE_METROPOLIS_SAMPLER_HPP_
#define MCMC_ADAPTIVE_METROPOLIS_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
// Adaptive Metropolis (AM) sampler (Haario et al., 2001).
// Uses a random-walk Metropolis proposal whose covariance is adapted
// from the empirical covariance of the chain history:
//   C_t = s_d * Cov(X_0,...,X_{t-1}) + s_d * epsilon * I_d
// where s_d = (2.38)^2 / d is the asymptotically optimal scaling.
// Adaptation begins after `adapt_after` accepted/rejected steps.
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename matrix_type                = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class adaptive_metropolis_sampler
{
public:
  explicit adaptive_metropolis_sampler(
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const matrix_type&                                    initial_covariance         ,
    const std::uint32_t                                   adapt_after                = 100u,
    const density_type                                    epsilon                    = density_type(1e-6),
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , initial_chol_               (initial_covariance.llt().matrixLLT())
  , proposal_chol_              (initial_chol_)
  , adapt_after_                (adapt_after)
  , epsilon_                    (epsilon)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  , current_density_            (0)
  , iteration_                  (0u)
  {

  }
  adaptive_metropolis_sampler           (const adaptive_metropolis_sampler&  that) = default;
  adaptive_metropolis_sampler           (      adaptive_metropolis_sampler&& temp) = default;
  virtual ~adaptive_metropolis_sampler  ()                                         = default;
  adaptive_metropolis_sampler& operator=(const adaptive_metropolis_sampler&  that) = default;
  adaptive_metropolis_sampler& operator=(      adaptive_metropolis_sampler&& temp) = default;

  void       setup (const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
    iteration_       = 0u;
    running_mean_    = state;
    running_cov_sum_ = matrix_type::Zero(state.size(), state.size());
  }
  state_type apply (const state_type& state)
  {
    // Propose using the current (possibly adapted) Cholesky factor
    const state_type   random     = proposal_rng_.template generate<state_type>(state.size());
    const state_type   next_state = state + proposal_chol_ * random;
    const density_type density    = log_target_density_function_(next_state);

    const bool accepted = std::exp(std::min(density_type(0), density - current_density_)) >= acceptance_rng_.generate();
    const state_type& current = accepted ? next_state : state;
    if (accepted) current_density_ = density;

    // Update running statistics with Welford's online algorithm
    update_statistics(current);
    ++iteration_;

    return current;
  }

protected:
  void update_statistics(const state_type& sample)
  {
    // Welford's online mean and covariance (sum of squared deviations)
    const density_type t          = static_cast<density_type>(iteration_ + 1u);
    const state_type   old_mean   = running_mean_;
    running_mean_    += (sample - running_mean_) / t;
    running_cov_sum_ += (sample - old_mean) * (sample - running_mean_).transpose();

    // Adapt the proposal covariance once past the warm-up threshold
    if (iteration_ >= adapt_after_ && iteration_ >= 2u)
    {
      const density_type sd  = std::pow(density_type(2.38), 2) / static_cast<density_type>(sample.size());
      const density_type t1  = static_cast<density_type>(iteration_);
      const matrix_type  cov = sd * running_cov_sum_ / (t1 - density_type(1))
                             + sd * epsilon_ * matrix_type::Identity(sample.size(), sample.size());
      const Eigen::LLT<matrix_type> llt(cov);
      if (llt.info() == Eigen::Success)
        proposal_chol_ = llt.matrixLLT();
    }
  }

  std::function<density_type(const state_type&)>                        log_target_density_function_;
  matrix_type                                                           initial_chol_               ;
  matrix_type                                                           proposal_chol_              ;
  std::uint32_t                                                         adapt_after_                ;
  density_type                                                          epsilon_                    ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
  density_type                                                          current_density_            ;
  std::uint32_t                                                         iteration_                  ;
  state_type                                                            running_mean_               ;
  matrix_type                                                           running_cov_sum_            ;
};
}

#endif
