#ifndef MCMC_MULTIPLE_TRY_METROPOLIS_SAMPLER_HPP_
#define MCMC_MULTIPLE_TRY_METROPOLIS_SAMPLER_HPP_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
// Multiple-Try Metropolis (MTM) sampler (Liu et al., 2000).
// At each step, k candidate states are drawn from a symmetric Gaussian kernel.
// One candidate y* is selected with probability proportional to f(y_j).
// Then k-1 reference points are drawn from the same kernel centred on y*,
// and the acceptance ratio balances the forward and backward trial sums:
//   alpha = min(1, [sum_j f(y_j)] / [f(x) + sum_j f(x_j*)])
// All sums are computed in log-space via log-sum-exp for numerical stability.
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class multiple_try_metropolis_sampler
{
public:
  explicit multiple_try_metropolis_sampler(
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const covariance_matrix_type&                         covariance_matrix          ,
    const std::uint32_t                                   tries                      = 5u,
    const density_type                                    scale                      = density_type(1),
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , covariance_matrix_          ((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , tries_                      (tries)
  , proposal_rng_               (proposal_distribution)
  , selection_rng_              (0, 1)
  , acceptance_rng_             (0, 1)
  , current_density_            (0)
  {

  }
  multiple_try_metropolis_sampler           (const multiple_try_metropolis_sampler&  that) = default;
  multiple_try_metropolis_sampler           (      multiple_try_metropolis_sampler&& temp) = default;
  virtual ~multiple_try_metropolis_sampler  ()                                             = default;
  multiple_try_metropolis_sampler& operator=(const multiple_try_metropolis_sampler&  that) = default;
  multiple_try_metropolis_sampler& operator=(      multiple_try_metropolis_sampler&& temp) = default;

  void       setup (const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
  }
  state_type apply (const state_type& state)
  {
    // Step 1: draw k candidate proposals from q(·|x) = N(x, C)
    std::vector<state_type>   candidates    (tries_);
    std::vector<density_type> log_weights_f (tries_);
    for (auto j = 0u; j < tries_; ++j)
    {
      const state_type random = proposal_rng_.template generate<state_type>(state.size());
      candidates   [j] = state + covariance_matrix_ * random;
      log_weights_f[j] = log_target_density_function_(candidates[j]);
    }

    // Step 2: select y* from candidates proportional to exp(log_weights_f)
    const density_type log_T1  = log_sum_exp(log_weights_f);
    const std::size_t  y_index = categorical_sample(log_weights_f, log_T1);
    const state_type&  y_star  = candidates[y_index];

    // Step 3: draw k-1 reference points from q(·|y*) = N(y*, C)
    std::vector<density_type> log_weights_b(tries_);
    log_weights_b[0] = current_density_;  // x_0* = x (current state)
    for (auto j = 1u; j < tries_; ++j)
    {
      const state_type random = proposal_rng_.template generate<state_type>(state.size());
      const state_type x_ref  = y_star + covariance_matrix_ * random;
      log_weights_b[j]        = log_target_density_function_(x_ref);
    }
    const density_type log_T2 = log_sum_exp(log_weights_b);

    // Step 4: accept y* with probability min(1, T1/T2)
    if (std::exp(std::min(density_type(0), log_T1 - log_T2)) < acceptance_rng_.generate())
      return state;

    current_density_ = log_weights_f[y_index];
    return y_star;
  }

protected:
  static density_type log_sum_exp(const std::vector<density_type>& log_values)
  {
    const density_type max_val = *std::max_element(log_values.begin(), log_values.end());
    density_type sum = 0;
    for (const auto& v : log_values)
      sum += std::exp(v - max_val);
    return max_val + std::log(sum);
  }

  std::size_t categorical_sample(const std::vector<density_type>& log_weights, const density_type log_total)
  {
    const density_type u = selection_rng_.generate();
    density_type cumulative = 0;
    for (std::size_t j = 0; j < log_weights.size(); ++j)
    {
      cumulative += std::exp(log_weights[j] - log_total);
      if (u <= cumulative) return j;
    }
    return log_weights.size() - 1;
  }

  std::function<density_type(const state_type&)>                        log_target_density_function_;
  covariance_matrix_type                                                covariance_matrix_          ;
  std::uint32_t                                                         tries_                      ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> selection_rng_              ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
  density_type                                                          current_density_            ;
};
}

#endif
