#ifndef MCMC_SLICE_SAMPLER_HPP_
#define MCMC_SLICE_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
// Componentwise univariate slice sampler (Neal, 2003).
// For each dimension, samples a "slice" height u ~ Uniform(0, f(x)), then
// uses a step-out/shrink procedure to draw the next state uniformly from
// the slice S = {x : f(x) >= u}.
template<
  typename density_type = float,
  typename state_type   = Eigen::VectorXf>
class slice_sampler
{
public:
  explicit slice_sampler(
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const density_type                                    step_size = density_type(1),
    const std::uint32_t                                   max_steps = 10u)
  : log_target_density_function_(log_target_density_function)
  , step_size_                  (step_size)
  , max_steps_                  (max_steps)
  , uniform_rng_                (0, 1)
  , exponential_rng_            (1)
  {

  }
  slice_sampler           (const slice_sampler&  that) = default;
  slice_sampler           (      slice_sampler&& temp) = default;
  virtual ~slice_sampler  ()                           = default;
  slice_sampler& operator=(const slice_sampler&  that) = default;
  slice_sampler& operator=(      slice_sampler&& temp) = default;

  void       setup (const state_type& state)
  {
    // Nothing to initialize; the sampler is stateless between calls.
  }
  state_type apply (const state_type& state)
  {
    state_type next_state = state;
    for (auto i = 0; i < next_state.size(); ++i)
    {
      // Sample the slice height in log space: log(u) = log_f(x) - Exp(1)
      const density_type log_height = log_target_density_function_(next_state) - exponential_rng_.generate();

      // Create initial interval [L, R] by positioning uniformly within one step
      const density_type u = uniform_rng_.generate();
      density_type       L = next_state[i] - u * step_size_;
      density_type       R = L + step_size_;

      // Step out: expand the interval while the density exceeds the slice height
      state_type state_L = next_state;
      state_type state_R = next_state;
      for (auto j = 0u; j < max_steps_; ++j)
      {
        state_L[i] = L;
        if (log_target_density_function_(state_L) <= log_height) break;
        L -= step_size_;
      }
      for (auto j = 0u; j < max_steps_; ++j)
      {
        state_R[i] = R;
        if (log_target_density_function_(state_R) <= log_height) break;
        R += step_size_;
      }

      // Shrink and sample: draw uniformly from [L, R], shrinking on rejection
      state_type proposal = next_state;
      while (true)
      {
        const density_type x_proposal = L + uniform_rng_.generate() * (R - L);
        proposal[i] = x_proposal;
        if (log_target_density_function_(proposal) >= log_height)
        {
          next_state[i] = x_proposal;
          break;
        }
        if (x_proposal < next_state[i]) L = x_proposal;
        else                            R = x_proposal;
      }
    }
    return next_state;
  }

protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_;
  density_type                                                          step_size_                  ;
  std::uint32_t                                                         max_steps_                  ;
  random_number_generator<std::uniform_real_distribution<density_type>> uniform_rng_                ;
  random_number_generator<std::exponential_distribution<density_type>>  exponential_rng_            ;
};
}

#endif
