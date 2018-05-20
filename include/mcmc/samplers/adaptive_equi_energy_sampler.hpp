#ifndef MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_
#define MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>
#include <external/unsupported/Eigen/CXX11/Tensor>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename scalar_type                = float,
  typename vector_type                = Eigen::VectorXf,
  typename matrix_type                = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<scalar_type>>
class adaptive_equi_energy_sampler
{
public:
  explicit adaptive_equi_energy_sampler  (
    const std::function<scalar_type(const vector_type&)>& log_target_density_function,
    const matrix_type&                                    covariance_matrix          ,
    const scalar_type                                     scale                      = scalar_type(1),
    const scalar_type                                     interaction_probability    = scalar_type(0.1),
    const std::uint32_t                                   rings                      = 5u,
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , covariance_matrix_          ((scale * covariance_matrix).llt().matrixLLT())
  , interaction_probability_    (interaction_probability)
  , rings_                      (rings)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  , iteration_                  (0)
  , initial_draws_              (0)
  {

  }
  adaptive_equi_energy_sampler           (const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler           (      adaptive_equi_energy_sampler&& temp) = default;
  virtual ~adaptive_equi_energy_sampler  ()                                          = default;
  adaptive_equi_energy_sampler& operator=(const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler& operator=(      adaptive_equi_energy_sampler&& temp) = default;

  matrix_type setup(
    const vector_type& state        , 
    const vector_type& temperatures , 
    const std::size_t  initial_draws,
    const std::size_t  actual_draws )
  {
    auto k           = temperatures.size() + 1;
    auto total_draws = k * initial_draws + actual_draws;
    
    iteration_     = std::size_t(0);
    initial_draws_ = initial_draws;
    temperatures_.resize(k);
    temperatures_[k - 1] = scalar_type(1);
    std::copy(temperatures .data(), temperatures .data() + temperatures .size(), temperatures_.data());
    std::sort(temperatures_.data(), temperatures_.data() + temperatures_.size(), [ ] (scalar_type lhs, scalar_type rhs) { return lhs < rhs; });

    ring_matrix_       .resize(k, rings_ - 1 ); ring_matrix_       .setZero();
    density_history_   .resize(k, total_draws); density_history_   .setZero();
    previous_densities_.resize(2, k          ); previous_densities_.setZero();
    current_densities_ .resize(2, k          ); current_densities_ .setZero();
    state_history_     .resize(total_draws   );
    for (auto& history : state_history_)
    {
      history.resize (state.size(), k);
      history.setZero();
    }
    
    matrix_type initial_state(state.size(), k);
    initial_state.setZero();
    initial_state.col    (0) = state;
    return initial_state;
  }
  matrix_type apply(const matrix_type& state)
  {
    auto next_state   = state;
    auto mh_output    = tempered_mh_step(state.col(0), temperatures_[0]);
    next_state.col(0) = std::get<0>(mh_output);

    previous_densities_ = current_densities_;
    current_densities_.col(0).fill(std::get<1>(mh_output));

    for (auto i = 1; i < temperatures_.size(); ++i) // This for loop is very suitable for parallelization.
    {
      if (iteration_ <= i * initial_draws_) continue;

      if (acceptance_rng_.generate() > interaction_probability_)
      {
        mh_output = tempered_mh_step(state.col(i), temperatures_[i]);
        next_state.col(i) = std::get<0>(mh_output);
        current_densities_(0, i) = std::get<1>(mh_output) / temperatures_[i - 1];
        current_densities_(1, i) = std::get<1>(mh_output) / temperatures_[i];
      }
      else
      {
        const auto initial      = (i - 1) * initial_draws_;
        const auto ring_spacing = (iteration_ - initial + 1) / rings_;
        if (ring_spacing == 0)
        {
          next_state        .col(i) = state              .col(i);
          current_densities_.col(i) = previous_densities_.col(i);
        }
        else
        {
          vector_type              previous_densities = density_history_.block(i - 1, initial, 1, iteration_ - initial).transpose();
          std::vector<std::size_t> indices (previous_densities.size());
          std::iota(indices.begin(), indices.end (), std::size_t(0));
          std::sort(indices.begin(), indices.end (), 
            [&] (std::size_t lhs, std::size_t rhs) { return previous_densities[lhs] > previous_densities[rhs]; });
          std::sort(previous_densities.data (), previous_densities.data() + previous_densities.size(), 
            [&] (scalar_type lhs, scalar_type rhs) { return lhs > rhs; });

          for (auto j = 0; j < rings_ - 1; ++j)
          {
            auto index = (j + 1) * ring_spacing;
            ring_matrix_(i - 1, j) = (previous_densities[index] + previous_densities[index - 1]) / scalar_type(2);
          }

          auto ring_index = 0;
          while (ring_index < rings_ - 1 && ring_matrix_(i - 1, ring_index) < density_history_(i, iteration_ - 1))
            ring_index++;

          auto history_index = indices[static_cast<std::size_t>(ring_spacing * ring_index + std::floor(acceptance_rng_.generate() * ring_spacing))];

          next_state.col(i)        = state_history_[history_index].col(i - 1);
          auto density             = log_target_density_function_(next_state.col(i));
          current_densities_(0, i) = density / temperatures_[i - 1];
          current_densities_(1, i) = density / temperatures_[i];

          if (acceptance_rng_.generate() > std::exp(std::min(scalar_type(0), previous_densities_(0, i) + current_densities_(1, i) - previous_densities_(1, i) - current_densities_(0, i))))
          {
            next_state        .col(i) = state              .col(i);
            current_densities_.col(i) = previous_densities_.col(i);
          }
        }
      }

      density_history_(i, iteration_) = log_target_density_function_(next_state.col(i));
    }

    state_history_[++iteration_] = next_state;
    return next_state;
  }
  
protected:
  std::tuple<vector_type, scalar_type> tempered_mh_step(const vector_type& state, const scalar_type& temperature)
  {
    vector_type random       = proposal_rng_.template generate<vector_type>(state.size());
    vector_type next_state   = state + std::sqrt(temperature) * covariance_matrix_ * random;
    auto        density      = log_target_density_function_(state);
    auto        next_density = log_target_density_function_(next_state);
    if (std::exp(std::min(scalar_type(0), (next_density - density) / temperature)) < acceptance_rng_.generate())
      return {state, density};
    return {next_state, next_density};
  }

  std::function<scalar_type(const vector_type&)>                       log_target_density_function_;
  matrix_type                                                          covariance_matrix_          ;
  scalar_type                                                          interaction_probability_    ;
  std::uint32_t                                                        rings_                      ;

  random_number_generator<proposal_distribution_type>                  proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<scalar_type>> acceptance_rng_             ;
  
  std::size_t                                                          iteration_                  ;
  std::size_t                                                          initial_draws_              ;
  vector_type                                                          temperatures_               ;
  matrix_type                                                          ring_matrix_                ;
  matrix_type                                                          previous_densities_         ;
  matrix_type                                                          current_densities_          ;
  matrix_type                                                          density_history_            ;
  std::vector<matrix_type>                                             state_history_              ;
};
}

#endif