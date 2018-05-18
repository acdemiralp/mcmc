#ifndef MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_
#define MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include <math.h>

#include <external/Eigen/Dense>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename scalar_type = float,
  typename vector_type = Eigen::VectorXf,
  typename matrix_type = Eigen::MatrixXf>
class differential_evolution_sampler
{
public:
  explicit differential_evolution_sampler(
    const std::function<scalar_type(const vector_type&)>& log_target_density_function,
    const std::uint32_t                                   populations                ,
    const vector_type&                                    lower_bounds               ,
    const vector_type&                                    upper_bounds               ,
    const scalar_type                                     bandwidth_factor           = 1E-04,
    const bool                                            jumping                    = false,
    const std::uint32_t                                   jumping_interval           = 10,
    const scalar_type                                     jump_gamma                 = scalar_type(2),
    const std::function<scalar_type(std::uint32_t)>&      temperature_function       = std::bind(default_temperature_function, std::placeholders::_1))
  : log_target_density_function_(log_target_density_function          )
  , temperature_function_       (temperature_function                 )
  , populations_                (populations                          )
  , lower_bounds_               (lower_bounds                         )
  , upper_bounds_               (upper_bounds                         )
  , bandwidth_factor_           (bandwidth_factor                     )
  , jumping_                    (jumping                              )
  , jumping_interval_           (jumping_interval                     )
  , jump_gamma_                 (jump_gamma                           )
  , generation_rng_             (0, 1                                 )
  , selection_rng_              (0, populations_ - 1                  )
  , proposal_rng_               (-bandwidth_factor_, bandwidth_factor_)
  , acceptance_rng_             (0, 1                                 )
  , gamma_                      (0                                    )
  , current_gamma_              (0                                    )
  , current_iteration_          (0                                    )
  {

  }
  differential_evolution_sampler           (const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler           (      differential_evolution_sampler&& temp) = default;
  virtual ~differential_evolution_sampler  ()                                            = default;
  differential_evolution_sampler& operator=(const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler& operator=(      differential_evolution_sampler&& temp) = default;
  
  matrix_type setup(const vector_type& state)
  {
    matrix_type next_state(populations_, state.size());

    gamma_             = scalar_type(2.38) / std::sqrt(2.0 * state.size());
    current_gamma_     = gamma_;
    current_iteration_ = 0u;
    
    fitness_vector_.resize(populations_);

    auto range = upper_bounds_ - lower_bounds_;
    for (auto i = 0; i < populations_; ++i) // This for loop is very suitable for parallelization.
    {
      vector_type random   = generation_rng_.template generate<vector_type>(state.size());
      vector_type proposal = lower_bounds_ + range.cwiseProduct(random);
      fitness_vector_[i]   = log_target_density_function_(proposal);
      next_state.row (i)   = proposal.transpose();
    }

    return next_state;
  }
  matrix_type apply(const matrix_type& state)
  {
    matrix_type next_state = state;

    if (jumping_ && (current_iteration_ + 1) % jumping_interval_ == 0)
      current_gamma_ = jump_gamma_;

    for (auto i = 0; i < populations_; ++i) // This for loop is very suitable for parallelization.
    {
      auto lhs_index = selection_rng_.generate();
      auto rhs_index = selection_rng_.generate();
      while (lhs_index == i )                          lhs_index = selection_rng_.generate();
      while (rhs_index == i || lhs_index == rhs_index) rhs_index = selection_rng_.generate();

      const auto random      = proposal_rng_.template generate<vector_type>(state.size());
      const auto proposal    = next_state.row(i) + current_gamma_ * (next_state.row(lhs_index) - next_state.row(rhs_index)) + random;
      const auto fitness     = log_target_density_function_(proposal.transpose());
      const auto temperature = temperature_function_       (current_iteration_  );
      if (std::exp(fitness - fitness_vector_[i]) / temperature < acceptance_rng_.generate())
        continue;
      fitness_vector_[i] = fitness ;
      next_state.row (i) = proposal;
    }
    
    if (jumping_ && (current_iteration_ + 1) % jumping_interval_ == 0)
      current_gamma_ = gamma_;

    current_iteration_++;

    return next_state;
  }

protected:
  static scalar_type default_temperature_function(std::uint32_t index)
  {
    return scalar_type(1);
  }
  
  std::function<scalar_type(const matrix_type&)>                         log_target_density_function_  ;
  std::function<scalar_type(std::uint32_t)>                              temperature_function_         ;
                                                                                                       
  std::uint32_t                                                          populations_                  ;
  vector_type                                                            lower_bounds_                 ;
  vector_type                                                            upper_bounds_                 ;
  scalar_type                                                            bandwidth_factor_             ;
  bool                                                                   jumping_                      ;
  std::uint32_t                                                          jumping_interval_             ;
  scalar_type                                                            jump_gamma_                   ;
                                                                                                       
  random_number_generator<std::uniform_real_distribution<scalar_type>>   generation_rng_               ;
  random_number_generator<std::uniform_int_distribution <std::uint32_t>> selection_rng_                ;
  random_number_generator<std::uniform_real_distribution<scalar_type>>   proposal_rng_                 ;
  random_number_generator<std::uniform_real_distribution<scalar_type>>   acceptance_rng_               ;
                                                                                                       
  scalar_type                                                            gamma_                        ;
  scalar_type                                                            current_gamma_                ;
  std::uint32_t                                                          current_iteration_            ;
  vector_type                                                            fitness_vector_               ;
};
}

#endif