#ifndef MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_
#define MCMC_DIFFERENTIAL_EVOLUTION_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include <math.h>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename scalar_type                = float,
  typename vector_type                = Eigen::VectorXf,
  typename matrix_type                = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<scalar_type>>
class differential_evolution_sampler
{
public:
  explicit differential_evolution_sampler(
    const std::function<scalar_type(const vector_type&)>& log_target_density_function,
    const std::uint32_t                                   populations                ,
    const std::uint32_t                                   generations                ,
    const vector_type&                                    lower_bounds               ,
    const vector_type&                                    upper_bounds               ,
    const scalar_type                                     bandwidth_factor           ,
    const bool                                            jumping                    ,
    const std::uint32_t                                   jumping_interval           ,
    const scalar_type                                     gamma_jump                 ,
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , populations_                (populations                )
  , generations_                (generations                )
  , lower_bounds_               (lower_bounds               )
  , upper_bounds_               (upper_bounds               )
  , bandwidth_factor_           (bandwidth_factor           )
  , jumping_                    (jumping                    )
  , jumping_interval_           (jumping_interval           )
  , gamma_jump_                 (gamma_jump                 )
  , proposal_rng_               (proposal_distribution      )
  , uniform_rng_                (0, 1                       )
  , gamma_                      (0                          )
  {

  }
  differential_evolution_sampler           (const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler           (      differential_evolution_sampler&& temp) = default;
  virtual ~differential_evolution_sampler  ()                                            = default;
  differential_evolution_sampler& operator=(const differential_evolution_sampler&  that) = default;
  differential_evolution_sampler& operator=(      differential_evolution_sampler&& temp) = default;
  
  void        setup(const matrix_type& state)
  {
    gamma_         = scalar_type(2.38) / std::sqrt(2.0 * state.size());
    current_gamma_ = gamma_;

    vector_type vector(populations_);               // TODO: Preserve during application.
    matrix_type matrix(populations_, state.size()); // TODO: Preserve during application.
    for (auto i = 0; i < populations_; ++i)
    {
      vector_type random = uniform_rng_.template generate<Eigen::VectorXf>(100);
      matrix.row(i)      = lower_bounds_.transpose() + (upper_bounds_ - lower_bounds_).cwiseProduct(random).transpose();
      vector[i]          = log_target_density_function_(matrix.row(i).transpose());
    }
  }
  matrix_type apply(const matrix_type& state)
  {
    // TODO: Implement.
    return state;
  }

protected:
  std::function<scalar_type(const matrix_type&)>                       log_target_density_function_;
  
  std::uint32_t                                                        populations_                ;
  std::uint32_t                                                        generations_                ;
  
  vector_type                                                          lower_bounds_               ;
  vector_type                                                          upper_bounds_               ;
  scalar_type                                                          bandwidth_factor_           ;
  bool                                                                 jumping_                    ;
  std::uint32_t                                                        jumping_interval_           ;
  scalar_type                                                          gamma_jump_                 ;

  random_number_generator<proposal_distribution_type>                  proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<scalar_type>> uniform_rng_                ;
  
  scalar_type                                                          gamma_                      ;
  scalar_type                                                          current_gamma_              ;
};
}

#endif