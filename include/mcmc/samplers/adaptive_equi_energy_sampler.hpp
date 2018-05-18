#ifndef MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_
#define MCMC_ADAPTIVE_EQUI_ENERGY_SAMPLER_HPP_

#include <functional>
#include <math.h>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class adaptive_equi_energy_sampler
{
public:
  explicit adaptive_equi_energy_sampler  (
    const std::function<density_type(const state_type&)>& log_target_density_function,
    const covariance_matrix_type&                         covariance_matrix          ,
    const density_type                                    scale                      = density_type(1),
    const proposal_distribution_type&                     proposal_distribution      = proposal_distribution_type())
  : log_target_density_function_(log_target_density_function)
  , covariance_matrix_          ((std::pow(scale, 2) * covariance_matrix).llt().matrixLLT())
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  , current_density_            (0)
  {

  }
  adaptive_equi_energy_sampler           (const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler           (      adaptive_equi_energy_sampler&& temp) = default;
  virtual ~adaptive_equi_energy_sampler  ()                                          = default;
  adaptive_equi_energy_sampler& operator=(const adaptive_equi_energy_sampler&  that) = default;
  adaptive_equi_energy_sampler& operator=(      adaptive_equi_energy_sampler&& temp) = default;

  void       setup(const state_type& state)
  {
    current_density_ = log_target_density_function_(state);
  }
  state_type apply(const state_type& state)
  {
    return state;
  }
  
protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_;
  covariance_matrix_type                                                covariance_matrix_          ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
  density_type                                                          current_density_            ;
};
}

#endif