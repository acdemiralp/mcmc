#ifndef MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <algorithm>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Cholesky>
#include <external/Eigen/Core>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename state_type                 = Eigen::VectorXf,
  typename precondition_matrix_type   = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class hamiltonian_monte_carlo_sampler
{
public:
  template<typename... arguments_type>
  explicit hamiltonian_monte_carlo_sampler(
    const std::function<float(const state_type&, state_type&)>& kernel_function       ,
    const precondition_matrix_type&                             precondition_matrix   ,
    const std::uint32_t                                         leaps                 = 1,
    const float                                                 step_size             = 1.0f,
    const proposal_distribution_type&                           proposal_distribution = proposal_distribution_type())
  : kernel_function_            (kernel_function)
  , precondition_matrix_        (precondition_matrix.llt().matrixLLT())
  , inverse_precondition_matrix_(precondition_matrix.inverse())
  , leaps_                      (leaps)
  , step_size_                  (step_size)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0.0f, 1.0f)
  , last_log_density_           (0.0f)
  {
    momentum_function_ = [ ] ()
    {
      
    };
  }
  hamiltonian_monte_carlo_sampler           (const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler           (      hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~hamiltonian_monte_carlo_sampler  ()                                             = default;
  hamiltonian_monte_carlo_sampler& operator=(const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler& operator=(      hamiltonian_monte_carlo_sampler&& temp) = default;

  state_type apply(const state_type& state, state_type& gradients)
  {
    return state;
  }

protected:
  std::function<float(const state_type&, state_type&)>           kernel_function_            ;
  std::function<float(const state_type&, state_type&)>           momentum_function_          ;
  precondition_matrix_type                                       precondition_matrix_        ;
  precondition_matrix_type                                       inverse_precondition_matrix_;
  std::uint32_t                                                  leaps_                      ;
  float                                                          step_size_                  ;
  random_number_generator<proposal_distribution_type>            proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<float>> acceptance_rng_             ;
  float                                                          last_log_density_           ;
};
}

#endif