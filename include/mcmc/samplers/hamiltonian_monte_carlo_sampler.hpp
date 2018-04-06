#ifndef MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <algorithm>
#include <functional>
#include <math.h>
#include <random>

#include <external/Eigen/Dense>

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
    const std::function<float(const state_type&, state_type*)>& kernel_function       ,
    const precondition_matrix_type&                             precondition_matrix   ,
    const std::uint32_t                                         leaps                 = 1,
    const float                                                 step_size             = 1.0f,
    const proposal_distribution_type&                           proposal_distribution = proposal_distribution_type())
  : precondition_matrix_        (precondition_matrix.llt().matrixLLT())
  , inverse_precondition_matrix_(precondition_matrix.inverse())
  , leaps_                      (leaps)
  , step_size_                  (step_size)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0.0f, 1.0f)
  , potential_energy_           (0.0f)
  , kinetic_energy_             (0.0f)
  {
    log_kernel_function_ = [=] (const state_type& state) -> float
    {
      return kernel_function(state, nullptr);
    };
    momentum_function_   = [=] (const state_type& state, state_type& momentum) -> state_type
    {
      state_type gradients(state.size());
      kernel_function(state, &gradients);
      return momentum + step_size_ * gradients / 2.0;
    };
  }
  hamiltonian_monte_carlo_sampler           (const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler           (      hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~hamiltonian_monte_carlo_sampler  ()                                             = default;
  hamiltonian_monte_carlo_sampler& operator=(const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler& operator=(      hamiltonian_monte_carlo_sampler&& temp) = default;

  state_type apply(const state_type& state, state_type* gradients)
  {
    state_type random_vector(state.size());
    std::generate_n(&random_vector[0], random_vector.size(), proposal_rng_.function());

    state_type momentum = precondition_matrix_ * random_vector;
    kinetic_energy_     = momentum.dot(inverse_precondition_matrix_ * momentum) / 2.0;
    
    auto next_state = state;
    for(auto i = 0; i < leaps_; ++i)
    {
      momentum    = momentum_function_(next_state, momentum);
      next_state += step_size_ * inverse_precondition_matrix_ * momentum;
      momentum    = momentum_function_(next_state, momentum);
    }

    const float potential_energy = -log_kernel_function_(next_state);
    const float kinetic_energy   = momentum.dot(inverse_precondition_matrix_ * momentum) / 2.0;

    if (std::exp(std::min(0.0f, - potential_energy - kinetic_energy + potential_energy_ + kinetic_energy_)) < acceptance_rng_.generate())
      return state;

    potential_energy_ = potential_energy;
    kinetic_energy_   = kinetic_energy  ;
    return next_state;
  }

protected:
  std::function<float(const state_type&)>                        log_kernel_function_        ;
  std::function<state_type(const state_type&, state_type&)>      momentum_function_          ;
  precondition_matrix_type                                       precondition_matrix_        ;
  precondition_matrix_type                                       inverse_precondition_matrix_;
  std::uint32_t                                                  leaps_                      ;
  float                                                          step_size_                  ;
  random_number_generator<proposal_distribution_type>            proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<float>> acceptance_rng_             ;
  float                                                          potential_energy_           ;
  float                                                          kinetic_energy_             ;
};
}

#endif