#ifndef MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#include <cstdint>
#include <functional>
#include <math.h>

#include <external/Eigen/Dense>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename precondition_matrix_type   = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class hamiltonian_monte_carlo_sampler
{
public:
  explicit hamiltonian_monte_carlo_sampler(
    const std::function<density_type(const state_type&, state_type*)>& log_target_density_function,
    const precondition_matrix_type&                                    precondition_matrix        ,
    const std::uint32_t                                                leap_steps                      = 1u,
    const density_type                                                 step_size                  = density_type(1),
    const proposal_distribution_type&                                  proposal_distribution      = proposal_distribution_type())
  : precondition_matrix_        (precondition_matrix.llt().matrixLLT())
  , inverse_precondition_matrix_(precondition_matrix.inverse())
  , leap_steps_                 (leap_steps)
  , step_size_                  (step_size)
  , proposal_rng_               (proposal_distribution)
  , acceptance_rng_             (0, 1)
  , potential_energy_           (0)
  , kinetic_energy_             (0)
  {
    log_target_density_function_ = [=] (const state_type& state) -> density_type
    {
      return log_target_density_function(state, nullptr);
    };
    log_momentum_function_       = [=] (const state_type& state, state_type& momentum) -> state_type
    {
      state_type gradients(state.size());
      log_target_density_function(state, &gradients);
      return momentum + step_size_ * gradients / density_type(2);
    };
  }
  hamiltonian_monte_carlo_sampler           (const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler           (      hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~hamiltonian_monte_carlo_sampler  ()                                             = default;
  hamiltonian_monte_carlo_sampler& operator=(const hamiltonian_monte_carlo_sampler&  that) = default;
  hamiltonian_monte_carlo_sampler& operator=(      hamiltonian_monte_carlo_sampler&& temp) = default;

  void       setup(const state_type& state)
  {
    potential_energy_ = -log_target_density_function_(state);
  }
  state_type apply(const state_type& state)
  {
    state_type random   = proposal_rng_.template generate<state_type>(state.size());
    state_type momentum = precondition_matrix_ * random;
    kinetic_energy_     = momentum.dot(inverse_precondition_matrix_ * momentum) / density_type(2);
    
    auto next_state = state;
    for(auto i = 0u; i < leap_steps_; ++i)
    {
      momentum    = log_momentum_function_(next_state, momentum);
      next_state += step_size_ * inverse_precondition_matrix_ * momentum;
      momentum    = log_momentum_function_(next_state, momentum);
    }

    const density_type potential_energy = -log_target_density_function_(next_state);
    const density_type kinetic_energy   = momentum.dot(inverse_precondition_matrix_ * momentum) / density_type(2);
    
    if (std::exp(std::min(density_type(0), - potential_energy - kinetic_energy + potential_energy_ + kinetic_energy_)) < acceptance_rng_.generate())
      return state;

    potential_energy_ = potential_energy;
    kinetic_energy_   = kinetic_energy  ;
    return next_state;
  }

protected:
  std::function<density_type(const state_type&)>                        log_target_density_function_;
  std::function<state_type  (const state_type&, state_type&)>           log_momentum_function_      ;
  precondition_matrix_type                                              precondition_matrix_        ;
  precondition_matrix_type                                              inverse_precondition_matrix_;
  std::uint32_t                                                         leap_steps_                 ;
  density_type                                                          step_size_                  ;
  random_number_generator<proposal_distribution_type>                   proposal_rng_               ;
  random_number_generator<std::uniform_real_distribution<density_type>> acceptance_rng_             ;
  density_type                                                          potential_energy_           ;
  density_type                                                          kinetic_energy_             ;
};
}

#endif