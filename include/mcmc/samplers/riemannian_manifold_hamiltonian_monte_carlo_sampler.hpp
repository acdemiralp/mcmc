#ifndef MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_
#define MCMC_RIEMANNIAN_MANIFOLD_HAMILTONIAN_MONTE_CARLO_SAMPLER_HPP_

#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include <random>

#include <external/Eigen/Dense>
#include <external/unsupported/Eigen/CXX11/Tensor>

#include <mcmc/random_number_generator.hpp>

namespace mcmc
{
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename matrix_type                = Eigen::MatrixXf,
  typename tensor_type                = Eigen::Tensor<density_type, 3>,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class riemannian_manifold_hamiltonian_monte_carlo_sampler
{
public:
  explicit riemannian_manifold_hamiltonian_monte_carlo_sampler(
    const std::function<density_type(const state_type&, state_type* )>& log_target_density_function,
    const std::function<matrix_type (const state_type&, tensor_type*)>& tensor_function            ,
    const std::uint32_t                                                 leap_steps                 = 1u,
    const std::uint32_t                                                 fixed_point_steps          = 5u,
    const density_type                                                  step_size                  = density_type(1),
    const proposal_distribution_type&                                   proposal_distribution      = proposal_distribution_type())
  : tensor_function_  (tensor_function)
  , leap_steps_       (leap_steps)
  , fixed_point_steps_(fixed_point_steps)
  , step_size_        (step_size)
  , proposal_rng_     (proposal_distribution)
  , acceptance_rng_   (0, 1)
  , constant_term_    (0)
  , potential_energy_ (0)
  , kinetic_energy_   (0)
  {
    log_target_density_function_ = [=] (const state_type& state) -> density_type
    {
      return log_target_density_function(state, nullptr);
    };
    log_momentum_function_       = [=] (const state_type& state, const state_type& momentum, const tensor_type& tensor_derivative, const matrix_type& inverse_tensor_matrix) -> state_type
    {
      state_type gradients(state.size());
      log_target_density_function(state, &gradients);
      for (auto i = 0; i < gradients.size(); ++i) 
      {
        auto dimensions    = tensor_derivative.dimensions();
        auto sliced_tensor = tensor_derivative.slice(
          std::array<std::size_t, 3>{std::size_t(0),             std::size_t(0),             std::size_t(i)}, 
          std::array<std::size_t, 3>{std::size_t(dimensions[0]), std::size_t(dimensions[1]), std::size_t(1)});

        Eigen::MatrixXf temp = inverse_tensor_matrix * tensor_to_matrix(sliced_tensor.expression(), dimensions[0], dimensions[1]);
        gradients[i]         = -gradients[i] + density_type(0.5) * (temp.trace() - (momentum.transpose() * temp * inverse_tensor_matrix * momentum));
      }
      return step_size_ * gradients / density_type(2);
    };
  }
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler           (      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;
  virtual ~riemannian_manifold_hamiltonian_monte_carlo_sampler  ()                                                                 = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(const riemannian_manifold_hamiltonian_monte_carlo_sampler&  that) = default;
  riemannian_manifold_hamiltonian_monte_carlo_sampler& operator=(      riemannian_manifold_hamiltonian_monte_carlo_sampler&& temp) = default;
  
  void       setup(const state_type& state)
  {
    constant_term_                  = density_type(0.5) * state.size() * std::log(2.0 * M_PI);

    tensor_                         = tensor_type();
    tensor_matrix_                  = tensor_function_(state, &tensor_);
    inverse_tensor_matrix_          = tensor_matrix_.inverse();
    
    previous_tensor_                = tensor_;
    previous_tensor_matrix_         = tensor_matrix_;
    previous_inverse_tensor_matrix_ = inverse_tensor_matrix_;
    
    potential_energy_               = constant_term_ - log_target_density_function_(state) + std::log(tensor_matrix_.determinant());
  }
  state_type apply(const state_type& state)
  {
    state_type random   = proposal_rng_.template generate<state_type>(state.size());
    state_type momentum = previous_tensor_matrix_.llt().matrixLLT() * random;
    kinetic_energy_     = momentum.dot(previous_inverse_tensor_matrix_ * momentum) / density_type(2);
    
    state_type next_state = state;
    for (auto i = 0u; i < leap_steps_; ++i)
    {
      auto temp_momentum = momentum;
      for (auto j = 0u; j < fixed_point_steps_; ++j)
        temp_momentum = momentum + log_momentum_function_(next_state, temp_momentum, previous_tensor_, previous_inverse_tensor_matrix_);
      momentum = temp_momentum;

      auto temp_state = next_state;
      for (auto j = 0u; j < fixed_point_steps_; ++j)
      {
        inverse_tensor_matrix_ = tensor_function_(temp_state, nullptr).inverse();
        temp_state             = next_state + density_type(0.5) * step_size_ * (previous_inverse_tensor_matrix_ + inverse_tensor_matrix_) * momentum;
      }
      next_state = temp_state;

      tensor_matrix_         = tensor_function_(next_state, &tensor_);
      inverse_tensor_matrix_ = tensor_matrix_.inverse();

      momentum += log_momentum_function_(next_state, momentum, tensor_, inverse_tensor_matrix_);
    }

    const density_type potential_energy = constant_term_ - log_target_density_function_(next_state) + density_type(0.5) * std::log(tensor_matrix_.determinant());
    const density_type kinetic_energy   = momentum.dot(inverse_tensor_matrix_ * momentum) / density_type(2);

    if (std::exp(std::min(density_type(0), - potential_energy - kinetic_energy + potential_energy_ + kinetic_energy_)) < acceptance_rng_.generate())
      return state;

    potential_energy_               = potential_energy      ;
    kinetic_energy_                 = kinetic_energy        ;
    previous_tensor_                = tensor_               ;
    previous_tensor_matrix_         = tensor_matrix_        ;
    previous_inverse_tensor_matrix_ = inverse_tensor_matrix_;

    return next_state;
  }

protected:
  template <std::size_t rank>
  static auto tensor_to_matrix(const Eigen::Tensor<density_type, rank>& tensor, const std::size_t rows, const std::size_t columns)
  {
    return Eigen::Map<const Eigen::Matrix<density_type, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), rows, columns);
  }
  template <typename... dimension_types>
  static auto matrix_to_tensor(const Eigen::Matrix<density_type, Eigen::Dynamic, Eigen::Dynamic>& matrix, dimension_types... dimensions)
  {
    return Eigen::TensorMap<Eigen::Tensor<const density_type, sizeof... (dimension_types)>>(matrix.data(), {dimensions...});
  }

  std::function<density_type(const state_type&)>                                                            log_target_density_function_   ;
  std::function<state_type  (const state_type&, const state_type&, const tensor_type&, const matrix_type&)> log_momentum_function_         ;
  std::function<matrix_type (const state_type&, tensor_type*)>                                              tensor_function_               ;

  std::uint32_t                                                                                             leap_steps_                    ;
  std::uint32_t                                                                                             fixed_point_steps_             ;
  density_type                                                                                              step_size_                     ;

  random_number_generator<proposal_distribution_type>                                                       proposal_rng_                  ;
  random_number_generator<std::uniform_real_distribution<density_type>>                                     acceptance_rng_                ;
  
  density_type                                                                                              constant_term_                 ;
  density_type                                                                                              potential_energy_              ;
  density_type                                                                                              kinetic_energy_                ;

  tensor_type                                                                                               previous_tensor_               ;
  matrix_type                                                                                               previous_tensor_matrix_        ;
  matrix_type                                                                                               previous_inverse_tensor_matrix_;

  tensor_type                                                                                               tensor_                        ;
  matrix_type                                                                                               tensor_matrix_                 ;
  matrix_type                                                                                               inverse_tensor_matrix_         ;
};
}

#endif