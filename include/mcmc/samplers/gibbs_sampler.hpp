#ifndef MCMC_GIBBS_SAMPLER_HPP_
#define MCMC_GIBBS_SAMPLER_HPP_

#include <functional>
#include <random>

#include <external/Eigen/Core>

namespace mcmc
{
template<
  typename density_type               = float,
  typename state_type                 = Eigen::VectorXf,
  typename proposal_distribution_type = std::normal_distribution<density_type>>
class gibbs_sampler
{
public:
  explicit gibbs_sampler(
    const std::function<density_type(const state_type&, const std::size_t)>& log_conditional_density_function)
  : log_conditional_density_function_(log_conditional_density_function)
  {

  }
  gibbs_sampler           (const gibbs_sampler&  that) = default;
  gibbs_sampler           (      gibbs_sampler&& temp) = default;
  virtual ~gibbs_sampler  ()                           = default;
  gibbs_sampler& operator=(const gibbs_sampler&  that) = default;
  gibbs_sampler& operator=(      gibbs_sampler&& temp) = default;
  
  state_type apply (const state_type& state)
  {
    state_type next_state = state;
    for(auto i = 0; i < next_state.size(); ++i)
      next_state[i] = log_conditional_density_function_(next_state, i);
    return next_state;
  }

protected:
  std::function<density_type(const state_type&, const std::size_t)> log_conditional_density_function_;
};
}

#endif
