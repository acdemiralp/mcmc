#ifndef MCMC_GIBBS_SAMPLER_HPP_
#define MCMC_GIBBS_SAMPLER_HPP_

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
  typename covariance_matrix_type     = Eigen::MatrixXf,
  typename proposal_distribution_type = std::normal_distribution<float>>
class gibbs_sampler
{
public:
  gibbs_sampler           ()                           = default;
  gibbs_sampler           (const gibbs_sampler&  that) = default;
  gibbs_sampler           (      gibbs_sampler&& temp) = default;
  virtual ~gibbs_sampler  ()                           = default;
  gibbs_sampler& operator=(const gibbs_sampler&  that) = default;
  gibbs_sampler& operator=(      gibbs_sampler&& temp) = default;
  
  void       setup(const state_type& state)
  {

  }
  state_type apply(const state_type& state)
  {
    return state;
  }

protected:
};
}

#endif
