#ifndef MCMC_GIBBS_SAMPLER_HPP_
#define MCMC_GIBBS_SAMPLER_HPP_

namespace mcmc
{
class gibbs_sampler
{
public:
  gibbs_sampler           ()                           = default;
  gibbs_sampler           (const gibbs_sampler&  that) = default;
  gibbs_sampler           (      gibbs_sampler&& temp) = default;
  virtual ~gibbs_sampler  ()                           = default;
  gibbs_sampler& operator=(const gibbs_sampler&  that) = default;
  gibbs_sampler& operator=(      gibbs_sampler&& temp) = default;

protected:

};
}

#endif
