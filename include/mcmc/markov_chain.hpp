#ifndef MCMC_MARKOV_CHAIN_HPP_
#define MCMC_MARKOV_CHAIN_HPP_

namespace mcmc
{
class markov_chain
{
public:
  markov_chain           ()                          = default;
  markov_chain           (const markov_chain&  that) = default;
  markov_chain           (      markov_chain&& temp) = default;
  virtual ~markov_chain  ()                          = default;
  markov_chain& operator=(const markov_chain&  that) = default;
  markov_chain& operator=(      markov_chain&& temp) = default;

protected:

};
}

#endif
