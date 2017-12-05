#ifndef MCMC_MARKOV_CHAIN_STATE_HPP_
#define MCMC_MARKOV_CHAIN_STATE_HPP_

namespace mcmc
{
class markov_chain_state
{
public:
  markov_chain_state           ()                                = default;
  markov_chain_state           (const markov_chain_state&  that) = default;
  markov_chain_state           (      markov_chain_state&& temp) = default;
  virtual ~markov_chain_state  ()                                = default;
  markov_chain_state& operator=(const markov_chain_state&  that) = default;
  markov_chain_state& operator=(      markov_chain_state&& temp) = default;

protected:

};
}

#endif
