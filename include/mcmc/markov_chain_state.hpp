#ifndef MCMC_MARKOV_CHAIN_STATE_HPP_
#define MCMC_MARKOV_CHAIN_STATE_HPP_

#include <array>
#include <cstddef>

namespace mcmc
{
template <typename probability_type, std::size_t state_space_size>
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
  std::array<probability_type, state_space_size> transition_vector_;
};
}

#endif
