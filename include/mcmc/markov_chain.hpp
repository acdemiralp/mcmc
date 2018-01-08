#ifndef MCMC_MARKOV_CHAIN_HPP_
#define MCMC_MARKOV_CHAIN_HPP_

#include <vector>

namespace mcmc
{
template<typename state_type>
class markov_chain
{
public:
  explicit markov_chain  (state_type initial_state ) : state_history_{initial_state}
  {
    
  }
  markov_chain           (const markov_chain&  that) = default;
  markov_chain           (      markov_chain&& temp) = default;
  virtual ~markov_chain  ()                          = default;
  markov_chain& operator=(const markov_chain&  that) = default;
  markov_chain& operator=(      markov_chain&& temp) = default;

  template<typename update_strategy, typename... argument_types>
  void                           update       (argument_types&&... arguments)
  {
    state_history_.push_back(update_strategy::apply(state_history_[state_history_.size() - 1], arguments...));
  }
  const state_type&              state        () const
  {
    return state_history_.back();
  }
  const std::vector<state_type>& state_history() const
  {
    return state_history_;
  }

protected:
  // The states prior to the last state have no effect on the update (as per definition of a Markov chain). 
  // They are nevertheless stored for retrospection purposes.
  std::vector<state_type> state_history_;
};
}

#endif
