#ifndef MCMC_MARKOV_CHAIN_HPP_
#define MCMC_MARKOV_CHAIN_HPP_

#include <fstream>
#include <string>
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

  template<typename update_strategy_type, typename... argument_types>
  void                           update       (update_strategy_type& update_strategy, argument_types&&... arguments)
  {
    state_history_.push_back(update_strategy.apply(state_history_[state_history_.size() - 1], arguments...));
  }
  const state_type&              state        () const
  {
    return state_history_.back();
  }
  const std::vector<state_type>& state_history() const
  {
    return state_history_;
  }
  void                           to_csv       (const std::string& filepath)
  {
    std::ofstream file(filepath);
    for (auto& state : state_history_)
    {
      auto size = state.size();
      for (auto i = 0; i < size; ++i)
      {
        file << state[i];
        if (i != size - 1) 
          file << ", ";
      }
      file << "\n";
    }
  }

protected:
  // The states prior to the last state have no effect on the update (as per definition of a Markov chain). 
  // They are nevertheless stored for retrospection purposes.
  std::vector<state_type> state_history_;
};
}

#endif
