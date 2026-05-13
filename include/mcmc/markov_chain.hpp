#ifndef MCMC_MARKOV_CHAIN_HPP_
#define MCMC_MARKOV_CHAIN_HPP_

#include <functional>
#include <utility>

namespace mcmc
{
template<typename state_type>
class markov_chain
{
public:
  explicit markov_chain  (
    state_type                                   initial_state ,
    std::function<void(const state_type& state)> state_callback = {})
  : state_         (std::move(initial_state))
  , state_callback_(std::move(state_callback))
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
    state_ = update_strategy.apply(state_, std::forward<argument_types>(arguments)...);
    if(state_callback_)
      state_callback_(state_);
  }
  const state_type&              state        () const
  {
    return state_;
  }
  void                           subscribe    (
    std::function<void(const state_type& state)> state_callback,
    bool                                         emit_current_state = false)
  {
    state_callback_ = std::move(state_callback);
    if (emit_current_state && state_callback_)
      state_callback_(state_);
  }
  void                           unsubscribe  ()
  {
    state_callback_ = {};
  }

protected:
  state_type                                   state_         ;
  std::function<void(const state_type& state)> state_callback_;
};
}

#endif
