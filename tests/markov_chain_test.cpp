#include "catch.hpp"

#include <algorithm>
#include <array>

#include <mcmc/markov_chain.hpp>

TEST_CASE("Markov chain is tested.", "[mcmc::markov_chain]") 
{
  GIVEN("A Markov Chain with four floating point states with initial probabilities of 0.1, 0.2, 0.3, 0.4 respectively.")
  {
    std::array<float, 4>                     initial_state {0.1F, 0.2F, 0.3F, 0.4F};
    mcmc::markov_chain<std::array<float, 4>> markov_chain  (initial_state);

    THEN("The state history size should be equal to one.")
    {
      REQUIRE(markov_chain.state_history().size() == 1);
    }
    THEN("The state should be equal to the initial state.")
    {
      REQUIRE(markov_chain.state() == initial_state);
    }
    THEN("The last entry in the state history should be equal to the initial state.")
    {
      REQUIRE(markov_chain.state_history().back() == initial_state);
    }

    WHEN("The state is updated using a trivial update strategy which only inverts the state.")
    {
      struct state_inversion_strategy
      {
        static std::array<float, 4> apply(std::array<float, 4> state)
        {
          std::reverse(state.begin(), state.end());
          return state;
        }
      };
      markov_chain.update<state_inversion_strategy>();

      auto inverted_state = initial_state;
      std::reverse(inverted_state.begin(), inverted_state.end());

      THEN("The state history size should be equal to two.")
      {
        REQUIRE(markov_chain.state_history().size() == 2);
      }
      THEN("The state should be equal to the inverted state.")
      {
        REQUIRE(markov_chain.state() == inverted_state);
      }
      THEN("The last entry in the state history should be equal to the inverted state.")
      {
        REQUIRE(markov_chain.state_history().back() == inverted_state);
      }
    }
  }
}