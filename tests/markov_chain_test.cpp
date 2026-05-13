#include "catch.hpp"

#include <algorithm>
#include <array>
#include <optional>

#include <mcmc/markov_chain.hpp>

TEST_CASE("Markov chain is tested.", "[mcmc::markov_chain]") 
{
  GIVEN("A Markov Chain with four floating point states with initial probabilities of 0.1, 0.2, 0.3, 0.4 respectively.")
  {
    std::array<float, 4>                     initial_state {0.1F, 0.2F, 0.3F, 0.4F};
    mcmc::markov_chain<std::array<float, 4>> markov_chain  (initial_state);

    THEN("The state should be equal to the initial state.")
    {
      REQUIRE(markov_chain.state() == initial_state);
    }
    THEN("Subscribing can emit the current state immediately.")
    {
      std::optional<std::array<float, 4>> observed_state;
      markov_chain.subscribe([&](const std::array<float, 4>& state) { observed_state = state; }, true);

      REQUIRE(observed_state.has_value());
      REQUIRE(observed_state.value() == initial_state);
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
      const  state_inversion_strategy strategy;
      markov_chain.update(strategy);

      auto inverted_state = initial_state;
      std::reverse(inverted_state.begin(), inverted_state.end());

      THEN("The state should be equal to the inverted state.")
      {
        REQUIRE(markov_chain.state() == inverted_state);
      }
      THEN("A subscriber receives the updated state.")
      {
        std::optional<std::array<float, 4>> observed_state;
        mcmc::markov_chain<std::array<float, 4>> observed_chain(initial_state);
        observed_chain.subscribe([&](const std::array<float, 4>& state) { observed_state = state; });
        observed_chain.update(strategy);

        REQUIRE(observed_state.has_value());
        REQUIRE(observed_state.value() == inverted_state);
      }
    }
  }
}
