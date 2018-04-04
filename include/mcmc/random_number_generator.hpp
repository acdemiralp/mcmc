#ifndef MCMC_RANDOM_NUMBER_GENERATOR_HPP_
#define MCMC_RANDOM_NUMBER_GENERATOR_HPP_

#include <functional>
#include <random>
#include <utility>

template<typename distribution_type = std::uniform_real_distribution<double>>
class random_number_generator
{
public:
  using result_type = typename distribution_type::result_type;

  template<typename... argument_types>
  explicit random_number_generator  (argument_types&&...      arguments   ) 
  : mersenne_twister_(random_device_()), distribution_(std::forward<argument_types>(arguments)...)
  {
    
  }
  explicit random_number_generator  (const distribution_type& distribution) 
  : mersenne_twister_(random_device_()), distribution_(distribution)
  {
    
  }
  random_number_generator           (const random_number_generator&  that) = default;
  random_number_generator           (      random_number_generator&& temp) = default;
  virtual ~random_number_generator  ()                                     = default;
  random_number_generator& operator=(const random_number_generator&  that) = default;
  random_number_generator& operator=(      random_number_generator&& temp) = default;

  result_type                  generate()
  {
    return distribution_(mersenne_twister_);
  }
  std::function<result_type()> function()
  {
    return std::bind(&random_number_generator<distribution_type>::generate, this);
  }

protected:
  std::random_device random_device_   ;
  std::mt19937       mersenne_twister_;  
  distribution_type  distribution_    ;
};

#endif