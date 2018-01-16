#include "catch.hpp"

#include <cstddef>
#include <fstream>
#include <random>

#include <mcmc/metropolis_hastings_sampler.hpp>

#define M_LN_SQRT_2PI	0.918938533204672741780329736406
#define M_1_SQRT_2PI	0.398942280401432677939946059934

double normal_distribution_density(double x, const double mu = 0.0, const double sigma = 1.0, const int give_log = false)
{
  x = fabs((x - mu) / sigma);
  return give_log ? -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma)) : M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;
}

TEST_CASE("Metropolis-Hastings sampler is tested.", "[mcmc::metropolis_hastings_sampler]") 
{
  //mcmc::metropolis_hastings_sampler<float> sampler                            ;

  std::random_device                       random_device                        ;
  std::mt19937                             mersenne_twister(random_device())    ;
  std::uniform_real_distribution<float>    uniform_real_distribution(0.0F, 1.0F);

  std::random_device                       random_device_2                      ;
  std::mt19937                             mersenne_twister_2(random_device_2());
  std::normal_distribution<float>          normal_distribution(0.0F, 5.0F)      ;

  std::vector<std::size_t>                 samples(50000)                       ;
  samples[0] = 110;
  for(auto i = 1; i < samples.size(); ++i)
  {
    const auto proposal = samples[i - 1] + normal_distribution(mersenne_twister_2);
    samples[i] = normal_distribution_density(proposal, 100, 15) / normal_distribution_density(samples[i - 1], 100, 15) > uniform_real_distribution(mersenne_twister)
      ? samples[i] = proposal
      : samples[i] = samples[i - 1];
  }

  std::ofstream file("samples.txt");
  for(auto i = 0; i < samples.size(); ++i)
    file << samples[i] << std::endl;
  file.close();
}