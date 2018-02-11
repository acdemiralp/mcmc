#include "catch.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <random>

#include <mcmc/metropolis_hastings_sampler.hpp>

#define M_LN_SQRT_2PI	0.918938533204672741780329736406
#define M_1_SQRT_2PI	0.398942280401432677939946059934

double normal_distribution_density(double x, const double mu = 0.0, const double sigma = 1.0,
                                   const int give_log = false)
{
  x = fabs((x - mu) / sigma);
  return give_log ? -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma)) : M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;
}

TEST_CASE("Metropolis-Hastings sampler is tested.", "[mcmc::metropolis_hastings_sampler]")
{
  // At the minimum, it is necessary to know how to compute the density of the distribution.

  //mcmc::metropolis_hastings_sampler<float> sampler                            ;

  // The probability of the parameters given the data (posterior), is proportional to, 
  // the probability of the data given the parameters (likelihood) * apriori probability of the parameters (prior).

  // The probability of u, given the data, is proportional to,
  // the probability of the data given u * apriori probability of u.

  // The posterior is proportional to likelihood * prior.

  std::random_device random_device;
  std::mt19937 mersenne_twister(random_device());
  std::uniform_real_distribution<float> uniform_real_distribution(0.0F, 1.0F);

  std::random_device random_device_2;
  std::mt19937 mersenne_twister_2(random_device_2());
  std::normal_distribution<float> normal_distribution(0.0F, 5.0F);

  std::vector<std::size_t> samples(50000);
  samples[0] = 110;
  for (auto i = 1; i < samples.size(); ++i)
  {
    const auto proposal = samples[i - 1] + normal_distribution(mersenne_twister_2);
    samples[i] = normal_distribution_density(proposal, 100, 15) / normal_distribution_density(samples[i - 1], 100, 15) >
                 uniform_real_distribution(mersenne_twister)
                   ? samples[i] = proposal
                   : samples[i] = samples[i - 1];
  }

  std::ofstream file("samples.txt");
  for (auto i = 0; i < samples.size(); ++i)
    file << samples[i] << std::endl;
}





using parameters = std::array<float, 2>; // d, C

struct data
{
  std::size_t present      = 1000;
  std::size_t absent       =  100;
  std::size_t hits         =   85;
  std::size_t false_alarms =   12;
};

auto posterior_density = [ ] (const parameters& parameters, const data& data)
{
  // TODO
  return 0.0F;
};

TEST_CASE("Metropolis Test", "[metropolis]")
{
  std::random_device                          random_device                               ;
  std::mt19937                                mersenne_twister         (random_device  ());
  const std::uniform_real_distribution<float> uniform_real_distribution(0.0F, 1.0F       );

  std::random_device                          random_device_2                             ;
  std::mt19937                                mersenne_twister_2       (random_device_2());
  std::normal_distribution<float>             normal_distribution      (0.0F, 0.1F       ); // Adjust to application or even per parameter.

  data                    data;
  std::vector<parameters> samples(500);
  samples[0] = {0.5F, 1.0F};
  for (auto i = 1; i < samples.size(); ++i)
  {
    auto proposal = samples[i - 1];
    for (auto& sample : proposal)
      sample += normal_distribution(mersenne_twister_2);
    samples[i] = posterior_density(proposal, data) / posterior_density(samples[i - 1], data) > uniform_real_distribution(mersenne_twister)
      ? proposal
      : samples[i - 1];
  }
}

TEST_CASE("Gibbs Test", "[gibbs]")
{
  std::random_device                          random_device                               ;
  std::mt19937                                mersenne_twister         (random_device  ());
  const std::uniform_real_distribution<float> uniform_real_distribution(0.0F, 1.0F       );

  std::random_device                          random_device_2                             ;
  std::mt19937                                mersenne_twister_2       (random_device_2());
  std::normal_distribution<float>             normal_distribution      (0.0F, 0.1F       ); // Adjust to application or even per parameter.

  data                    data;
  std::vector<parameters> samples(500);
  samples[0] = {0.5F, 1.0F};
  for (auto i = 1; i < samples.size(); ++i)
  {
    samples[i] = samples[i - 1];
    for (auto j = 0; j < samples[i].size(); ++j)
    {
      auto proposal = samples[i];
      proposal[j] += normal_distribution(mersenne_twister_2);
      if (posterior_density(proposal, data) / posterior_density(samples[i], data) > uniform_real_distribution(mersenne_twister))
        samples[i] = proposal;
    }
  }
}