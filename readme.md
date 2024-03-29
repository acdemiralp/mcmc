## Getting Started ##
Copy the include folder to your project.

## Example Usage (Random Walk a Normal distribution) ##
```cpp
#include <mcmc/samplers/random_walk_metropolis_hastings_sampler.hpp>
#include <mcmc/markov_chain.hpp>

void main()
{
  Eigen::VectorXf initial_state(1);
  initial_state[0] = 450.0f;
  
  Eigen::MatrixXf covariance_matrix(1, 1);
  covariance_matrix.setIdentity();

  mcmc::random_walk_metropolis_hastings_sampler<Eigen::VectorXf, Eigen::MatrixXf> sampler(
    [ ] (const Eigen::VectorXf& state)
    {
      return normal_distribution_density(500.0f, 1.0f, state[0]);
    },
    covariance_matrix, 
    100.0f);

  mcmc::markov_chain<Eigen::VectorXf> markov_chain(initial_state);
  for(auto i = 0; i < 100000; ++i)
  {
    markov_chain.update(sampler);
    std::cout << markov_chain.state().format(Eigen::IOFormat()) << "\n";
  }
}
```
See the tests for the details and usage of individual samplers.

## Notes ##
- All kernel functions should return logarithmic scale values unless explicitly stated otherwise.
- [Boost/Math](https://www.boost.org/doc/libs/1_36_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/nmp.html#math.dist.pdf) provides probability density functions for common distributions.

## Coverage ##
The following list is based on [Michael Clark's MCMC algorithms directory](https://m-clark.github.io/docs/ld_mcmc/index_onepage.html) with a few extensions.
Please feel free to add any missing samplers.
- [ ] Adaptive Directional Metropolis-within-Gibbs
- [x] Adaptive Equi-Energy Sampler
- [ ] Adaptive Griddy-Gibbs
- [ ] Adaptive Hamiltonian Monte Carlo
- [ ] Adaptive Metropolis
- [ ] Adaptive Metropolis-within-Gibbs
- [ ] Adaptive-Mixture Metropolis
- [ ] Affine-Invariant Ensemble Sampler
- [ ] Automated Factor Slice Sampler
- [ ] Componentwise Hit-And-Run Metropolis
- [ ] Delayed Rejection Adaptive Metropolis
- [ ] Delayed Rejection Metropolis
- [x] Differential Evolution Markov Chain
- [ ] Elliptical Slice Sampler
- [ ] Equi-Energy Sampler
- [x] Gibbs Sampler
- [ ] Griddy-Gibbs
- [x] Hamiltonian Monte Carlo
- [ ] Hamiltonian Monte Carlo with Dual-Averaging
- [ ] Hessian-Hamiltonian Monte Carlo
- [ ] Hit-And-Run Metropolis
- [x] Independence Metropolis
- [ ] Interchain Adaptation
- [x] Metropolis-Adjusted Langevin Algorithm
- [ ] Metropolis-Coupled Markov Chain Monte Carlo
- [x] Metropolis-Hastings 1953
- [ ] Metropolis-within-Gibbs
- [ ] Multiple-Try Metropolis
- [ ] No-U-Turn Sampler
- [ ] Oblique Hyperrectangle Slice Sampler
- [ ] Preconditioned Crank-Nicolson
- [ ] Random Dive Metropolis-Hastings
- [x] Random-Walk Metropolis
- [ ] Reflective Slice Sampler
- [ ] Refractive Sampler
- [ ] Reversible-Jump
- [x] Riemannian Manifold Hamiltonian Monte Carlo
- [ ] Riemannian Manifold Metropolis Adjusted Langevin Algorithm
- [ ] Robust Adaptive Metropolis
- [ ] Sequential Adaptive Metropolis-within-Gibbs
- [ ] Sequential Metropois-within-Gibbs
- [ ] Slice Sampler
- [ ] Stochastic Gradient Fisher Scoring
- [ ] Stochastic Gradient Langevin Dynamics
- [ ] Stochastic Gradient Nose-Hoover Thermostat
- [ ] T-Walk
- [ ] Tempered Hamiltonian Monte Carlo
- [ ] Univariate Eigenvector Slice Sampler
- [ ] Updating Sequential Adaptive Metropolis-within-Gibbs
- [ ] Updating Sequential Metropolis-within-Gibbs

The following list contains stochastic Variational Inference (VI) samplers.
- [ ] Black Box Sampler
- [ ] Doubly Stochastic Sampler
- [ ] Nonparametric Sampler
- [ ] Particle Mirror Descent Sampler
- [x] Stein Variational Gradient Descent

## Future Work ##
- More samplers.
- Quasi-random number generators (van der Corput-Halton sequence, Faure-Niederreiter sequence, Hammersley set, Poisson disk sampling, Sobol low-discrepancy sequence, ...).

## Acknowledgements ##
Several samplers in this library are inspired by [Keith O'Hara's library with the same name](https://github.com/kthohr/mcmc).
