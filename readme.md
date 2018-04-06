## Coverage ##
The following list is based on https://m-clark.github.io/docs/ld_mcmc/index_onepage.html with a few extensions. Please inform if anything is missing.
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
- [x] Equi-Energy Sampler
- [x] Gibbs Sampler
- [ ] Griddy-Gibbs
- [x] Hamiltonian Monte Carlo
- [ ] Hamiltonian Monte Carlo with Dual-Averaging
- [ ] Hit-And-Run Metropolis
- [x] Independence Metropolis
- [ ] Interchain Adaptation
- [x] Metropolis-Adjusted Langevin Algorithm
- [ ] Metropolis-Coupled Markov Chain Monte Carlo
- [ ] Metropolis-within-Gibbs
- [ ] Multiple-Try Metropolis
- [ ] No-U-Turn Sampler
- [ ] Oblique Hyperrectangle Slice Sampler
- [ ] Preconditioned Crank-Nicolson
- [ ] Random Dive Metropolis-Hastings
- [ ] Random-Walk Metropolis
- [ ] Reflective Slice Sampler
- [ ] Refractive Sampler
- [ ] Reversible-Jump
- [x] Riemannian Manifold Hamiltonian Monte Carlo
- [ ] Riemannian Manifold Metropolis Adjusted Langevin Algorithm
- [ ] Robust Adaptive Metropolis
- [ ] Sequential Adaptive Metropolis-within-Gibbs
- [ ] Sequential Metropois-within-Gibbs
- [ ] Slice Sampler
- [ ] Stochastic Gradient Langevin Dynamics
- [ ] T-Walk
- [ ] Tempered Hamiltonian Monte Carlo
- [ ] Univariate Eigenvector Slice Sampler
- [ ] Updating Sequential Adaptive Metropolis-within-Gibbs
- [ ] Updating Sequential Metropolis-within-Gibbs

## Notes ##
See the tests for usage of the individual samplers.
All kernel functions are expected to return logarithmic scale values.

## Acknowledgements ##
The majority of the samplers in this library are inspired by Keith O'Hara's library with the same name: https://github.com/kthohr/mcmc
