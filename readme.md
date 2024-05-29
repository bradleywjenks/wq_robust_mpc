# wq_robust_mpc
Code implementation for robust model predictive control (MPC) of disinfectant residuals in water networks.

### To-do:
- [ ] Add water quality constraints (with implicit-upwind discretization scheme) to `optimize_hydraulic_&_wq.jl` problem formulation 
- [ ] Include forest-core decomposition algorithm as a preprocessing step to fix hydraulic variables on branch links
- [ ] Register for Imperial's HPC and try running larger problems
- [ ] Formulate uncertainty sets for random variables (e.g. demands, disinfectant decay rates) 