# Interview Master Prep — Physics-Aware ML for Aerodynamics & Wind Energy

**Supervisor / interviewer:** Jincheng Zhang (Assistant Prof, Warwick — PINNs & FNOs for wind energy, wakes, turbulence)
**Your two assets:** (1) physics-informed FNO predicting a time-varying Hurst exponent (rough-Heston); (2) MSc on the stochastic FKPP equation (numerics + existence/uniqueness/stability).
**Golden rule:** lead with *methods*, state things *precisely*, and *own the gaps*. Overclaiming to the one person who can check is the only real way to lose this.

---

## 1. The 60-second framing (say this early)

"My background sits exactly at the intersection of this project: a PDE/SPDE side and a neural-operator side. In my MSc I took a stochastic reaction–diffusion PDE — the Fisher-KPP equation — discretised it, validated the solver, and studied existence, uniqueness and stability of its solutions. Separately I built a physics-informed Fourier Neural Operator that learns the solution operator of a rough-volatility SPDE to predict a roughness exponent. The FNO and physics-informed-loss machinery I used is the same machinery you use for turbulence and wake surrogates — so I'm not switching fields, I'm changing the PDE."

---

## 2. What this PhD actually involves (day-to-day)

It's a **computational / ML PhD**, not experimental. Realistic breakdown:

- **Writing & debugging PyTorch** — building PINNs and neural operators, training on Warwick HPC, diagnosing convergence. The bulk of the work.
- **Simulation data** — generating/curating high-fidelity CFD (LES/RANS, likely OpenFOAM or existing datasets) as ground truth; building data pipelines.
- **Embedding physics in the loss** — Navier–Stokes / RANS-closure residuals, autodiff derivatives, balancing physics-vs-data weighting.
- **Validation + UQ** — comparing surrogates to high-fidelity sims and sparse real data (Lidar), attaching calibrated uncertainty.
- **Wind/aero science** — wake modelling, farm power/layout optimisation, possibly control or forecasting.
- **Academic rhythm** — reading, weekly meetings, group talks, papers, some teaching.

**Year one** = onboarding + reproducing a baseline (a PINN for a canonical flow, or his digital-twin pipeline) → then carving out the novel contribution. Saying "I'd expect to reproduce a known result before pushing past it" signals research maturity.

---

## 3. Papers — tiered (skim abstracts + figures only)

**Tier 1 — open these:**
- Zhang et al., *Digital twin of wind farms via physics-informed deep learning* (Energy Conversion & Management, 2023). His flagship — real-time wake modelling via PINNs.
- Zhang et al., *Reconstruction of dynamic wind-turbine wake flow fields from virtual Lidar via PINNs* (2024). Sparse noisy data + 2D Navier–Stokes — your closest bridge.
- Li et al., *Fourier Neural Operator for Parametric PDEs* (ICLR 2021). The FNO.
- Raissi, Perdikaris & Karniadakis, *Physics-Informed Neural Networks* (J. Comp. Physics 2019). PDE residual in the loss.
- Li et al., *Physics-Informed Neural Operator (PINO)* (2021). FNO + physics loss — names his whole area.

**Tier 2 — know the one-line idea:**
- Lu et al., *DeepONet* (Nature Machine Intelligence 2021) — the other operator architecture.
- Brunton, Noack & Koumoutsakos, *Machine Learning for Fluid Mechanics* (Annu. Rev. Fluid Mech. 2020) — field overview + vocabulary.
- Gal & Ghahramani, *Dropout as a Bayesian Approximation* (2016) — MC-Dropout.
- Lakshminarayanan et al., *Deep Ensembles* (2017) — better-calibrated UQ.
- Rahaman et al., *Spectral Bias of Neural Networks* (2019) — motivates Sobolev loss.

**Tier 3 — your own anchors:** Gatheral, Jaisson & Rosenbaum, *Volatility is Rough* (2018); Henry, *Geometric Theory of Semilinear Parabolic Equations* (FKPP).

---

## 4. Core methods — know these cold

### FNO (Fourier Neural Operator)
- Learns a **mapping between function spaces** (an operator), not just a vector→vector function.
- **Spectral conv layer:** FFT the input → keep the lowest k Fourier modes → apply a learned **complex linear transform** → inverse FFT; plus a pointwise (1×1) skip path and a nonlinearity.
- **Why powerful:** resolution/discretisation invariance (train at one resolution, evaluate at another); global receptive field in one layer; cheap O(N log N). Mode truncation = a learnable low-pass filter (smoothness prior).
- Your model: function (feature window) → scalar (H); an operator-*encoder*. Say that precisely.

### PINN vs Operator learning vs PINO
- **PINN:** learns *one* solution of *one* PDE instance by putting the PDE residual in the loss. No data needed; retrain per new condition.
- **Neural operator (FNO/DeepONet):** learns the *solution operator* across many instances from data; generalises in one forward pass.
- **PINO:** FNO + physics-residual loss = data efficiency + physical consistency. The natural framing for "physics-informed FNO" and the centre of Zhang's work.

### Sobolev loss
- Hˢ norm in Fourier space: `‖u‖²_{Hˢ} = Σ (1+|k|²)ˢ |û(k)|²` — upweights high frequencies (derivative errors) for s>0.
- **Why:** plain MSE (=L²=H⁰) matches values only; a Sobolev/H¹ loss also matches **derivatives**, countering spectral bias and improving gradient-dependent downstream quantities (vorticity, strain rate).
- **Your case (be exact):** your output is a scalar H, so your "Sobolev smoothness penalty" is a **smoothness prior on the predicted H(t) trajectory**, NOT the H¹ field data-loss of PINO. Don't conflate them.

### Davies–Harte (fractional Gaussian noise generation)
- Generates **exact** fGn (increments of fractional Brownian motion) for a given Hurst H.
- **Method:** embed the Toeplitz covariance in a **circulant** matrix → its eigenvalues = **FFT of the first row** → √eigenvalues × complex Gaussian → inverse FFT → two exact paths.
- **vs Cholesky:** O(n log n) vs O(n³), both exact; GPU-batchable. Caveat: needs non-negative circulant eigenvalues (grow the embedding if not).
- **Bridge:** same FFT-of-a-spectrum idea as synthetic turbulent inflow (Mann model, Kaimal/von Kármán spectra).

### MC-Dropout (UQ)
- Keep dropout **on at inference**, run T forward passes; mean = prediction, variance = uncertainty. A variational Bayesian approximation.
- Captures **epistemic** (model) uncertainty, not **aleatoric** (data noise) — know that distinction cold.
- **Limits:** miscalibrated for a fixed dropout rate; **deep ensembles** are better but N× cost. Calibration checks: reliability diagram / ECE / predictive-interval coverage.

### DDP (DistributedDataParallel)
- Data-parallel: one process per GPU, each a full model replica on a different data shard; gradients **all-reduced (averaged) via NCCL**, overlapped with backward.
- **vs DataParallel:** DDP is multi-process, no GIL, scales to multi-node — strictly preferred.
- Gotchas: `DistributedSampler` + `set_epoch`; effective batch = per-GPU × world_size (scale LR); only rank 0 logs/checkpoints; often paired with AMP.

### Cross-cutting one-liners
- **Rough vol ↔ turbulence:** both rough, multifractal, scale-invariant; fBm was originally a turbulence model (Mandelbrot); estimating H ≈ estimating turbulence scaling exponents.
- **Kolmogorov:** energy spectrum E(k) ∝ k^(−5/3) in the inertial range; energy cascade; intermittency. fBm spectrum ∝ |k|^(−(2H+1)).
- **Spectral bias:** nets learn low frequencies first → why turbulence is hard for ML and why Sobolev weighting helps.
- **CFD fidelity:** DNS (resolve all scales, exact, costly) → LES (resolve large eddies, model small) → RANS (model all turbulence, cheap). Surrogates aim for LES/DNS quality at RANS cost.

---

## 5. The quant project — full talking points

**Pitch:** "A physics-informed FNO that maps a window of price/volatility features to a time-varying Hurst exponent, trained on rough-Heston SPDE paths, used to regime-switch a strategy (low H → mean-revert, H≈0.5 → neutral, high H → momentum)."

**Transfer:** estimating a roughness exponent from noisy data, regularised by the underlying SPDE = same problem class as reconstructing a flow field from sparse noisy Lidar regularised by Navier–Stokes (his 2024 paper). Say this explicitly.

**Components — done vs in-progress (be honest):** Davies–Harte data-gen on GPU; Sobolev smoothness prior on H(t); MC-Dropout UQ; DDP scaling. If any aren't finished by interview time, say "I'm currently implementing X — here's the design and why," which reads as *stronger*.

**Honesty flags specific to this repo:**
- Don't claim a Navier–Stokes-style PDE residual — it's a smoothness/SPDE-consistency prior on H(t).
- The data generator currently uses **Cholesky**, you're moving to Davies–Harte — describe accordingly.
- Don't claim live alpha — it's a **research prototype validated by walk-forward backtesting**, not a deployed strategy.
- If asked about parameter count / exact loss, state what's actually in the code, not the aspirational version.

---

## 6. The FKPP project — existence/uniqueness/stability (the other half of your CV)

**Setting:** steady states u : ℝ → [0,1] of the random reaction–diffusion equation −u″(x) = ξ(x) f(u(x)), with FKPP nonlinearity f(u)=u(1−u) and a smooth stationary random field ξ.

- **Main result:** almost surely a **unique nontrivial** steady state (not ≡0 or ≡1).
- **Existence:** trivial states 0,1 are linearly unstable — the Anderson operator ∂ₓ² + ξ has a positive principal eigenvalue on large boxes, so the solution settles on a nontrivial profile. *(Caveat: this section is still being completed in your draft — present it as the mechanism, not a finished proof.)*
- **Uniqueness:** shooting method — rewrite as u′ = w, w′ = −ξ f(u); adapted solutions a.s. hit {0,1}; a comparison principle keeps ordered starts ordered; a squeeze argument rules out two distinct profiles → uniqueness on the half-line, extended to ℝ.
- **Stability:** on a bounded interval [−τ,τ] with Dirichlet BCs, stability is set by the principal eigenvalue of ∂ₓ² + λ s(x) f′(0); a threshold λ₀ separates a stable trivial state (λ<λ₀) from instability + a bifurcating nontrivial branch (λ>λ₀).
- **Open:** asymptotic stability + global attraction under the parabolic flow still to be established in full — a good, honest thing to volunteer.

**Why he'll care:** shows you reason rigorously about SPDEs, stochastic noise, numerical validation, and spectral/eigenvalue stability — the conceptual underpinnings of his ML-for-PDEs work.

---

## 7. Likely questions → model answers

- **"Why an FNO, not a CNN/LSTM?"** → Resolution invariance, global receptive field, spectral inductive bias suited to power-law/rough processes; O(N log N).
- **"What exactly is your physics-informed loss?"** → A smoothness/SPDE-consistency prior on the predicted H(t) — not a Navier–Stokes residual. State it plainly.
- **"How do you differentiate a rough path?"** → The real tension: rough Bergomi gives Hölder-rough, non-differentiable paths while PINN losses want derivatives; the Sobolev/spectral treatment is how I handle it. (Great question to get — shows depth.)
- **"Davies–Harte vs Cholesky?"** → Same exact distribution, O(n log n) vs O(n³), GPU-batchable; needs non-negative circulant eigenvalues.
- **"Is MC-Dropout well-calibrated?"** → Often not; epistemic only; deep ensembles better but costlier; I'd check ECE / interval coverage.
- **"How does DDP keep replicas synced?"** → Gradient all-reduce via NCCL, overlapped with backward; DistributedSampler + set_epoch; rank-0-only checkpointing.
- **"Does the trading strategy make money?"** → It's a research prototype validated by walk-forward backtesting, not a live strategy — I won't claim alpha I haven't demonstrated.
- **"How does any of this relate to wind/turbulence?"** → Roughness/scaling analogy; FFT-spectrum sampling = synthetic inflow; FNO surrogates for wakes; UQ for trustworthy control.
- **"What's the difference between a PINN and a neural operator?"** → PINN solves one instance via physics residual; operators learn the solution map across instances; PINO combines them.
- **"DNS vs LES vs RANS?"** → Decreasing fidelity/cost: resolve all scales → resolve large eddies + model small → model all turbulence.
- **"What would you do in the first year / where would you take this?"** → Reproduce a baseline (a PINN for a canonical flow or your digital-twin pipeline), then move from scalar regression to field-to-field operator learning, add a true PDE-residual (PINO) loss, replace MC-Dropout with calibrated/ensemble UQ, scale with DDP.
- **"What's your weakness here?"** → Limited formal CFD/turbulence background; mitigated by strong PDE/SPDE foundations and hands-on operator-learning experience, and I learn fast (point to teaching yourself the FNO stack).

---

## 8. Questions to ask him (shows engagement — pick 2–3)

- Which thread would I likely start on — wake reconstruction, farm-level optimisation, or the foundation-model direction?
- How much of the data is high-fidelity CFD you generate in-house vs public datasets / real Lidar?
- How do you see foundation models fitting in — pretraining on simulation data, then fine-tuning per farm?
- What's the biggest open problem in making these surrogates trustworthy enough for operational use (UQ, generalisation to unseen inflow)?
- What compute is available (HPC scale), and how do students typically split time between method development and the wind-energy application?

---

## 9. One-line mantra before you walk in

*"I take a governing equation, discretise and validate it, embed the physics into a neural operator, quantify uncertainty, and scale it on HPC — I've done exactly that twice, and I want to do it for wind."*
