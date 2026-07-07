# PhD Interview Prep — Technical Notes

**Role:** Physics-Aware ML for Aerodynamics & Wind Energy (Jincheng Zhang, Warwick)
**Strategy:** Lead with the FNO project. Be able to defend every claim on your CV from first principles. Zhang builds FNO + physics-informed surrogates for wind-farm flow fields — he will probe exactly the four areas below.

---

## 0. FNO foundations (know this cold — it underpins everything)

- **What an FNO is:** a neural network that learns a *mapping between function spaces* (an operator), not just a function between vectors. Input function → output function. Reference: Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021.
- **The core layer (SpectralConv):** for each layer, take FFT of the input field → keep only the lowest `k` Fourier modes → apply a learned **complex linear transform** to those modes → inverse FFT. Add a parallel pointwise linear (1×1 conv) "bypass/skip" path, then a nonlinearity (GELU). Your `SpectralConv1d` does exactly this (rfft → truncate to `n_modes` → einsum with complex weights → irfft).
- **Why it's powerful:**
  - **Resolution / discretisation invariance** — because the operation is defined in Fourier space on modes, you can train at one resolution and evaluate at another. This is the headline property versus a CNN.
  - **Global receptive field in one layer** — a spectral convolution is a global operation (each output point depends on all input points), unlike a local CNN kernel.
  - **Cheap** — FFT is O(N log N); truncating to low modes is a learnable spectral filter.
- **Mode truncation = a low-pass filter.** You keep `n_modes` (you use 16). This is a deliberate inductive bias: the operator's kernel is assumed smooth/low-rank in frequency. Be ready to justify your choice and discuss the trade-off (too few modes → underfit fine structure; too many → overfit / cost).
- **Operator learning vs PINNs (be able to contrast cleanly):**
  - **PINN** (Raissi et al. 2019): learns *one* solution of *one* PDE instance by putting the PDE residual into the loss. No training data needed, but must retrain per new boundary/initial condition.
  - **Neural operator (FNO/DeepONet):** learns the *solution operator* across many instances/parameters/ICs from data. Generalises to new inputs in one forward pass.
  - **PINO** (Physics-Informed Neural Operator, Li et al. 2021): FNO **+** a physics-residual loss. Best of both — data efficiency of an operator plus physical consistency of a PINN. **This is the natural framing for your "physics-informed FNO" and directly matches Zhang's work.**
- **Your specific task:** the input is a window of returns/realised-vol features; the output is the Hurst exponent H. Be honest that yours is currently an operator-*encoder* (function in → scalar out) rather than function → function. That's a legitimate use; just describe it precisely.

---

## 1. Sobolev penalty / Sobolev-norm loss

**Definition**
- A **Sobolev space Hˢ** contains functions whose derivatives up to order *s* are square-integrable. The norm, in Fourier space, is
  `‖u‖²_{Hˢ} = Σ_k (1 + |k|²)ˢ |û(k)|²`.
- The weight `(1+|k|²)ˢ` **upweights high frequencies** for s > 0 (penalises errors in derivatives / fine structure) and **downweights** them for s < 0 (a smoothing norm).

**Why use it as a loss (instead of plain MSE = L² = H⁰)**
- Plain MSE only matches *values*. A Sobolev (e.g. **H¹**) loss also matches **gradients/derivatives** of the field.
- Counters **spectral bias**: neural nets preferentially fit low frequencies first, so MSE-trained models smear out fine-scale features. Weighting high modes forces the model to learn them.
- Critical when the output is later **differentiated** — e.g. velocity → vorticity/strain-rate in turbulence, or any downstream quantity that depends on gradients. A field can have small L² error but large derivative error.
- Cheap inside an FNO because you're already in Fourier space — the Sobolev weighting is just a multiply on the FFT of the error.

**Trade-offs / what he might push on**
- Upweighting high modes can **amplify noise**; pick *s* and the weighting coefficient carefully, or use a blended loss `L = MSE + λ·(Sobolev term)`.
- **Be precise about what your penalty acts on.** Your model outputs a *scalar* H per window, so a "Sobolev smoothness penalty" is **not** the same H¹ data-loss used in PINO. In your setup it most naturally means a **smoothness regulariser on the predicted H(t) trajectory** across consecutive windows (penalise `|dH/dt|²` or its discrete/Fourier analogue) to stop the regime signal jittering. Know exactly which interpretation you implemented and say it plainly — conflating the two is the easiest place to get caught.

**One-line bridge to his work:** "A Sobolev loss on a wake/velocity field would force the surrogate to get *velocity gradients* right, not just the velocity — which matters for load and vorticity prediction."

---

## 2. Davies–Harte generator (circulant embedding) for fGn/fBm

**The goal**
- Generate **exact** samples of **fractional Gaussian noise** (fGn = stationary increments of fractional Brownian motion) with a given Hurst H. fGn is a zero-mean stationary Gaussian process with autocovariance
  `γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})`.

**The method (be able to walk through the steps)**
1. The covariance matrix of n samples is **Toeplitz** (constant along diagonals). Davies–Harte **embeds** this n×n Toeplitz matrix inside a larger **circulant** matrix of size m = 2(n−1).
2. A circulant matrix is **diagonalised by the DFT**, so its eigenvalues are just the **FFT of its first row** — computed in O(m log m).
3. Take √(eigenvalues), multiply by complex Gaussian white noise, **inverse FFT** → two independent exact fGn sample paths.
4. Cumulatively sum fGn → fBm.

**Why it's better than your current Cholesky approach**
- **Cholesky** factorises the full covariance: **O(n³)** time (O(n²) memory). Exact but expensive and doesn't scale.
- **Davies–Harte: O(n log n)** time, O(n) memory, and **also exact** — same distribution, far faster. This is the standard method for long paths and large batches.
- **GPU-friendly:** FFTs parallelise extremely well. `torch.fft` on CUDA lets you generate **thousands of paths in a batched tensor** in one shot — this is literally what makes "GPU data generation" true.

**Caveats he might probe**
- The embedding only works if the **circulant eigenvalues are non-negative** (otherwise √(λ) is imaginary). For fGn with H ∈ (0,1) the minimal embedding m = 2(n−1) is provably valid; in general you may need to **grow the embedding** (pad m) until all eigenvalues are ≥ 0. Know this condition.
- It produces stationary fGn on a regular grid; non-uniform sampling needs other methods (e.g. Hosking, or approximate hybrid schemes).

**Strong bridge to wind energy:** the *same* idea — sampling a correlated Gaussian random field by taking the FFT of a target spectrum/covariance — is how **synthetic turbulent inflow** is generated for wind-turbine simulation (spectral methods; Kaimal/von Kármán spectra; the **Mann model** for sheared anisotropic turbulence). Say this. It shows you see that "Davies–Harte for fBm" and "spectral synthetic turbulence for wind inflow" are the same mathematical machinery.

---

## 3. MC-Dropout for uncertainty quantification

**The idea (Gal & Ghahramani, ICML 2016)**
- Keep **dropout active at inference time** and run **T stochastic forward passes**. Each pass drops different units → T different predictions.
- **Predictive mean** = average of the T outputs. **Predictive uncertainty** = variance across the T outputs (plus an inverse-model-precision term).
- Theoretically interpreted as a **variational (Bayesian) approximation** — dropout-at-test approximates sampling from an approximate posterior over weights, equivalent to a deep Gaussian process approximation.

**What kind of uncertainty it gives**
- Captures **epistemic** uncertainty (model uncertainty — reducible with more data; large where you extrapolate).
- Does **not** by itself capture **aleatoric** uncertainty (irreducible data noise). For that, add a **heteroscedastic head** that predicts a variance, and combine the two.
- Know the distinction cold — "epistemic = reducible/model, aleatoric = irreducible/data" is a near-certain question.

**Limitations (he WILL push here — UQ is his trust theme)**
- Quality depends heavily on the **dropout rate**, which acts as a fixed variational parameter. A fixed p is often **miscalibrated**; **Concrete Dropout** learns p per layer to fix this.
- MC-Dropout often **underestimates** uncertainty and can be poorly calibrated out-of-distribution.
- **Deep Ensembles** (Lakshminarayanan et al. 2017) — train N independent models — generally give **better-calibrated** UQ, at N× cost. Be ready to say *why you chose MC-Dropout anyway* (single model, cheap, no retraining, good enough as a first pass) and that ensembles are the upgrade path.

**How to evaluate calibration (have an answer ready)**
- **Reliability diagram** / **Expected Calibration Error (ECE)**.
- **Predictive-interval coverage**: do your nominal 90% intervals actually contain the truth ~90% of the time? Under-coverage = overconfident.
- Sharpness vs calibration trade-off (well-calibrated *and* tight is the goal).

**Bridge:** "A wake surrogate that doesn't know when it's extrapolating to an unseen inflow condition is dangerous for control/load decisions — UQ tells you when to fall back to the high-fidelity solver." That sentence is squarely his motivation.

---

## 4. PyTorch DistributedDataParallel (DDP)

**What it is**
- **Data-parallel** training. Each **process** owns **one GPU** and a **full replica** of the model. Each process gets a **different shard** of every batch.
- Each replica does its own forward + backward; then gradients are **all-reduced (averaged) across all processes** (via the **NCCL** backend) before the optimizer step → all replicas stay identical.
- The all-reduce is **bucketed and overlapped with the backward pass**, so communication hides under computation — that's why DDP scales well.

**DDP vs DataParallel (DP) — a classic question**
- **DP** = single process, multi-thread, one Python GIL → slow; replicates the model and scatters/gathers every step; single-node only.
- **DDP** = multi-process (one per GPU), no GIL contention, gradients synced via all-reduce, **scales to multiple nodes**. **DDP is strictly preferred.**

**The moving parts (be able to list them)**
- `init_process_group(backend="nccl")`; concepts of **rank**, **world_size**, **local_rank**.
- **`DistributedSampler`** so each rank sees a **disjoint** slice of data — and call **`sampler.set_epoch(epoch)`** each epoch or your shuffling is broken (common bug).
- Wrap: `model = DDP(model, device_ids=[local_rank])`.
- Launch with **`torchrun`** (sets the env vars, spawns one process per GPU).

**Gotchas he might test**
- **Effective batch size = per-GPU batch × world_size** → you usually **scale the learning rate** (linear scaling rule) and add **warmup**.
- **Only rank 0** should log, checkpoint, and write files (else N processes clobber each other).
- Avoid **duplicated/again-sampled validation** across ranks; gather metrics properly.
- Use **`SyncBatchNorm`** if batchnorm stats matter across GPUs (less relevant for your InstanceNorm).
- Often combined with **mixed precision (AMP)** and **gradient accumulation** for memory.

**Why it matters here:** the PhD explicitly mentions **large-scale foundation models** and high-fidelity simulation — multi-GPU / multi-node training is non-negotiable at that scale, and Warwick has HPC (Avon/Sulis). Demonstrating DDP says "I can train at the scale your research needs." Don't undersell this; many strong mathematicians can't do it.

---

## 5. Cross-cutting talking points (the "I think across domains" signal)

- **Rough volatility ↔ turbulence.** Both are **rough, multifractal, scale-invariant** processes. Gatheral, Jaisson & Rosenbaum, *Volatility is Rough* (2018) found H ≈ 0.1 for volatility. **fBm was originally introduced by Mandelbrot partly to model turbulence.** Estimating H (your task) is mathematically the same flavour as estimating **scaling exponents** in a turbulent field.
- **Kolmogorov turbulence basics** (headline level): energy spectrum **E(k) ∝ k^{−5/3}** in the inertial range; the **energy cascade** from large to small scales; **intermittency**. Compare: fBm has a power-law spectrum **|k|^{−(2H+1)}**. The link is "power-law spectra ⇄ self-similar roughness."
- **Spectral bias of neural networks** (Rahaman et al. 2019): nets learn low frequencies first. This *motivates* your Sobolev penalty and is central to why turbulence (broadband, high-frequency) is hard for vanilla ML.
- **Fidelity hierarchy in CFD** (one sentence each): **DNS** (resolve all scales, exact, hugely expensive) → **LES** (resolve large eddies, model the small) → **RANS** (model all turbulence, cheap, least accurate). Surrogates aim to get LES/DNS-quality output at RANS-like cost.
- **His actual papers** — read abstracts + figures: *Digital twin of wind farms via physics-informed deep learning* (Energy Conversion & Management, 2023); *Reconstruction of dynamic wind turbine wake flow fields from virtual Lidar via PINNs* (2024). The second — fusing **sparse, noisy Lidar measurements** with the **2D Navier–Stokes equations** — is conceptually parallel to your ensemble/noise/regularisation work. Name-drop it specifically.

---

## 6. Likely questions → crisp answers to rehearse

- **"Why an FNO and not a CNN/LSTM?"** → Resolution invariance, global receptive field, spectral inductive bias suited to processes with power-law/spectral structure; cheap O(N log N).
- **"What exactly is your physics-informed term?"** → State precisely what residual/penalty you added and what it acts on (see §1 — don't overclaim it's a Navier-Stokes-style PDE residual if it's a smoothness prior on H(t)).
- **"Davies–Harte vs Cholesky — why switch?"** → Same exact distribution, O(n log n) vs O(n³), GPU-batchable; mention the non-negative-eigenvalue embedding condition.
- **"Is MC-Dropout well-calibrated?"** → Often not; epistemic only; deep ensembles better but costlier; here's how I'd check (ECE, coverage).
- **"How does DDP keep replicas in sync?"** → All-reduce of gradients via NCCL, overlapped with backward; DistributedSampler + set_epoch; rank-0-only checkpointing.
- **"How does any of this relate to wind/turbulence?"** → Roughness/scaling analogy; spectral field generation = synthetic inflow; FNO surrogates for wakes; UQ for trustworthy control.
- **"What would you do next / in the PhD?"** → Move from scalar-H regression to full **field-to-field** operator learning; add a **true PDE-residual (PINO) loss**; replace MC-Dropout with calibrated/ensemble UQ; scale training with DDP on HPC.

---

## 7. Honesty guardrail

Only claim what is implemented **by interview day**. For anything still in progress, say "I'm currently implementing X — here's the design and the reason." That framing reads as *stronger*, not weaker, to someone who builds these systems: it shows you understand the method deeply enough to be adding it deliberately.
