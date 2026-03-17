# Task 1: Wasserstein GAN Theory Questions - Answers

## 1. Feature Matching and GAN Training Instability

Feature matching is a technique used to address the instability in training GANs. The core idea is to modify the generator loss function to not directly fool the discriminator, but instead to match the expected value of activations in an intermediate layer of the discriminator.

**Traditional approach**: The generator tries to minimize the difference between generated and real data in the discriminator's output space, which can lead to instability.

**Feature matching approach**: Instead, the generator optimizes its loss to match the features extracted by the discriminator. Specifically:
- Let $f(x)$ be activations from an intermediate layer of the discriminator for input $x$
- Generator loss: $\mathbb{E}_{z \sim p(z)}[||f(G(z)) - \mathbb{E}_{x \sim p_{data}}[f(x)]||^2_2]$

**Benefits**:
- Stabilizes training by providing a smoother gradient signal to the generator
- Reduces mode collapse by encouraging diversity in generator outputs
- Provides more meaningful feedback based on data distribution statistics rather than binary classification

---

## Q2: What is meant by mode collapse in GANs?

Mode collapse occurs when a generator learns to produce a limited variety of outputs, often ignoring substantial portions of the training data distribution. Instead of generating diverse samples across the full range of real data, the generator "collapses" to producing only a few distinct modes or types of data.

**Characteristics:**
- The generator learns to fool the discriminator by repeatedly producing similar (but convincing) samples
- Only a subset of the data distribution is learned
- Low diversity in generated samples despite having trained for many iterations

**Why it happens:**
- The generator finds a local optimum by exploiting a weakness in the discriminator
- Once successful on a subset of patterns, the generator has little incentive to explore other patterns
- The discriminator gradually becomes unable to distinguish between the limited fake outputs and real data

**Impact:**
- Models fail to capture the full complexity of the target distribution
- Generated samples lack diversity and appear repetitive

---

## Q3: Explain the concept of minibatch discrimination.

Minibatch discrimination is a technique to help the discriminator detect mode collapse by examining the diversity within a minibatch of samples rather than evaluating each sample independently.

**How it works:**
1. For each sample in a minibatch, compute a "minibatch feature vector" based on the distances to all other samples in the same batch
2. Add this minibatch feature to the discriminator's input before the final classification layer
3. If all samples in a batch are identical (mode collapse), these features will reflect low diversity
4. The discriminator can then more easily reject batches with low sample diversity

**Benefits:**
- The discriminator explicitly learns to penalize homogeneous batches
- Encourages the generator to produce diverse samples within each batch to avoid detection
- Reduces mode collapse without changing the generator's loss function
- Works well in practice and has been shown to improve sample diversity significantly

**Key insight:**
- Minibatch discrimination gives the discriminator an additional signal about diversity, not just whether individual samples are real or fake

---

## Q4: Explain the significance of the following stabilization terms in GANs.

### a) Historical Averaging

**Definition:**
Historical averaging involves maintaining a rolling average of past parameter values during training. Rather than using only the current parameter values, the discriminator and generator use a weighted combination of current and historical parameters.

**Significance:**
- Smooths parameter trajectories, preventing sudden jumps in the loss landscape
- Reduces oscillations between the generator and discriminator during training
- Acts as a regularizer, constraining how much parameters can change per iteration
- Helps maintain stability when the generator and discriminator are in an adversarial cycle

**Effect:** Training becomes more predictable and less prone to divergence.

---

### b) One-sided Label Smoothing

**Definition:**
Rather than using hard labels (0 for fake, 1 for real), one-sided label smoothing replaces hard positive labels (1) with soft labels (e.g., 0.9).

**Significance:**
- Real images: discriminator targets 0.9 instead of 1.0
- Fake images: discriminator still targets 0.0

**Why it helps:**
- Prevents the discriminator from becoming overconfident, which can cause gradient collapse
- Gradient signals remain informative even when the discriminator is very confident
- Reduces the risk of the discriminator driving the generator gradient to zero (vanishing gradients)
- Encourages the discriminator to learn more robust decision boundaries

**Effect:** More stable and gradual learning curves.

---

### c) Virtual Batch Normalization

**Definition:**
Virtual batch normalization involves normalizing each sample in a batch with respect to a fixed "reference batch" of real data, instead of computing batch statistics from the current minibatch.

**Significance:**
- Stabilizes the discriminator by using consistent normalization statistics
- Prevents the discriminator's behavior from changing abruptly when fed different minibatches
- Reduces dependence on minibatch composition, which can cause training instability

**Benefits:**
- The discriminator sees more consistent feature statistics across batches
- Generators cannot exploit batch-specific statistics to generate fake data
- Particularly helpful for smaller batch sizes where batch statistics are noisy

**Effect:** Training is more stable and less sensitive to within-batch variation.

---

## Q5: What are the proposed suggestions to stabilize the training in GANs?

**1. Architecture Design:**
   - Use strided convolutions instead of pooling for downsampling
   - Use transposed convolutions for upsampling
   - Avoid sparse gradients; ensure all layers contribute meaningful gradients

**2. Training Techniques:**
   - Feature matching (discussed in Q1)
   - Minibatch discrimination (discussed in Q3)
   - Historical averaging (discussed in Q4a)
   - One-sided label smoothing (discussed in Q4b)
   - Virtual batch normalization (discussed in Q4c)

**3. Loss Function Modifications:**
   - Use least squares loss instead of binary cross-entropy (LSGAN)
   - Use Wasserstein distance instead of JS divergence (WGAN)
   - Use spectral normalization to regularize the discriminator

**4. Hyperparameter Strategies:**
   - Use a learning rate schedule that decays over time
   - Adjust the learning rates for generator and discriminator independently
   - Use different optimizers (Adam for generator, SGD or Adam with momentum for discriminator)

**5. Inspection and Monitoring:**
   - Monitor the Inception Score (IS) or Fréchet Inception Distance (FID) to track sample quality
   - Visualize generated samples regularly to detect mode collapse
   - Log discriminator and generator losses to ensure both are learning

---

## Q6: Define and explain the JS divergence loss as used in GANs, and how it compares to KL divergence in VAEs.

### JS Divergence (Jensen-Shannon Divergence)

**Definition:**
The Jensen-Shannon (JS) divergence is a symmetric measure of the difference between two probability distributions:

$$D_{JS}(P || Q) = \frac{1}{2}D_{KL}(P || M) + \frac{1}{2}D_{KL}(Q || M)$$

where $M = \frac{1}{2}(P + Q)$ is the average distribution.

**In GANs:**
- The original GAN loss can be shown to minimize the JS divergence between the real data distribution and the generated distribution
- The discriminator is trained to distinguish real from fake (minimize cross-entropy loss)
- The generator is trained to minimize the JS divergence

**Advantages of JS over KL:**
- **Symmetric:** $D_{JS}(P || Q) = D_{JS}(Q || P)$, whereas $D_{KL}(P || Q) \neq D_{KL}(Q || Q)$
- **Always defined:** JS divergence is defined even when distributions have non-overlapping support, whereas KL divergence becomes infinite

---

### KL Divergence in VAEs

**Definition:**
The Kullback-Leibler (KL) divergence measures how one probability distribution diverges from another:

$$D_{KL}(P || Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

**In VAEs:**
- Used to measure divergence between the variational posterior $q(z|x)$ and the prior $p(z)$
- The VAE loss includes: $\mathcal{L} = E_q[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$
- The first term reconstructs the data; the second term regularizes the latent space

**Key Differences:**

| Aspect | JS Divergence (GANs) | KL Divergence (VAEs) |
|--------|----------------------|----------------------|
| **Symmetry** | Symmetric | Asymmetric |
| **Mode Coverage** | Mode-averaging (misses modes) | Mode-covering (attempts all modes) |
| **Behavior** | Smoother gradients when distributions overlap | Steeper gradients, can vanish when distributions are far apart |
| **Use Case** | Learning generator distribution in adversarial game | Variational inference in explicit probabilistic models |

---

## Q7: Define Wasserstein distance. Why is Wasserstein distance better than JS or KL divergence? Explain with an example.

### Wasserstein Distance Definition

The Wasserstein distance (also called Earth Mover's Distance or EMD) between two distributions P and Q is:

$$W(P, Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[||x - y||]$$

where $\Pi(P,Q)$ is the set of all joint distributions (couplings) with marginals P and Q.

**Intuition:** The minimum "cost" to transport one distribution to another, where cost is measured as the total distance traveled.

---

### Why Wasserstein is Better than JS/KL

**1. Meaningful gradients with disjoint support:**
- JS and KL divergences are constant when distributions have **zero overlap** 
- Wasserstein provides a smooth, meaningful gradient signal even when supports don't overlap
- This enables learning even in early training when real and fake distributions are far apart

**2. Convergence properties:**
- Optimization w.r.t. Wasserstein distance is more stable
- Better alignment with human perception (distance between samples matters)
- Gradients are informative throughout training

**3. Continuous metric:**
- Wasserstein distance varies smoothly as distributions change
- JS/KL can remain constant over large regions, then jump

---

### Example: Two Disjoint Distributions

Suppose:
- **P (real):** All mass on x = 0 (point mass)
- **Q_t (fake at iteration t):** All mass on x = t (point mass moving from 1 to 0)

**As t → 0 (generator improves):**

| Metric | Value | Gradient |
|--------|-------|----------|
| **JS Divergence** | $\log 2$ | 0 (constant) |
| **KL Divergence** | $\infty$ | Undefined/Crash |
| **Wasserstein** | $t$ | 1 (smooth, informative) |

**Interpretation:**
- JS and KL don't help the generator improve during early training (no gradient)
- Wasserstein continuously measures progress (generator at x=0.5 is "closer" than at x=1.0)
- This allows the generator to receive meaningful feedback and improve steadily

**Real-world implication:**
In GANs, Wasserstein distance enables stable training because:
- Even when generated images are completely unrealistic (disjoint support), the gradient signal guides improvement
- Early training is typically problematic for standard GANs; WGAN solves this

---

## Q8: Explain the term Lipschitz continuity.

**Definition:**
A function $f: X \rightarrow Y$ is **Lipschitz continuous** if there exists a constant $K \geq 0$ (the Lipschitz constant) such that for all $x_1, x_2 \in X$:

$$|f(x_1) - f(x_2)| \leq K \cdot ||x_1 - x_2||$$

**Intuition:** The function's output cannot change faster than a fixed rate relative to the input change.

---

### Properties and Significance

**1. Bounded gradients:**
- If $f$ is differentiable and Lipschitz with constant $K$, then $||\nabla f|| \leq K$ everywhere
- Gradients are bounded, preventing exploding or vanishing gradients

**2. Geometric interpretation:**
- The slope of the function (or worst-case slope) is limited by K
- Steep cliffs or discontinuities are prohibited

**3. Stability:**
- Lipschitz functions are continuous and stable to small perturbations
- Small input changes produce bounded output changes

---

### Lipschitz Continuity in WGAN

The Wasserstein distance formula requires the critic (discriminator) to be **1-Lipschitz continuous** (K=1):

$$W(P, Q) = \sup_{f \text{is 1-Lipschitz}} \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)]$$

**Why enforce Lipschitz constraint?**
- Ensures the critic's output is a valid Wasserstein distance estimate
- Prevents the critic from becoming arbitrarily sensitive to small changes
- Stabilizes training by bounding the gradient magnitudes

**How WGAN enforces it:**
- **Weight clipping:** Clip weights to [-c, c] after each update
- **Gradient penalty:** Penalize violations of the gradient norm constraint during training
- These ensure the critic remains 1-Lipschitz throughout training

---

## Q9: Compared to the original GAN algorithm, state the changes made for WGAN implementation.

### Original GAN vs WGAN Comparison

| Aspect | Original GAN | WGAN |
|--------|-------------|------|
| **Divergence** | JS divergence | Wasserstein distance |
| **Generator Loss** | $-\log(D(G(z)))$ | $-\mathbb{E}[D(G(z))]$ |
| **Discriminator Loss** | $\mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ | $\mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$ |
| **Discriminator Role** | Binary classifier (0 to 1) | Critic/Regression (unbounded) |
| **Activation** | Sigmoid output layer | No restriction (linear) |
| **Lipschitz Constraint** | None | Weight clipping or gradient penalty |
| **Optimal Discriminator** | Classifies correctly | Approximates Wasserstein distance |
| **Early Training** | Poor gradients when distributions disjoint | Useful gradients from start |
| **Training Stability** | Unstable, mode collapse common | Significantly more stable |

---

### Key Changes in WGAN Implementation

**1. Loss Functions:**
   - Remove sigmoid from discriminator output
   - Generator: minimize $-\mathbb{E}[D(G(z))]$ instead of maximizing $\log D(G(z))$
   - Discriminator: maximize $\mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$

**2. Constraint Enforcement:**
   - Add weight clipping: After each discriminator update, clip weights $w \in [-c, c]$
   - Alternatively, use spectral normalization or gradient penalty

**3. Architecture Changes:**
   - Remove batch normalization from discriminator (can interfere with Lipschitz constraint)
   - Use layer normalization or instance normalization in discriminator if needed
   - Discriminator outputs a scalar (not a probability)

**4. Training Dynamics:**
   - Can use RMSprop with very small learning rate (e.g., 0.00005)
   - Train discriminator multiple times per generator update (typically 5:1 ratio)
   - Discriminator loss should decrease smoothly, indicating convergence

**5. Monitoring:**
   - Plot the discriminator loss (should decrease) and generator loss (less critical)
   - Use Inception Score or FID to track sample quality
   - Wasserstein distance estimate itself is informative (lower is better)

---

## Summary

The WGAN addresses fundamental training instabilities in GANs by:
1. Using Wasserstein distance instead of JS divergence
2. Enforcing Lipschitz continuity of the critic
3. Providing meaningful gradients even when distributions are disjoint
4. Enabling training with meaningful loss values that correlate with sample quality
5. Significantly reducing mode collapse and improving convergence stability
