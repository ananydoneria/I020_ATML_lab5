# ATML Lab 05 (WGAN) - Required Tasks

## Scope identified from the assignment sheet
Experiment No. 05 asks for two major deliverables:

1. Task 1 (Theory): Read the cited papers and answer all listed GAN/WGAN conceptual questions.
2. Task 2 (Implementation): Study and run a WGAN implementation on the CIFAR dataset.

## Required outputs to prepare

1. `answers/task1_answers.md`
- Feature matching and GAN instability.
- Mode collapse.
- Minibatch discrimination.
- Historical averaging.
- One-sided label smoothing.
- Virtual batch normalization.
- Suggested GAN training stabilization methods.
- JS divergence in GANs and comparison with KL divergence in VAEs.
- Wasserstein distance and why it can be better than JS/KL divergence (with example).
- Lipschitz continuity.
- Changes from original GAN to WGAN.

2. `src/train_wgan_cifar.py`
- WGAN model implementation for CIFAR.
- Training loop with optimizer settings.
- Logging of generator/discriminator (critic) losses.

3. `reports/task2_results.md`
- Training configuration (epochs, batch size, learning rate, optimizer, clipping/penalty choice).
- Qualitative sample outputs from generator.
- Brief analysis of training behavior and stability.

4. `reports/partb_submission.md`
- B.1 Task 1 summary.
- B.2 Task 2 summary.
- B.5 Conclusion in your own words.

## Recommended sequence

1. Complete Task 1 answers in `answers/task1_answers.md`.
2. Implement and run WGAN in `src/train_wgan_cifar.py`.
3. Document training outcomes in `reports/task2_results.md`.
4. Finalize `reports/partb_submission.md` for submission.

## Validation
Use the VS Code task "Lab05: Check required deliverables" to verify required files are present.
