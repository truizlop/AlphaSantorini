# Earlier Experiments (different network architecture, pre-masked-policy-loss)

## expA
- gamesPerIteration: 100
- mctsSimulations: 512
- trainingStepsPerIteration: 100
- batchSize: 128
- learningRate: 0.001
- valueTargetStrategy: mcts
- noise: default (epsilon 0.25, alpha 0.3)
- noiseAnnealIterations: 150
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expB
- gamesPerIteration: 200
- mctsSimulations: 256
- trainingStepsPerIteration: 150
- batchSize: 128
- learningRate: 0.001
- valueTargetStrategy: mcts
- noise: default (epsilon 0.25, alpha 0.3)
- noiseAnnealIterations: 150
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expC
- gamesPerIteration: 200
- mctsSimulations: 384
- trainingStepsPerIteration: 100
- batchSize: 128
- learningRate: 0.001
- valueTargetStrategy: terminal
- noise: default (epsilon 0.25, alpha 0.3)
- noiseAnnealIterations: 150
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expD
- gamesPerIteration: 200
- mctsSimulations: 256
- trainingStepsPerIteration: 200
- batchSize: 256
- learningRate: 0.0005
- valueTargetStrategy: mcts
- noise: epsilon 0.2, alpha 0.3
- noiseAnnealIterations: 300
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 150000

## expE
- gamesPerIteration: 200
- mctsSimulations: 256
- trainingStepsPerIteration: 150
- batchSize: 256
- learningRate: 0.0007
- valueTargetStrategy: mcts
- noise: default (epsilon 0.25, alpha 0.3)
- noiseAnnealIterations: 150
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expF
- gamesPerIteration: 200
- mctsSimulations: 384
- trainingStepsPerIteration: 150
- batchSize: 128
- learningRate: 0.001
- valueTargetStrategy: mcts
- noise: epsilon 0.25, alpha 0.3
- noiseAnnealIterations: 300
- noiseEpsilonFloor: 0.1
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expG
- gamesPerIteration: 200
- mctsSimulations: 512
- trainingStepsPerIteration: 200
- batchSize: 256
- learningRate: 0.0005
- valueTargetStrategy: mcts
- noise: default (epsilon 0.25, alpha 0.3)
- noiseAnnealIterations: 150
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expH
- gamesPerIteration: 200
- mctsSimulations: 256
- trainingStepsPerIteration: 150
- batchSize: 128
- learningRate: 0.001
- valueTargetStrategy: mcts
- noise: epsilon 0.15, alpha 0.3
- noiseAnnealIterations: 300
- noiseEpsilonFloor: 0.05
- hiddenDimension: 256 (default)
- replayBufferSize: 100000 (default)

## expI (initial heavy run, aborted)
- hiddenDimension: 384
- gamesPerIteration: 300
- mctsSimulations: 512
- trainingStepsPerIteration: 250
- batchSize: 256
- learningRate: 0.0005
- replayBufferSize: 200000
- valueTargetStrategy: mcts
- noise: epsilon 0.2, alpha 0.3
- noiseAnnealIterations: 400
- noiseEpsilonFloor: 0.05

## expI (balanced restart)
- hiddenDimension: 320
- gamesPerIteration: 200
- mctsSimulations: 384
- trainingStepsPerIteration: 200
- batchSize: 256
- learningRate: 0.0005
- replayBufferSize: 150000
- valueTargetStrategy: mcts
- noise: epsilon 0.2, alpha 0.3
- noiseAnnealIterations: 300
- noiseEpsilonFloor: 0.05

---

# Masked Policy Loss Experiments (Feb 2026)

All experiments below include the masked policy loss change: illegal action logits are set to -1e9 before logSoftmax in `policyLoss()` and diagnostic logging, so the softmax denominator only includes legal actions. This dropped initial policy loss from ~4.5 to ~3.0.

## Experiment 1: Original params + masked policy loss (from scratch)
- **Architecture**: 5 res blocks, 256 filters
- gamesPerIteration: 100
- mctsSimulations: 256
- trainingStepsPerIteration: 100
- batchSize: 128
- learningRate: 0.001
- replayBufferSize: 100000
- noise: epsilon 0.25→0.05 over 150 iters, alpha 0.3
- temperature: 1.0 for 30 moves, then 0.0
- valueTargetStrategy: terminal
- **Result**: 9 promotions in 289 iterations (last at iter 230). Policy loss 3.35→1.77, value loss 1.00→0.58, value corr 0.66. Plateaued after iter 230.
- **Log**: `training_masked_policy_500.log`
- **Best model**: `checkpoint_230.safetensors`

## Experiment 2 (v1): Tuned + temperature change (from checkpoint 230)
- **Architecture**: 5 res blocks, 256 filters
- trainingStepsPerIteration: 250
- replayBufferSize: 50000
- noise alpha: 0.20
- temperature: `max(0.2, 1.0 - move/80)` (smooth decay)
- All other params same as Experiment 1
- **Rationale**: More learning per iter (250 steps), fresher data (50k buffer), better noise calibration (alpha≈10/branching), extended exploration (temp decay)
- **Result**: FAILED. 0 promotions in 71 iterations. Policy loss went UP 1.88→2.35. Temperature change shifted data distribution too far.
- **Log**: `training_tuned_from230.log`
- **Lesson**: Don't change temperature schedule mid-training.

## Experiment 3 (v2): Tuned, no temperature change (from checkpoint 230)
- **Architecture**: 5 res blocks, 256 filters
- trainingStepsPerIteration: 250
- replayBufferSize: 50000
- noise alpha: 0.20
- temperature: original (1.0 for 30 moves, then 0.0)
- All other params same as Experiment 1
- **Rationale**: Same as v1 but keeping original temperature
- **Result**: FAILED. 0 promotions in 100 iterations (early-stopped). Policy loss mean 2.12, value loss 0.46. Within-iteration KL dropped to ~0.8 but reset to ~2.8 next iteration — overfitting.
- **Log**: `training_tuned_v2_from230.log`
- **Lesson**: 250 steps on 50k buffer causes overfitting. Network memorizes buffer each iteration but doesn't generalize.

## Experiment 4 (v3): More games + sims + steps (from checkpoint 230)
- **Architecture**: 5 res blocks, 256 filters
- gamesPerIteration: 150 (+50%)
- mctsSimulations: 320 (+25%)
- trainingStepsPerIteration: 150 (+50%)
- replayBufferSize: 100000
- noise alpha: 0.3
- temperature: original
- All other params same as Experiment 1
- **Rationale**: Higher quality MCTS targets (320 sims), more diverse data (150 games), modest training increase (150 steps, shouldn't overfit on 100k buffer)
- **Result**: FAILED. 0 promotions in 100 iterations (early-stopped). Policy loss mean 2.01, value loss 0.57, value corr 0.40-0.56. Mostly 0.50 ties (7/10 evals) — parity with best model but can't surpass 0.55 threshold.
- **Log**: `training_v3_from230.log`
- **Lesson**: Hyperparameter tuning alone can't break the checkpoint-230 plateau. Consistent 0.50 ties = capacity saturation.

## Experiment 5 (v4): Deeper network, 8 res blocks (from scratch)
- **Architecture**: 8 res blocks, 256 filters
- gamesPerIteration: 100
- mctsSimulations: 320
- trainingStepsPerIteration: 100
- batchSize: 128
- learningRate: 0.001
- replayBufferSize: 100000
- noise: epsilon 0.25→0.05 over 150 iters, alpha 0.3
- temperature: 1.0 for 30 moves, then 0.0
- valueTargetStrategy: terminal
- **Rationale**: Three tuning attempts from checkpoint 230 all failed (0 promotions), suggesting the 5-block architecture has hit a capacity ceiling. Deeper networks can capture more complex positional patterns. Using 320 MCTS sims for better target quality. Conservative training params (100 steps, 100k buffer) that worked well in Experiment 1.
- **Log**: `training_v4_8blocks.log`
- **Result**: 21 promotions in 500 iterations (all 20-0 sweeps when promoted). Policy loss mean 3.18→1.67, value loss 0.93→0.53, value corr peaked at 0.72. Multiple plateau-breaking phases at iters 250, 290, 350, 440, 470-480. Late burst of back-to-back promotions at 470-480 right before the 500-iter cap.
- **Best model**: `checkpoint_480.safetensors`
- **Lesson**: Deeper architecture (5→8 blocks) was the right call — 2.3x promotions vs 5-block (21 vs 9), better final metrics, and remarkable ability to break through plateaus. Network kept finding improvements even after 90-iteration dry spells.

## Plateau Analysis (after Experiment 5)

At the v4 plateau (iter 350+), the metrics are:
- Policy KL ~2.1 — network matches less than half of what MCTS finds. Strong network should be <1.0.
- Value correlation ~0.55-0.65 (peaked 0.72 once) — for strong play, want 0.80+.
- Value loss ~0.54 — capturing only ~46% of learnable value signal.
- 0.50 ties in eval — improvements too small for 20-game evals to detect.

Three possible bottlenecks identified:
1. **Target quality**: 320 MCTS sims produce noisy targets. At the plateau, the network needs to learn subtle distinctions that noisy targets obscure.
2. **Value signal quality**: Terminal outcomes (±1) are noisy — a well-played position gets -1.0 because the player blundered 10 moves later. This limits the value head.
3. **Network capacity**: 256 filters may not be enough width to represent all features simultaneously. On a 5×5 board, more depth has diminishing returns (receptive field already covers everything), but more width could help.

## Planned Next Experiments (in priority order)

### Experiment 6a: 512 MCTS sims (from checkpoint 350)
- **Architecture**: 8 res blocks, 256 filters (same)
- mctsSimulations: 512 (up from 320)
- All other params same as Experiment 5
- **Rationale**: Cheapest test — no architecture change, resume from checkpoint 350. Higher sim count produces cleaner policy targets and more accurate value estimates. At the plateau, marginal improvements need high signal quality to emerge. AlphaZero used 800 sims for chess.
- **Expected**: If target quality is the bottleneck, should see a promotion within 20-30 iterations. If not, we've only lost a few hours.

### Experiment 6b: 512 sims + MCTS root value (from scratch)
- **Architecture**: 8 res blocks, 256 filters
- mctsSimulations: 512
- valueTargetStrategy: mcts (MCTS root value instead of terminal outcome)
- All other params same as Experiment 5
- **Rationale**: Switching value targets mid-training is risky (distribution shift, like the temperature disaster in Exp 2). Start fresh. Continuous value labels give the value head position-specific gradient signal instead of same ±1 for the whole game. Combined with higher sims, should unlock both policy and value heads.
- **When**: Only if Experiment 6a plateaus at the same level.

### Experiment 6c: 384 filters (from scratch)
- **Architecture**: 8 res blocks, 384 filters
- mctsSimulations: 512
- valueTargetStrategy: mcts
- All other params same as Experiment 5
- **Rationale**: Width upgrade — ~2.25x more parameters per conv layer. Only try if 6b shows the network is saturating despite better targets. On a 5×5 board, width matters more than depth (spatial reasoning depth is sufficient, feature breadth may not be).
- **When**: Only if Experiment 6b plateaus.

### Not recommended: 12 res blocks
- On a 5×5 board, receptive field already covers everything with 8 blocks.
- Diminishing returns on depth. Prefer width over depth at this point.

## Key Insights

1. **Masked policy loss** is a clear win — immediate convergence improvement.
2. **Don't change temperature mid-training** — shifts data distribution, network can't adapt.
3. **More training steps + smaller buffer = overfitting** — 250 steps on 50k buffer memorizes within-iteration.
4. **Hyperparameter tuning has diminishing returns** once architecture is the bottleneck.
5. **0.50 eval ties** indicate capacity saturation — network learned everything it can represent.
6. **Higher MCTS sims** (320 vs 256) produce better targets, but benefit requires network capacity to exploit them.
7. **Architecture depth (5→8 blocks)** was the single biggest improvement — doubled promotions and broke through the 5-block ceiling.
8. **Late-stage plateau breakthroughs** are possible — the 8-block network broke through at iters 250, 290, 310, 350 after long stretches of ties/losses.
9. **On a 5×5 board, width may matter more than depth** — receptive field covers the whole board after ~3 blocks, but feature representation capacity (filters) could be limiting.
10. **Terminal value targets are noisy at high skill levels** — a single late-game blunder labels all preceding positions as losses, limiting value head learning.
