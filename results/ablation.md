_30 seeds per cell, attempt budget 3; success = simulator ground truth at episode end; brackets are 95% Wilson intervals; 0 episodes errored._

### Success rate by fault profile (all tasks pooled)

| Fault profile | No verifier | Verifier + blind retry | Verifier + corrected retry | Verifier + damped correction | Oracle verifier + corrected retry |
|---|---|---|---|---|---|
| nominal (n=90) | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] | 100% [96-100] |
| noisy (n=90) | 34% [25-45] | 66% [55-75] | 52% [42-62] | 58% [47-67] | 54% [44-64] |
| biased (n=90) | 33% [24-44] | 33% [24-44] | 98% [92-99] | 99% [94-100] | 97% [91-99] |
| slippery (n=90) | 52% [42-62] | 78% [68-85] | 72% [62-80] | 73% [63-81] | 68% [58-77] |

### Per task

| Task | Fault profile | No verifier | Verifier + blind retry | Verifier + corrected retry | Verifier + damped correction | Oracle verifier + corrected retry |
|---|---|---|---|---|---|---|
| pick_place | nominal | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] |
| pick_place | noisy | 67% [49-81] | 97% [83-99] | 97% [83-99] | 97% [83-99] | 100% [89-100] |
| pick_place | biased | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] |
| pick_place | slippery | 63% [46-78] | 87% [70-95] | 83% [66-93] | 87% [70-95] | 80% [63-90] |
| stack | nominal | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] |
| stack | noisy | 27% [14-44] | 47% [30-64] | 37% [22-54] | 50% [33-67] | 40% [25-58] |
| stack | biased | 0% [0-11] | 0% [0-11] | 93% [79-98] | 97% [83-99] | 90% [74-97] |
| stack | slippery | 53% [36-70] | 70% [52-83] | 67% [49-81] | 70% [52-83] | 67% [49-81] |
| two_step | nominal | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] | 100% [89-100] |
| two_step | noisy | 10% [3-26] | 53% [36-70] | 23% [12-41] | 27% [14-44] | 23% [12-41] |
| two_step | biased | 0% [0-11] | 0% [0-11] | 100% [89-100] | 100% [89-100] | 100% [89-100] |
| two_step | slippery | 40% [25-58] | 77% [59-88] | 67% [49-81] | 63% [46-78] | 57% [39-73] |

### Paired comparisons (same seeds, exact McNemar)

| Fault profile | Verifier(blind) vs none | Corrected vs blind retry | Perception vs oracle verifier |
|---|---|---|---|
| nominal | 0 vs 0 (p=1, n=90) | 0 vs 0 (p=1, n=90) | 0 vs 0 (p=1, n=90) |
| noisy | 28 vs 0 (p=7.45e-09, n=90) | 3 vs 15 (p=0.00754, n=90) | 1 vs 3 (p=0.625, n=90) |
| biased | 0 vs 0 (p=1, n=90) | 58 vs 0 (p=6.94e-18, n=90) | 1 vs 0 (p=1, n=90) |
| slippery | 23 vs 0 (p=2.38e-07, n=90) | 0 vs 5 (p=0.0625, n=90) | 5 vs 1 (p=0.219, n=90) |

Each cell reads `episodes only the first arm solved vs only the second arm solved`.

### Mean attempts per episode and false successes

| Fault profile | No verifier | Verifier + blind retry | Verifier + corrected retry | Verifier + damped correction | Oracle verifier + corrected retry |
|---|---|---|---|---|---|
| nominal | 1.33 att, 0 false-ok | 1.33 att, 0 false-ok | 1.33 att, 0 false-ok | 1.33 att, 0 false-ok | 1.33 att, 0 false-ok |
| noisy | 1.33 att, 59 false-ok | 2.59 att, 1 false-ok | 2.66 att, 1 false-ok | 2.61 att, 1 false-ok | 2.64 att, 0 false-ok |
| biased | 1.33 att, 60 false-ok | 2.67 att, 0 false-ok | 2.03 att, 0 false-ok | 2.64 att, 0 false-ok | 2.03 att, 0 false-ok |
| slippery | 1.33 att, 43 false-ok | 2.17 att, 0 false-ok | 2.20 att, 0 false-ok | 2.19 att, 0 false-ok | 2.28 att, 0 false-ok |

### Perception verifier vs ground truth (attempt level)

accepted-and-correct 1193, **false accepts 3**, rejected-and-correct 1121, **false rejects 1** (of 2318 verdicts).
