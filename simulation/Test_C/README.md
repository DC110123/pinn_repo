# Test_C

Time-series "digital twin" simulation for the universal PINN model, generated
by `panel_simulation.py`.

## Contents

| File | Description |
| `C1.png` | MPP tracking result for sampled panel 1 |
| `C2.png` | MPP tracking result for sampled panel 2 |
| `C3.png` | MPP tracking result for sampled panel 3 |
| `panel_simulation.py` | Script used to run the time-series simulation and generate the plots in this folder |

## What this validates

While `Test_A` checks individual I-V curve *shape* and `Test_B` checks
aggregate accuracy across a large static population, `Test_C` validates the
model's performance as a high-speed **"digital twin"** in a realistic,
time-series environment.

This simulation subjects **10 randomly sampled panels** to a full **12-hour
weather profile**, including a simulated **"cloud notch" event** (a rapid
transient shadow, testing the model's response to sudden irradiance drops).
For each panel, the **Maximum Power Point (MPP)** is tracked in real time and
compared against a traditional iterative physics solver, measuring both
accuracy and responsiveness under rapidly changing environmental inputs.

This test proves the model can maintain physical consistency (correct MPP
tracking, no unphysical artifacts) even under fast-changing conditions —
making it suitable for live asset monitoring and energy forecasting use
cases, not just static/offline evaluation.

**`C1.png` / `C2.png` / `C3.png`**

Each plot shows one of the 10 sampled panels' MPP trajectory across the
12-hour profile, comparing the PINN's real-time tracked MPP against the
reference physics solver's MPP — including behavior through the cloud notch
transient.

> Note: only 3 of the 10 simulated panels appear to be plotted here (C1–C3).
> Confirm whether the remaining 7 are generated elsewhere / omitted
> intentionally, or whether this folder should include all 10.

## Reproducing these results

#bash
```
python panel_simulation.py
```

> Update this section with actual usage — e.g. how the 12-hour weather
> profile is generated/configured, how the cloud notch event is parameterized,
> and how the 10 panels are sampled (random seed, panel pool, etc.).

## Notes

- **Panel selection**: confirm whether the 10 simulated panels are drawn from
  the held-out validation set or the full database.
- **Responsiveness metric**: consider adding a quantitative measure (e.g. lag
  time or MPP error during the cloud notch transient specifically) alongside
  the qualitative plots, to make the "digital twin" claim easier to verify at
  a glance.