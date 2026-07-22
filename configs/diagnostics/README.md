# Calibration diagnostic seed references

`b1_round_0_5_seeds.txt` contains four fixed references to existing Calibration
seeds for the B1 Round 0.5 passive diagnostic replay.

This file is not a Calibration, Development, or Held-out Test split. It is not
included in `split_manifest.json`, does not create new statistical samples, and
must not be used to calculate formal success rates. The diagnostic runner rejects
any addition, removal, replacement, reordering, or automatic seed selection.

`development_d0_5_seeds.txt` is the fixed ten-seed snapshot for passive
Development D0.5 mechanism replay. Every listed seed already belongs to the
registered Development split. It is not a new sample, is excluded from
production metrics, and is intentionally absent from the split manifest.
