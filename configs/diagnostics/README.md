# Calibration diagnostic seed references

`b1_round_0_5_seeds.txt` contains four fixed references to existing Calibration
seeds for the B1 Round 0.5 passive diagnostic replay.

This file is not a Calibration, Development, or Held-out Test split. It is not
included in `split_manifest.json`, does not create new statistical samples, and
must not be used to calculate formal success rates. The diagnostic runner rejects
any addition, removal, replacement, reordering, or automatic seed selection.
