# Isaac Sim Validation Readiness

This directory records the Isaac Sim environment-readiness baseline for the
Franka Panda manipulation project.

## Current Status

The following workflow has been verified locally:

- Isaac Sim 5.1 standalone installation;
- official Franka pick-and-place example;
- headless execution;
- viewport updates disabled;
- CPU worker threads limited to six;
- CPU physics simulation;
- damped-least-squares inverse kinematics;
- automatic termination after task completion;
- successful process exit with code 0.

The smoke test completed the official pick-and-place sequence in 280
simulation update steps.

## Run

Isaac Sim provides its own Python entry point. From the repository root,
run the following command in PowerShell:

    C:\isaacsim\python.bat `
        isaacsim\scripts\franka_pick_place_headless.py `
        --device cpu `
        --ik-method damped-least-squares `
        --max-steps 30000

The local Isaac Sim installation path may differ on another machine.

## Files

- `scripts/franka_pick_place_headless.py`: headless Franka smoke-test script;
- `environment/isaac_sim_environment.txt`: validated software and hardware environment;
- `results/smoke_test_result.json`: structured result from the verified run.

## Scope and Limitations

This module currently verifies only that the local hardware and Isaac Sim
installation can execute a lightweight, single-environment Franka
pick-and-place task without GUI rendering.

It does not yet demonstrate:

- transfer of the MuJoCo controller;
- transfer of the MuJoCo RGB-D perception pipeline;
- transfer of a reinforcement-learning policy;
- matched task configurations across simulators;
- cross-simulator success-rate comparison;
- dynamics or contact-model equivalence.

Formal MuJoCo-to-Isaac-Sim validation will be implemented after the MuJoCo
task interface, algorithm, and evaluation protocol are finalized.

## Hardware Boundary

The validated machine has:

- Intel Core i7-10700 CPU;
- 16 GB system memory;
- NVIDIA GeForce RTX 4060;
- 8 GB GPU VRAM.

GUI execution caused severe system resource pressure. The headless
configuration completed the smoke test without the previous CPU and memory
saturation.

Isaac Sim is therefore treated as a lightweight validation platform rather
than the primary training or large-scale evaluation platform.
