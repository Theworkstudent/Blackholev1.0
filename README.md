# Blackhole v1.0

Real-time GPU black hole simulations in Python, built with [Taichi](https://www.taichi-lang.org/).
Two separate simulations live here, each in a single file.

|                     | `kerr_blackhole.py`                              | `sim.py`                                   |
| ------------------- | ------------------------------------------------- | ------------------------------------------- |
| **What it is**      | A GPU ray tracer that actually integrates light through curved spacetime | A 2D N-body accretion disk with real self-gravity |
| **Physics engine**  | Full Kerr geodesic equations (Boyer-Lindquist coordinates, Carter's constants), 4th-order Runge-Kutta | Paczyński-Wiita pseudo-Newtonian potential (exact Schwarzschild ISCO/horizon) |
| **Disk model**      | Volumetric, turbulent Novikov-Thorne-like disk with Doppler beaming and gravitational redshift | Thousands of particles under mutual gravity, seeded into a spiral, disrupted by a passing star |
| **Look**            | A single black hole viewed from an orbiting camera, with a lensed starfield background | A top-down glowing particle disk with motion trails |

## Requirements

- Python 3.10–3.12 (Taichi does not yet support 3.13+)
- [`taichi`](https://pypi.org/project/taichi/) and `numpy`
- An NVIDIA GPU is strongly recommended (both scripts fall back to CPU automatically, but at much lower framerates)

```bash
python -m venv .venv
.venv\Scripts\activate      # or: source .venv/bin/activate on macOS/Linux
pip install taichi numpy
```

## Running

```bash
python kerr_blackhole.py
python sim.py
```

Each opens its own window; there is no on-screen UI, all interaction is via mouse/keyboard (see below). Both print nothing further to the console once running — that's normal, the window itself is where everything happens.

## `kerr_blackhole.py` — Kerr ray tracer

Fires one photon per pixel *backward* from the camera and integrates the real Kerr null-geodesic equations to see where each ray actually ends up: swallowed by the event horizon, absorbed into the accretion disk, or escaped to the (gravitationally lensed) starfield. The disk has genuine volumetric thickness — it's ray-marched as a turbulent, semi-transparent gas torus, not a flat plane — plus a faint afterglow from matter plunging between the ISCO and the horizon.

| Control          | Effect                                      |
| ---------------- | -------------------------------------------- |
| Left-drag        | Orbit the camera (azimuth / inclination)     |
| Up / Down arrows | Zoom (dolly camera distance)                 |
| A / D            | Decrease / increase black-hole spin `a` (0–0.998) |
| W / S            | Decrease / increase disk color-temperature multiplier |
| Esc              | Quit                                         |

## `sim.py` — N-body accretion disk

A companion piece: thousands of disk particles under real mutual gravity (not just orbiting a fixed potential), seeded with a two-armed spiral that differential rotation winds into a genuine spiral density wave, plus a massive interloper star on an eccentric orbit that rips tidal tails through the disk on every pass. Color is driven by a Shakura-Sunyaev-style temperature profile, Doppler-shifted and relativistically beamed by each particle's real orbital velocity.

| Control            | Effect                                          |
| ------------------- | ------------------------------------------------ |
| Left mouse (hold)   | Spawn a stream of particles at the cursor        |
| Right mouse (hold)  | Your cursor pulls nearby gas (a gravity "finger") |
| Space               | Pause / resume                                   |
| `]` / `[`           | Speed up / slow down time                        |
| G                   | Toggle disk self-gravity (spiral arms) on/off    |
| T                   | Toggle short/long particle trails                |
| R                   | Reset the simulation                             |
| Esc                 | Quit                                             |
