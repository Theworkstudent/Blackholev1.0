"""
GARGANTUA DISK -- a real-time N-body accretion disk around a black hole.

PHYSICS SUMMARY (read this before touching constants)
-------------------------------------------------------------------------
The black hole's gravity is modeled with the Paczynski-Wiita potential

    Phi(r) = -G*M_bh / (r - R_S)

instead of plain Newtonian -G*M/r. This is a well-known trick from
accretion-disk astrophysics (Paczynski & Wiita, 1980): it is a pure
Newtonian potential (no geodesics, no metric, dirt cheap to evaluate),
yet it reproduces two exact general-relativistic features of a
non-spinning (Schwarzschild) black hole:

  * the innermost stable circular orbit (ISCO) sits at r = 3*R_S,
  * the potential diverges at r = R_S, standing in for the event horizon.

So every particle here orbits under a force F(r) = -G*M_bh / (r - R_S)^2,
and the instability of circular orbits inside 3*R_S is not scripted --
it falls straight out of that formula. Particles that wander inside the
ISCO spiral in and get swallowed for real.

On top of that:
  * Disk particles feel a softened mutual gravity (real N-body, O(N^2)
    per frame on the GPU) so the disk is genuinely self-gravitating.
    A small two-armed density perturbation is seeded at start-up; because
    the disk rotates faster at small r than at large r (differential
    rotation), that seed reliably winds into trailing spiral arms within
    a few seconds -- the classic spiral density wave.
  * A single massive "interloper" star loops through the disk on a wide
    eccentric orbit and rips tidal tails out of the disk every passage.
  * Color encodes real physics: each particle's blackbody temperature comes
    from a Shakura-Sunyaev-style T(r) ~ r^-3/4 disk profile, then gets
    Doppler-shifted and relativistically beamed (brightness ~ g^3) by its
    orbital velocity, exactly like the redshift factor in real accretion
    disk images (approaching gas looks bluer and brighter, receding gas
    looks redder and dimmer).

Run with a Python 3.10-3.12 interpreter that has `taichi` installed:
    python sim.py
"""

import math

import numpy as np
import taichi as ti

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower N_DISK if it crawls.")

# =========================================================================
# CONSTANTS  (all in dimensionless "sim units": G = 1 implicitly folded
# into the GM products below, so only the combination G*M ever appears)
# =========================================================================

# --- Screen / window -----------------------------------------------------
WIN_W, WIN_H = 1280, 800          # window resolution (pixels)

# --- Black hole ------------------------------------------------------------
R_S = 6.0                          # Schwarzschild radius (event horizon), sim units
C_LIGHT = 40.0                      # speed of light in sim units (sets relativistic scale)
GM_BH = R_S * C_LIGHT ** 2 / 2.0    # G*M_bh, derived so R_S = 2*G*M/c^2 holds exactly
R_ISCO = 3.0 * R_S                 # innermost stable circular orbit (exact for this potential)
R_CAPTURE = R_S * 1.15             # particles crossing this radius are "eaten" and respawned
                                    # (kept safely above R_S so 1/(r-R_S) never blows up)

# --- Disk of test particles ------------------------------------------------
N_DISK = 11000                     # number of disk particles (push higher if your GPU allows)
DISK_R_IN = R_ISCO * 1.05          # disk starts just outside the ISCO
DISK_R_OUT = 150.0                 # disk outer edge
DENSITY_POWER = 1.7                # >1 concentrates more particles at small r (real disks are
                                    # denser inside), sampled as r = R_IN + (R_OUT-R_IN)*u^POWER
VEL_NOISE = 0.035                  # fractional random noise added to circular velocity, so
                                    # orbits are slightly eccentric instead of perfect circles
SPIRAL_ARM_FRACTION = 0.6          # fraction of particles seeded directly onto the 2 spiral arms
SPIRAL_ARM_WIDTH = 0.6              # angular scatter (radians) around each arm, i.e. arm thickness
SPIRAL_PITCH = 2.2                  # controls how tightly the seed spiral is wound (log-spiral rate)

# --- Disk self-gravity (real N-body term, this is what makes it "N-body") --
SELF_GRAVITY_GM = 2.6              # lumped G*mass contributed by *each* disk particle
SELF_SOFTENING = 3.0                # softening length so close encounters don't slingshot

# --- The interloper star (drives tidal tails) ------------------------------
PERT_GM = 950.0                    # G*mass of the interloper (a few hundred disk-particles' worth)
PERT_PERIAPSIS = 68.0               # closest approach to the black hole
PERT_APOAPSIS = 250.0                # farthest excursion
PERT_SOFTENING = 4.0

# --- User-controlled mouse "gravity finger" (right mouse button) ----------
MOUSE_PULL_GM = 2600.0
MOUSE_SOFTENING = 8.0

# --- Mouse spawn (left mouse button) --------------------------------------
SPAWN_PER_FRAME = 40                 # how many particles get relocated to the cursor per frame

# --- Integration -----------------------------------------------------------
DT_BASE = 0.05                      # base timestep (sim time units)
TIME_SCALE_MIN, TIME_SCALE_MAX = 0.1, 6.0
TIME_SCALE_STEP = 1.25              # multiplicative step for the [ and ] keys

# --- Color / temperature ----------------------------------------------------
T0 = 26000.0                        # disk temperature normalization (Kelvin), tuned for looks
DOPPLER_STRENGTH = 1.0              # how strongly velocity-toward-viewer shifts color/brightness
BEAMING_EXPONENT = 3.0               # relativistic beaming exponent (brightness ~ g^BEAMING_EXPONENT)
BRIGHTNESS_SCALE = 520.0             # overall disk brightness normalization (pre-tonemap)

# --- Rendering / camera projection ------------------------------------------
VIEW_SCALE = 3.55                   # pixels per sim-unit
TILT_Y = 0.46                        # vertical squash factor -> gives the disk a cinematic tilt
SPLAT_RADIUS = 3                    # half-width (in pixels) of each particle's glow splat
SPLAT_SIGMA = 0.45                   # softness of the glow falloff (smaller = softer/bigger glow)
TRAIL_DECAY_SLOW = 0.965             # framebuffer decay per frame -> long glowing trails
TRAIL_DECAY_FAST = 0.85              # shorter trails (toggle with T)
N_STARS = 400                        # decorative background starfield points

print(
    "\n".join(
        [
            "",
            "=== GARGANTUA DISK ===",
            "Controls:",
            "  Left mouse (hold)   spawn a stream of particles at the cursor",
            "  Right mouse (hold)  your cursor pulls nearby gas (a gravity 'finger')",
            "  SPACE               pause / resume",
            "  ]  /  [             speed up / slow down time",
            "  G                   toggle disk self-gravity (spiral arms) on/off",
            "  T                   toggle short/long particle trails",
            "  R                   reset the simulation",
            "  ESC                 quit",
            "",
        ]
    )
)

# =========================================================================
# FIELDS
# =========================================================================
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_DISK)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N_DISK)

pert_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
pert_vel = ti.Vector.field(2, dtype=ti.f32, shape=())

mouse_world = ti.Vector.field(2, dtype=ti.f32, shape=())
mouse_pull_on = ti.field(dtype=ti.i32, shape=())

self_gravity_on = ti.field(dtype=ti.i32, shape=())
time_scale = ti.field(dtype=ti.f32, shape=())

horizon_ring = ti.Vector.field(2, dtype=ti.f32, shape=160)   # static event-horizon ring points
isco_ring = ti.Vector.field(2, dtype=ti.f32, shape=200)      # static ISCO marker ring points
star_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)   # decorative background stars
star_bright = ti.field(dtype=ti.f32, shape=N_STARS)

framebuffer = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))   # linear HDR accumulator
display_img = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))   # tonemapped, shown on screen


# =========================================================================
# PHYSICS HELPERS
# =========================================================================
@ti.func
def circular_speed(r):
    # Circular-orbit speed for the Paczynski-Wiita potential Phi = -GM/(r-R_S):
    # centripetal accel v^2/r must equal the potential's radial force GM/(r-R_S)^2
    return ti.sqrt(GM_BH * r) / (r - R_S)


@ti.func
def sample_disk_particle(i: ti.i32, at: ti.template()):
    # Draws a fresh (position, velocity) for particle i somewhere in the disk
    # annulus, on a near-circular prograde orbit. A majority of particles are
    # placed directly onto one of two logarithmic-spiral density bands (with
    # some angular scatter for thickness) rather than uniformly at random, so
    # the disk starts with a genuine, strongly visible density contrast
    # between arm and inter-arm regions. Differential rotation (inner gas
    # orbits faster than outer gas) then continuously winds/re-shapes this
    # into the classic trailing spiral density wave.
    u = ti.random()
    r = DISK_R_IN + (DISK_R_OUT - DISK_R_IN) * ti.pow(u, DENSITY_POWER)

    theta = ti.random() * 2.0 * math.pi
    if ti.random() < SPIRAL_ARM_FRACTION:
        arm = 0.0 if ti.random() < 0.5 else math.pi
        arm_theta = arm + SPIRAL_PITCH * ti.log(r / DISK_R_IN)
        theta = arm_theta + (ti.random() - 0.5) * SPIRAL_ARM_WIDTH

    dirv = ti.Vector([ti.cos(theta), ti.sin(theta)])
    tang = ti.Vector([-ti.sin(theta), ti.cos(theta)])  # counter-clockwise = prograde

    speed = circular_speed(r) * (1.0 + VEL_NOISE * (ti.random() - 0.5) * 2.0)
    radial_kick = VEL_NOISE * speed * (ti.random() - 0.5) * 2.0

    pos[i] = dirv * r + at
    vel[i] = tang * speed + dirv * radial_kick


@ti.kernel
def init_disk():
    zero = ti.Vector([0.0, 0.0])
    for i in range(N_DISK):
        sample_disk_particle(i, zero)

    # interloper star: wide eccentric prograde orbit, vis-viva initial speed
    a_semi = 0.5 * (PERT_PERIAPSIS + PERT_APOAPSIS)
    v_apo = ti.sqrt(GM_BH * (2.0 / PERT_APOAPSIS - 1.0 / a_semi))
    pert_pos[None] = ti.Vector([-PERT_APOAPSIS, 0.0])
    pert_vel[None] = ti.Vector([0.0, -v_apo])

    time_scale[None] = 1.0
    self_gravity_on[None] = 1


@ti.kernel
def init_decorations():
    for k in range(horizon_ring.shape[0]):
        a = k / horizon_ring.shape[0] * 2.0 * math.pi
        horizon_ring[k] = ti.Vector([ti.cos(a), ti.sin(a)]) * R_S
    for k in range(isco_ring.shape[0]):
        a = k / isco_ring.shape[0] * 2.0 * math.pi
        isco_ring[k] = ti.Vector([ti.cos(a), ti.sin(a)]) * R_ISCO
    # Scatter stars over a world-space rectangle that fully covers the window
    # (accounting for the tilt squash), not just a small disc near the hole.
    half_x = (WIN_W * 0.5 / VIEW_SCALE) * 1.15
    half_y = (WIN_H * 0.5 / (VIEW_SCALE * TILT_Y)) * 1.15
    for k in range(N_STARS):
        star_pos[k] = ti.Vector(
            [(ti.random() * 2.0 - 1.0) * half_x, (ti.random() * 2.0 - 1.0) * half_y]
        )
        star_bright[k] = ti.random() * 0.25 + 0.05


# =========================================================================
# PHYSICS UPDATE
# =========================================================================
@ti.kernel
def step_perturber(dt: ti.f32):
    r_vec = pert_pos[None]
    r = r_vec.norm() + 1e-3
    r = ti.max(r, R_CAPTURE)  # keep the interloper outside the horizon, always
    accel = -GM_BH / (r - R_S) ** 2 * (r_vec / r)
    pert_vel[None] += accel * dt
    pert_pos[None] += pert_vel[None] * dt


@ti.kernel
def step_disk(dt: ti.f32, use_self_gravity: ti.i32, use_mouse: ti.i32):
    p_pos = pert_pos[None]
    m_pos = mouse_world[None]

    for i in range(N_DISK):
        r_vec = pos[i]
        r = r_vec.norm() + 1e-3

        # 1) Black hole gravity (Paczynski-Wiita -- exact ISCO/horizon stand-in)
        r_safe = ti.max(r, R_CAPTURE)
        accel = -GM_BH / (r_safe - R_S) ** 2 * (r_vec / r_safe)

        # 2) Disk self-gravity: real O(N^2) N-body sum over every other particle,
        #    softened so close pairs don't slingshot to infinity.
        if use_self_gravity:
            local_accel = ti.Vector([0.0, 0.0])
            for j in range(N_DISK):
                diff = pos[j] - r_vec
                d = diff.norm() + SELF_SOFTENING
                local_accel += SELF_GRAVITY_GM * diff / (d * d * d)
            accel += local_accel

        # 3) The interloper star's tidal pull
        diff_p = p_pos - r_vec
        d_p = diff_p.norm() + PERT_SOFTENING
        accel += PERT_GM * diff_p / (d_p * d_p * d_p)

        # 4) The user's mouse acting as a small point mass ("gravity finger")
        if use_mouse:
            diff_m = m_pos - r_vec
            d_m = diff_m.norm() + MOUSE_SOFTENING
            accel += MOUSE_PULL_GM * diff_m / (d_m * d_m * d_m)

        # Semi-implicit (symplectic) Euler: velocity first, then position.
        vel[i] += accel * dt
        pos[i] += vel[i] * dt

        # Respawn anything captured by the black hole or flung off to infinity,
        # so the disk stays populated forever (fresh gas falling in from afar).
        new_r = pos[i].norm()
        if new_r < R_CAPTURE or new_r > DISK_R_OUT * 2.2:
            sample_disk_particle(i, ti.Vector([0.0, 0.0]))


@ti.kernel
def spawn_burst(start: ti.i32, count: ti.i32, cx: ti.f32, cy: ti.f32):
    center = ti.Vector([cx, cy])
    r = ti.max(center.norm(), DISK_R_IN)
    theta = ti.atan2(center.y, center.x)
    tang = ti.Vector([-ti.sin(theta), ti.cos(theta)])
    speed = circular_speed(r)
    for k in range(count):
        i = (start + k) % N_DISK
        jitter = (ti.Vector([ti.random(), ti.random()]) - 0.5) * 3.0
        pos[i] = center + jitter
        # give freshly spawned gas a gentle outward "injection" kick plus the
        # local orbital velocity so it blends smoothly into the flow
        outward = jitter.normalized(1e-3)
        vel[i] = tang * speed * 0.9 + outward * speed * 0.25


# =========================================================================
# RENDERING
# =========================================================================
@ti.func
def world_to_pixel(w):
    px = WIN_W * 0.5 + w.x * VIEW_SCALE
    py = WIN_H * 0.5 + w.y * VIEW_SCALE * TILT_Y
    return px, py


@ti.func
def blackbody_rgb(temp: ti.f32):
    # Tanner Helland's polynomial fit to the Planckian locus (Kelvin -> sRGB)
    t = ti.min(ti.max(temp, 1000.0), 40000.0) / 100.0
    r, g, b = 0.0, 0.0, 0.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * ti.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ti.pow(t - 60.0, -0.1332047592)
        g = 288.1221695283 * ti.pow(t - 60.0, -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * ti.log(t - 10.0) - 305.0447927307
    rgb = ti.max(ti.Vector([r, g, b]) / 255.0, 0.0)
    lum = rgb.dot(ti.Vector([0.299, 0.587, 0.114]))
    return ti.max(lum + (rgb - lum) * 1.6, 0.0)  # push saturation up for a punchier palette


@ti.func
def splat(px: ti.f32, py: ti.f32, color: ti.template()):
    ix, iy = ti.i32(px), ti.i32(py)
    for dx, dy in ti.ndrange((-SPLAT_RADIUS, SPLAT_RADIUS + 1), (-SPLAT_RADIUS, SPLAT_RADIUS + 1)):
        x, y = ix + dx, iy + dy
        if 0 <= x < WIN_W and 0 <= y < WIN_H:
            falloff = ti.exp(-SPLAT_SIGMA * (dx * dx + dy * dy))
            framebuffer[x, y] += color * falloff


@ti.kernel
def decay_framebuffer(decay: ti.f32):
    for x, y in framebuffer:
        framebuffer[x, y] *= decay


@ti.kernel
def render_stars():
    for k in range(N_STARS):
        px, py = world_to_pixel(star_pos[k])
        b = star_bright[k]
        splat(px, py, ti.Vector([b, b, b * 1.05]))


@ti.kernel
def render_rings():
    for k in range(horizon_ring.shape[0]):
        px, py = world_to_pixel(horizon_ring[k])
        splat(px, py, ti.Vector([2.2, 1.7, 1.0]))
    for k in range(isco_ring.shape[0]):
        px, py = world_to_pixel(isco_ring[k])
        splat(px, py, ti.Vector([0.10, 0.09, 0.07]))


@ti.kernel
def render_perturber():
    px, py = world_to_pixel(pert_pos[None])
    for dx, dy in ti.ndrange((-5, 6), (-5, 6)):
        x, y = ti.i32(px) + dx, ti.i32(py) + dy
        if 0 <= x < WIN_W and 0 <= y < WIN_H:
            falloff = ti.exp(-0.25 * (dx * dx + dy * dy))
            framebuffer[x, y] += ti.Vector([2.6, 2.8, 3.2]) * falloff


@ti.kernel
def render_disk():
    for i in range(N_DISK):
        r_vec = pos[i]
        r = ti.max(r_vec.norm(), R_ISCO * 1.001)

        # Shakura-Sunyaev-like flux/temperature profile (same shape as a
        # thin accretion disk's radiated flux, F ~ r^-3 with a torque-free
        # inner boundary that forces both flux and temperature to zero at
        # the ISCO).
        flux_shape = ti.max(1.0 - ti.sqrt(R_ISCO / r), 1e-6)
        temp = T0 * ti.pow(r, -0.75) * ti.pow(flux_shape, 0.25)
        flux = ti.pow(r, -1.7) * flux_shape

        # Doppler shift + relativistic beaming from the component of orbital
        # velocity pointed at the viewer (approximated here as the world +x
        # axis, i.e. the viewer sits off to one side of the disk plane).
        g = 1.0 + DOPPLER_STRENGTH * vel[i].x / C_LIGHT
        g = ti.max(g, 0.15)

        color = blackbody_rgb(temp * g)
        brightness = BRIGHTNESS_SCALE * flux * ti.pow(g, BEAMING_EXPONENT)

        px, py = world_to_pixel(r_vec)
        splat(px, py, color * brightness)


@ti.kernel
def tonemap():
    for x, y in framebuffer:
        c = framebuffer[x, y]
        lum = ti.max(c[0], ti.max(c[1], c[2]))
        out = c / (1.0 + lum)
        display_img[x, y] = ti.pow(ti.max(out, 0.0), 1.0 / 2.2)


# =========================================================================
# MAIN LOOP
# =========================================================================
def pixel_to_world(cx, cy):
    # Inverse of world_to_pixel, using GGUI's normalized (0..1, y-up) cursor
    # coordinates. Matches the same convention as our framebuffer indexing.
    wx = (cx * WIN_W - WIN_W * 0.5) / VIEW_SCALE
    wy = (cy * WIN_H - WIN_H * 0.5) / (VIEW_SCALE * TILT_Y)
    return wx, wy


def main():
    init_decorations()
    init_disk()

    window = ti.ui.Window("Gargantua Disk", (WIN_W, WIN_H), vsync=True)
    canvas = window.get_canvas()

    paused = False
    trail_fast = False
    spawn_cursor = 0

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == "r":
                init_disk()
            elif e.key == "g":
                self_gravity_on[None] = 1 - self_gravity_on[None]
            elif e.key == "t":
                trail_fast = not trail_fast
            elif e.key == "]":
                time_scale[None] = float(
                    np.clip(time_scale[None] * TIME_SCALE_STEP, TIME_SCALE_MIN, TIME_SCALE_MAX)
                )
            elif e.key == "[":
                time_scale[None] = float(
                    np.clip(time_scale[None] / TIME_SCALE_STEP, TIME_SCALE_MIN, TIME_SCALE_MAX)
                )

        cx, cy = window.get_cursor_pos()
        wx, wy = pixel_to_world(cx, cy)
        mouse_world[None] = [wx, wy]
        mouse_pull_on[None] = 1 if window.is_pressed(ti.ui.RMB) else 0

        if window.is_pressed(ti.ui.LMB):
            spawn_burst(spawn_cursor, SPAWN_PER_FRAME, wx, wy)
            spawn_cursor = (spawn_cursor + SPAWN_PER_FRAME) % N_DISK

        if not paused:
            dt = DT_BASE * time_scale[None]
            step_perturber(dt)
            step_disk(dt, self_gravity_on[None], mouse_pull_on[None])

        decay = TRAIL_DECAY_FAST if trail_fast else TRAIL_DECAY_SLOW
        decay_framebuffer(decay)
        render_stars()
        render_rings()
        render_perturber()
        render_disk()
        tonemap()

        canvas.set_image(display_img)
        window.show()


if __name__ == "__main__":
    main()
