"""
Real-time Kerr black hole ray-tracer (Taichi GPU backend).

Fires one photon per pixel backward from an orbiting camera, integrates the
full Kerr null-geodesic equations (Boyer-Lindquist coordinates, Carter's
separated constants E=1, L, Q) with 4th-order Runge-Kutta, and volumetrically
marches each ray through a puffy, turbulent Novikov-Thorne-like accretion
disk (real vertical thickness and semi-transparency, not a flat opaque
plane) plus a faint "plunging matter" afterglow between the ISCO and the
event horizon, or out to a checkerboard celestial sphere if it escapes.

Controls (no on-screen UI):
    Left-drag     orbit camera (azimuth / inclination)
    Scroll wheel  zoom (dolly camera distance)
    A / D         decrease / increase black-hole spin a  (0 .. 0.998)
    W / S         decrease / increase disk color-temperature multiplier
    Esc           quit
"""

import math
import time
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 512, 512
MAX_STEPS = 400
BASE_DL = 0.09
ESCAPE_R = 45.0
DISK_R_OUT = 16.0
FOV = math.radians(50.0)
TAN_HALF_FOV = math.tan(FOV / 2.0)
T0 = 2.0e4          # disk temperature normalization (Kelvin)
BRIGHT_SCALE = 650.0  # disk flux -> pre-tonemap brightness normalization

# --- Disk volume (gives the disk real vertical thickness instead of being a
# mathematically flat, zero-width plane) ---------------------------------
DISK_H0 = 0.05          # scale height coefficient: local half-thickness ~= DISK_H0 * r
DISK_OPACITY = 0.55      # optical depth per unit (density * path length); higher = more opaque
TRANSMIT_CUTOFF = 0.03   # once this little light can still get through, treat the ray as blocked
PUFF_AMP = 0.35          # how much the local scale height billows via turbulence (0..~1)

# --- Plunging-matter afterglow: a faint, heavily redshifted glow from gas
# that has passed the ISCO and is free-falling toward the horizon (real disks
# don't cleanly cut off at the ISCO the way the idealized Novikov-Thorne flux
# does; a little residual light continues, fading fast) --------------------
PLUNGE_GLOW_SCALE = 55.0
PLUNGE_TEMP = 9000.0     # reference color temperature of the plunging gas, before redshift

# ----------------------------------------------------------------------
# Persistent state
# ----------------------------------------------------------------------
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

spin = ti.field(dtype=ti.f32, shape=())
temp_mult = ti.field(dtype=ti.f32, shape=())
sim_time = ti.field(dtype=ti.f32, shape=())

cam_azimuth = ti.field(dtype=ti.f32, shape=())
cam_incl = ti.field(dtype=ti.f32, shape=())
cam_dist = ti.field(dtype=ti.f32, shape=())

spin[None] = 0.9
temp_mult[None] = 1.0
sim_time[None] = 0.0
cam_azimuth[None] = 0.6
cam_incl[None] = 1.0
cam_dist[None] = 42.0


# ----------------------------------------------------------------------
# Kerr metric building blocks (G = c = M = 1)
# ----------------------------------------------------------------------
@ti.func
def horizon_radius(a: ti.f32) -> ti.f32:
    return 1.0 + ti.sqrt(ti.max(1.0 - a * a, 0.0))


@ti.func
def isco_radius(a: ti.f32) -> ti.f32:
    aa = ti.min(ti.max(a, -0.9999), 0.9999)
    z1 = 1.0 + ti.pow(ti.max(1.0 - aa * aa, 1e-6), 1.0 / 3.0) * (
        ti.pow(1.0 + aa, 1.0 / 3.0) + ti.pow(1.0 - aa, 1.0 / 3.0)
    )
    z2 = ti.sqrt(3.0 * aa * aa + z1 * z1)
    return 3.0 + z2 - ti.sqrt(ti.max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 1e-6))


@ti.func
def kerr_omega(r: ti.f32, a: ti.f32) -> ti.f32:
    # Keplerian angular velocity of prograde circular equatorial orbit
    return 1.0 / (ti.pow(r, 1.5) + a)


@ti.func
def kerr_ut(r: ti.f32, a: ti.f32) -> ti.f32:
    # u^t for a circular equatorial geodesic (Bardeen-Press-Teukolsky 1972)
    disc = ti.max(ti.pow(r, 1.5) - 3.0 * ti.sqrt(r) + 2.0 * a, 1e-6)
    return (ti.pow(r, 1.5) + a) / (ti.pow(r, 0.75) * ti.sqrt(disc))


# ----------------------------------------------------------------------
# Geodesic right-hand side. State = (r, theta, phi, p_r, p_theta)
# Conserved per-ray parameters: a (spin), L (ang. momentum), Q (Carter const)
# Energy at infinity fixed to E = 1.
# ----------------------------------------------------------------------
@ti.func
def geodesic_rhs(r, th, pr, pth, a, L, Q):
    sth = ti.sin(th)
    if ti.abs(sth) < 1e-4:
        sth = 1e-4 if sth >= 0.0 else -1e-4
    cth = ti.cos(th)
    sth2 = sth * sth

    sigma = r * r + a * a * cth * cth
    delta = r * r - 2.0 * r + a * a

    P = r * r + a * a - a * L
    K = (L - a) * (L - a) + Q

    dr = delta / sigma * pr
    dth = pth / sigma
    dphi = (a * P / delta - a + L / sth2) / sigma

    dRdr = 4.0 * r * P - (2.0 * r - 2.0) * K
    dpr = dRdr / (2.0 * sigma * delta) - pr * pr * (2.0 * r - 2.0) / sigma

    dThdth = L * L * cth / (sth2 * sth) - a * a * sth * cth
    dpth = dThdth / sigma

    return dr, dth, dphi, dpr, dpth


@ti.func
def rk4_step(r, th, phi, pr, pth, a, L, Q, dl):
    r1, th1, phi1, pr1, pth1 = geodesic_rhs(r, th, pr, pth, a, L, Q)
    r2, th2, phi2, pr2, pth2 = geodesic_rhs(
        r + 0.5 * dl * r1, th + 0.5 * dl * th1, pr + 0.5 * dl * pr1, pth + 0.5 * dl * pth1, a, L, Q
    )
    r3, th3, phi3, pr3, pth3 = geodesic_rhs(
        r + 0.5 * dl * r2, th + 0.5 * dl * th2, pr + 0.5 * dl * pr2, pth + 0.5 * dl * pth2, a, L, Q
    )
    r4, th4, phi4, pr4, pth4 = geodesic_rhs(
        r + dl * r3, th + dl * th3, pr + dl * pr3, pth + dl * pth3, a, L, Q
    )
    nr = r + dl / 6.0 * (r1 + 2.0 * r2 + 2.0 * r3 + r4)
    nth = th + dl / 6.0 * (th1 + 2.0 * th2 + 2.0 * th3 + th4)
    nphi = phi + dl / 6.0 * (phi1 + 2.0 * phi2 + 2.0 * phi3 + phi4)
    npr = pr + dl / 6.0 * (pr1 + 2.0 * pr2 + 2.0 * pr3 + pr4)
    npth = pth + dl / 6.0 * (pth1 + 2.0 * pth2 + 2.0 * pth3 + pth4)
    return nr, nth, nphi, npr, npth


# ----------------------------------------------------------------------
# Blackbody temperature (Kelvin) -> linear sRGB, Tanner Helland approximation
# ----------------------------------------------------------------------
@ti.func
def blackbody_rgb(temp: ti.f32):
    t = ti.min(ti.max(temp, 1000.0), 40000.0) / 100.0
    r = 0.0
    g = 0.0
    b = 0.0
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
    # punch up saturation a bit for a more dramatic cinematic palette
    lum = rgb.dot(ti.Vector([0.299, 0.587, 0.114]))
    rgb = ti.max(lum + (rgb - lum) * 1.6, 0.0)
    return rgb


# ----------------------------------------------------------------------
# Turbulent scale-height modulation: makes the disk's vertical puffiness
# billow and swirl with the gas's own local orbital rotation, instead of
# being a rigid, perfectly smooth torus. Same rotating-phase trick used for
# the brightness turbulence below, so the puffs visibly co-rotate with the
# gas flow (inner puffs orbit faster than outer ones, just like the gas).
# ----------------------------------------------------------------------
@ti.func
def puff_factor(r, phase):
    n = ti.sin(phase * 5.0) * 0.5 + ti.sin(phase * 11.0 + r * 0.7) * 0.3 + ti.sin(phase * 23.0 - r) * 0.2
    return ti.max(0.35, 1.0 + PUFF_AMP * n)


# ----------------------------------------------------------------------
# Celestial sphere: checkerboard by (theta, phi)
# ----------------------------------------------------------------------
@ti.func
def sky_color(th, phi):
    u = phi / (2.0 * math.pi)
    v = th / math.pi
    cu = ti.floor(u * 24.0)
    cv = ti.floor(v * 12.0)
    checker = (cu + cv) % 2.0
    base = 0.05 + 0.10 * checker
    # subtle blue-white tint + a couple of bright "stars" bands for depth cues
    col = ti.Vector([0.55, 0.62, 0.75]) * base
    band = ti.exp(-ti.pow((v - 0.5) * 6.0, 2.0)) * 0.02
    col += ti.Vector([1.0, 0.9, 0.8]) * band
    return col


# ----------------------------------------------------------------------
# Main render kernel: one photon per pixel
# ----------------------------------------------------------------------
@ti.kernel
def render(a: ti.f32, r_cam: ti.f32, th_cam: ti.f32, phi_cam: ti.f32,
           t_mult: ti.f32, sim_t: ti.f32):
    r_h = horizon_radius(a)
    r_isco = isco_radius(a)

    for i, j in pixels:
        sx = (2.0 * (i + 0.5) / WIDTH - 1.0) * TAN_HALF_FOV
        sy = (2.0 * (j + 0.5) / HEIGHT - 1.0) * TAN_HALF_FOV

        # local ray direction in (r_hat, theta_hat, phi_hat) ZAMO axes
        dvec = ti.Vector([-1.0, sy, sx])
        dvec = dvec / dvec.norm()
        n_r, n_th, n_ph = dvec[0], dvec[1], dvec[2]

        sth = ti.sin(th_cam)
        cth = ti.cos(th_cam)
        sigma = r_cam * r_cam + a * a * cth * cth
        delta = r_cam * r_cam - 2.0 * r_cam + a * a
        Abig = (r_cam * r_cam + a * a) ** 2 - delta * a * a * sth * sth
        omega = 2.0 * a * r_cam / Abig

        e0 = ti.sqrt(delta * sigma / Abig)
        e1 = ti.sqrt(sigma / delta)
        e2 = ti.sqrt(sigma)
        e3 = ti.sqrt(Abig / sigma) * sth

        p1h = n_r
        p2h = n_th
        p3h = n_ph

        pr0 = p1h * e1
        pth0 = p2h * e2
        pphi0 = p3h * e3
        pt0 = -e0 - p3h * e3 * omega

        E = -pt0
        pr0 /= E
        pth0 /= E
        L = pphi0 / E

        Q = pth0 * pth0 + cth * cth * (L * L / (sth * sth) - a * a)

        r = r_cam
        th = th_cam
        phi = phi_cam
        pr = pr0
        pth = pth0

        # Volumetric front-to-back accumulation: instead of stopping the ray
        # the instant it crosses a mathematically flat disk plane, the disk
        # is now a genuine 3D gas volume with a turbulent, billowing vertical
        # density profile. Each step the ray spends inside that volume adds a
        # little emitted light and eats a little transmittance, exactly like
        # marching through fog -- so the disk reads as a puffy, glowing torus
        # you can partly see through, not a razor-thin opaque sheet.
        color = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        escaped_to_sky = 0

        for _s in range(MAX_STEPS):
            dl = BASE_DL * ti.min(ti.max(r * 0.15, 0.06), 15.0)
            nr, nth, nphi, npr, npth = rk4_step(r, th, phi, pr, pth, a, L, Q, dl)

            # polar reflection to keep theta in [0, pi]
            if nth < 0.0:
                nth = -nth
                npth = -npth
                nphi += math.pi
            if nth > math.pi:
                nth = 2.0 * math.pi - nth
                npth = -npth
                nphi += math.pi

            if nr >= r_h and nr <= DISK_R_OUT:
                height = nr * ti.cos(nth)  # local height above the equatorial plane
                density = 0.0
                emission = ti.Vector([0.0, 0.0, 0.0])

                if nr >= r_isco:
                    # --- main Novikov-Thorne-like emitting disk ---
                    omega_g = kerr_omega(nr, a)
                    ut = kerr_ut(nr, a)
                    g = ti.max(1.0 / (ut * (1.0 - L * omega_g)), 1e-3)

                    flux_shape = ti.max(1.0 - ti.sqrt(r_isco / nr), 1e-6)
                    temp_prof = T0 * ti.pow(nr, -0.75) * ti.pow(flux_shape, 0.25)
                    flux_prof = ti.pow(nr, -3.0) * flux_shape  # = (temp_prof/T0)^4

                    phase = nphi - omega_g * sim_t
                    turb = 0.75 + 0.15 * ti.sin(phase * 6.0) + 0.10 * ti.sin(phase * 17.0 + nr)
                    turb = ti.max(turb, 0.15)

                    scale_h = DISK_H0 * nr * puff_factor(nr, phase)
                    density = ti.exp(-0.5 * (height / scale_h) ** 2)

                    t_obs = temp_prof * t_mult * g
                    emission = blackbody_rgb(t_obs) * (BRIGHT_SCALE * flux_prof * ti.pow(g, 4.0) * turb)
                else:
                    # --- plunging-matter afterglow between the ISCO and the
                    # horizon: real infalling gas doesn't switch off the
                    # instant it crosses the ISCO, it fades fast as it's
                    # swallowed. Reuses the same circular-orbit redshift
                    # formula (extended past its strict validity) purely as
                    # a smooth, physically-flavored dimming/reddening curve.
                    omega_g = kerr_omega(nr, a)
                    ut = kerr_ut(nr, a)
                    g = ti.max(1.0 / (ut * (1.0 - L * omega_g)), 1e-3)

                    plunge_frac = ti.max((nr - r_h) / (r_isco - r_h), 0.0)
                    scale_h = DISK_H0 * nr * 0.6
                    density = ti.exp(-0.5 * (height / scale_h) ** 2)

                    t_obs = PLUNGE_TEMP * t_mult * g
                    emission = blackbody_rgb(t_obs) * (PLUNGE_GLOW_SCALE * plunge_frac * plunge_frac * ti.pow(g, 4.0))

                if density > 1e-4:
                    alpha = 1.0 - ti.exp(-DISK_OPACITY * density * dl)
                    color += transmittance * alpha * emission
                    transmittance *= (1.0 - alpha)

            r = nr
            th = nth
            phi = nphi
            pr = npr
            pth = npth

            if r <= r_h + 0.08:
                transmittance = 0.0
                break
            if r >= ESCAPE_R:
                escaped_to_sky = 1
                break
            if transmittance < TRANSMIT_CUTOFF:
                break

        if escaped_to_sky or transmittance >= TRANSMIT_CUTOFF:
            color += transmittance * sky_color(th, phi % (2.0 * math.pi))

        # luminance-based tonemap (preserves hue/saturation of bright pixels,
        # unlike a per-channel Reinhard curve which washes bright colors to white)
        lum = ti.max(color[0], ti.max(color[1], color[2]))
        color = color / (1.0 + lum)
        color = ti.pow(ti.max(color, 0.0), 1.0 / 2.2)
        pixels[i, j] = color


# ----------------------------------------------------------------------
# Interaction / main loop
# ----------------------------------------------------------------------
def main():
    gui = ti.GUI("Kerr Black Hole", res=(WIDTH, HEIGHT), show_gui=False, fast_gui=True)

    dragging = False
    last_pos = (0.0, 0.0)

    min_incl, max_incl = 0.08, math.pi - 0.08
    last_t = time.perf_counter()

    while gui.running:
        now = time.perf_counter()
        dt = min(now - last_t, 0.1)
        last_t = now

        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB and e.type == ti.GUI.PRESS:
                dragging = True
                last_pos = gui.get_cursor_pos()
            elif e.key == ti.GUI.LMB and e.type == ti.GUI.RELEASE:
                dragging = False
            elif e.key == ti.GUI.WHEEL:
                dy = e.delta[1]
                cam_dist[None] = float(
                    np.clip(cam_dist[None] - dy * 0.01 * cam_dist[None], 3.5, 90.0)
                )

        if dragging:
            cur = gui.get_cursor_pos()
            dx = cur[0] - last_pos[0]
            dy = cur[1] - last_pos[1]
            cam_azimuth[None] = float(cam_azimuth[None] - dx * 3.5)
            cam_incl[None] = float(
                np.clip(cam_incl[None] - dy * 3.0, min_incl, max_incl)
            )
            last_pos = cur

        if gui.is_pressed('a'):
            spin[None] = float(np.clip(spin[None] - 0.4 * dt, 0.0, 0.998))
        if gui.is_pressed('d'):
            spin[None] = float(np.clip(spin[None] + 0.4 * dt, 0.0, 0.998))
        if gui.is_pressed('w'):
            temp_mult[None] = float(np.clip(temp_mult[None] + 0.6 * dt, 0.2, 3.0))
        if gui.is_pressed('s'):
            temp_mult[None] = float(np.clip(temp_mult[None] - 0.6 * dt, 0.2, 3.0))

        sim_time[None] += dt

        render(
            spin[None], cam_dist[None], cam_incl[None], cam_azimuth[None],
            temp_mult[None], sim_time[None],
        )
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
