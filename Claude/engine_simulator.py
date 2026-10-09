#!/usr/bin/env python3
"""
Procedural Car Engine Simulator  v2
===================================
Real-time engine physics + procedural (NumPy) audio synthesis + live cylinder
visualisation + dashboard HUD, rendered with pygame.

SETUP
-----
    pip install pygame numpy
    python engine_simulator.py            # optional:  --fuel 8   (start with 8 % fuel)

CONTROLS (every key also has an on-screen button)
-------------------------------------------------
    W / Up        Throttle (hold)            S / Down    Brake (hold)
    Shift / C     Clutch pedal (hold)        Space       Start / Stop engine
    1-7           Engine: I4 V6 V8 V10 V12 W16 Flat-6
    M             Toggle MANUAL / AUTO gearbox
    E / Q         Shift UP / DOWN  (manual: needs clutch in, else GEAR CRUNCH;
                  auto: temporary paddle-shift override)
    D / N         Drive (auto) / Neutral
    H             Rev-match assist (auto-blip on downshifts)      ON / OFF
    T             Induction: Naturally-aspirated -> Turbo -> Supercharger
    R             Refuel (engine off, car stopped)     K  Repair / service
    V variant     U units     X mute     Esc quit

THINGS TO TRY
-------------
* Manual: Space (clutch in or neutral), hold Shift, press E for 1st, add throttle and
  release Shift slowly.  Drop it too fast at idle and you stall.
* Turbo: build boost, lift off -> blow-off valve.  Lift off from high RPM -> backfire.
* Hold the throttle in Neutral near the limiter and watch the coolant temperature.
* Start with --fuel 5 to run out of fuel (sputter, then stall).

Sound model:  Hz = (RPM / 60) * (Cylinders / 2) firing pulses per second.  Each pulse is a
pre-synthesised burst (pitch-dropping sine + band-passed noise + tanh overdrive, per
cylinder / RPM band / load) overlap-added at the firing instants, plus phase-continuous
harmonics, rumble/intake beds, turbo whistle, supercharger whine, starter, BOV, gear
crunch, backfire and failure one-shots.

Demo time-scaling (real physics, accelerated clocks): fuel burn x5, thermal x6.
"""

import argparse
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

TAU = 2.0 * math.pi
CHUNK = 768                 # audio samples rendered per block (~17 ms)
FUEL_TIME_SCALE = 5.0       # accelerated fuel burn (demo)
THERMAL_SCALE = 2.5         # accelerated thermal dynamics (demo)
AMBIENT = 25.0
FUEL_DENSITY = 0.745        # kg / L
STOICH = 14.7


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp(t, 0.0, 1.0)
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def smoothstep(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------- #
# Cylinder & EngineConfig
# --------------------------------------------------------------------------- #
class Cylinder:
    """One cylinder: layout position + firing glow state."""
    FADE_RATE = 11.0

    def __init__(self, number: int, bank: int, col: int):
        self.number, self.bank, self.col = number, bank, col
        self.pos = (0.0, 0.0)
        self.radius = 24
        self.glow = 0.0

    def fire(self):
        self.glow = 1.0

    def update(self, dt: float):
        self.glow *= math.exp(-self.FADE_RATE * dt)
        if self.glow < 0.01:
            self.glow = 0.0

    @staticmethod
    def glow_color(g: float):
        cold, red, orange = (38, 44, 58), (205, 38, 28), (255, 160, 35)
        if g < 0.5:
            return lerp_color(cold, red, g * 2.0)
        return lerp_color(red, orange, (g - 0.5) * 2.0)


@dataclass
class EngineConfig:
    key: str
    name: str
    variant: str
    description: str
    fire_order: List[int]
    banks: List[List[int]]
    bank_angles: List[float]
    peak_torque: float
    peak_rpm: float
    redline: float
    idle: float
    top_speed: float
    displacement: float = 4.0         # litres
    tank: float = 60.0                # litres
    mass: float = 1500.0
    cda: float = 0.64
    traction: float = 0.9
    body_freq: float = 100.0
    body_decay: float = 0.025
    noise_band: Tuple[float, float] = (300.0, 3000.0)
    noise_gain: float = 0.5
    drive: float = 1.5
    hum_gain: float = 0.2
    hum_weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.15)
    sub_gain: float = 0.3
    interval_pattern: Tuple[float, ...] = (1.0,)
    bank_amps: Tuple[float, ...] = (1.0,)

    def __post_init__(self):
        n = len(self.fire_order)
        assert sorted(self.fire_order) == list(range(1, n + 1)), self.name
        assert sorted(c for b in self.banks for c in b) == list(range(1, n + 1))
        self.cylinders = n
        self.fric = 0.06 * self.peak_torque
        self.inertia = self.peak_torque / 550.0
        mean = sum(self.interval_pattern) / len(self.interval_pattern)
        self.interval_pattern = tuple(p / mean for p in self.interval_pattern)
        self.bank_index = [0] * n
        for bi, bank in enumerate(self.banks):
            for num in bank:
                self.bank_index[num - 1] = bi

    @property
    def label(self):
        return f"{self.name} {self.variant}".strip()

    def pulse_hz(self, rpm: float) -> float:
        """Hz = (RPM / 60) * (Cylinders / 2)"""
        return rpm / 60.0 * self.cylinders / 2.0

    def make_cylinders(self) -> List[Cylinder]:
        cyls = [None] * self.cylinders
        for bi, bank in enumerate(self.banks):
            for ci, num in enumerate(bank):
                cyls[num - 1] = Cylinder(num, bi, ci)
        return cyls


def build_catalog() -> List[List[EngineConfig]]:
    """One list of variants per engine (index = key 1..7)."""
    E = EngineConfig
    i4 = [E("I4", "Inline-4", "", "Buzzy, high-revving four. Firing 1-3-4-2.",
            [1, 3, 4, 2], [[1, 2, 3, 4]], [0],
            250, 4800, 7500, 850, 215, displacement=2.0, tank=45, mass=1250,
            body_freq=150, body_decay=0.016, noise_band=(700, 4000), noise_gain=0.55,
            drive=1.3, hum_gain=0.20, hum_weights=(1, .6, .35, .2), sub_gain=0.12)]

    v6c = dict(peak_torque=400, peak_rpm=5500, redline=7500, idle=800, top_speed=260,
               displacement=3.5, tank=55, mass=1450, body_freq=125, body_decay=0.019,
               noise_band=(500, 3500), noise_gain=0.5, drive=1.4, hum_gain=0.22,
               hum_weights=(1, .55, .3, .18), sub_gain=0.18)
    v6 = [E("V6", "V6", "60\u00b0", "Smooth, even-fire 60\u00b0 V6. Firing 1-2-5-6-3-4.",
            [1, 2, 5, 6, 3, 4], [[1, 3, 5], [2, 4, 6]], [-30, 30], **v6c),
          E("V6", "V6", "90\u00b0", "Odd-fire 90\u00b0 V6, uneven pulses. Firing 1-2-5-6-3-4.",
            [1, 2, 5, 6, 3, 4], [[1, 3, 5], [2, 4, 6]], [-45, 45],
            interval_pattern=(1.25, 0.75), bank_amps=(1.0, 0.85), **v6c)]

    v8c = dict(peak_torque=600, peak_rpm=5000, redline=7200, idle=750, top_speed=300,
               displacement=6.2, tank=70, mass=1600)
    v8 = [E("V8", "V8", "Cross-plane", "Deep burble, uneven exhaust pulses. Firing 1-8-4-3-6-5-7-2.",
            [1, 8, 4, 3, 6, 5, 7, 2], [[1, 3, 5, 7], [2, 4, 6, 8]], [-45, 45],
            body_freq=72, body_decay=0.036, noise_band=(180, 1800), noise_gain=0.55,
            drive=1.9, hum_gain=0.20, hum_weights=(1, .7, .35, .12), sub_gain=0.55,
            bank_amps=(1.0, 0.82), **v8c),
          E("V8", "V8", "Flat-plane", "Higher-pitched, raspy exotic V8. Firing 1-8-4-3-6-5-7-2.",
            [1, 8, 4, 3, 6, 5, 7, 2], [[1, 3, 5, 7], [2, 4, 6, 8]], [-45, 45],
            body_freq=112, body_decay=0.021, noise_band=(500, 5000), noise_gain=0.6,
            drive=1.5, hum_gain=0.22, hum_weights=(1, .5, .45, .3), sub_gain=0.25, **v8c)]

    v10 = [E("V10", "V10", "", "Screaming 90\u00b0 V10, uneven firing. Firing 1-6-5-10-2-7-3-8-4-9.",
             [1, 6, 5, 10, 2, 7, 3, 8, 4, 9], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [-45, 45],
             520, 7000, 8500, 900, 320, displacement=5.2, tank=80, mass=1550,
             body_freq=120, body_decay=0.018, noise_band=(900, 7000), noise_gain=0.75,
             drive=1.5, hum_gain=0.30, hum_weights=(1, .7, .55, .4), sub_gain=0.2,
             interval_pattern=(1.18, 0.82))]

    v12 = [E("V12", "V12", "", "Silky, howling 60\u00b0 V12. Firing 1-12-4-9-2-11-6-7-3-10-5-8.",
             [1, 12, 4, 9, 2, 11, 6, 7, 3, 10, 5, 8],
             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [-30, 30],
             700, 6500, 8500, 900, 340, displacement=6.0, tank=90, mass=1750,
             body_freq=100, body_decay=0.020, noise_band=(700, 5000), noise_gain=0.35,
             drive=1.2, hum_gain=0.55, hum_weights=(1, .8, .6, .45), sub_gain=0.22)]

    w16 = [E("W16", "W16", "", "Quad-bank 8.0 L monster. Firing 1-14-9-4-7-12-15-6-13-8-3-10-11-2-5-16.",
             [1, 14, 9, 4, 7, 12, 15, 6, 13, 8, 3, 10, 11, 2, 5, 16],
             [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
             [-40, -13, 13, 40],
             1500, 4500, 7000, 750, 400, displacement=8.0, tank=100, mass=1950, cda=0.60,
             traction=1.1, body_freq=55, body_decay=0.048, noise_band=(120, 1400),
             noise_gain=0.6, drive=2.1, hum_gain=0.25, hum_weights=(1, .65, .4, .2),
             sub_gain=0.85, interval_pattern=(1.07, 0.93), bank_amps=(1.0, 0.8, 0.92, 0.72))]

    f6 = [E("Flat-6", "Flat-6", "", "Boxer six, metallic rasp. Firing 1-6-2-4-3-5.",
            [1, 6, 2, 4, 3, 5], [[1, 2, 3], [4, 5, 6]], [-90, 90],
            450, 6200, 8000, 900, 290, displacement=3.8, tank=64, mass=1400,
            body_freq=130, body_decay=0.020, noise_band=(450, 3500), noise_gain=0.55,
            drive=1.6, hum_gain=0.22, hum_weights=(1, .6, .4, .25), sub_gain=0.22,
            bank_amps=(1.0, 0.88))]
    return [i4, v6, v8, v10, v12, w16, f6]


# --------------------------------------------------------------------------- #
# Procedural audio
# --------------------------------------------------------------------------- #
class AudioSynthesizer:
    """
    Streams procedural audio through a pygame Channel from a worker thread.

    Shared state (written by main thread): rpm, throttle, running, cranking,
    fuel_cut (= no combustion), induction (0 NA / 1 turbo / 2 supercharger),
    spool (0..1), boost (bar).  One-shot sounds are requested via trigger().
    """
    BAND_MULT = (0.75, 1.1, 1.6)

    def __init__(self):
        self.enabled = False
        self.muted = False
        self.master = 0.8
        self.rpm = self.throttle = 0.0
        self.running = self.cranking = self.fuel_cut = False
        self.induction = 0
        self.spool = self.boost = 0.0
        self.cfg = None
        self.scope = np.zeros(CHUNK, dtype=np.float32)
        self.rng = np.random.default_rng(7)
        self.events = deque()
        self._lock = threading.Lock()
        self._alive = False
        self._thread = None
        self.sr, self.channels = 44100, 1
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 1, 512, allowedchanges=0)
            freq, _fmt, ch = pygame.mixer.get_init()
            self.sr, self.channels = int(freq), int(ch)
            self.enabled = True
        except pygame.error as exc:
            print(f"[audio] disabled: {exc}")
        self.shots = {}
        if self.enabled:
            self._build_beds()
        self._reset_state()

    # ----- DSP helpers ------------------------------------------------------- #
    def _bandpass(self, x, lo, hi):
        n = len(x)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        mask = (1.0 / (1.0 + (f / max(hi, 1.0)) ** 4)) * \
               (1.0 / (1.0 + (lo / np.maximum(f, 1.0)) ** 4))
        y = np.fft.irfft(X * mask, n)
        m = np.max(np.abs(y))
        return (y / m if m > 0 else y).astype(np.float32)

    def _noise(self, n, lo, hi, rng):
        return self._bandpass(rng.standard_normal(n), lo, hi)

    def _build_beds(self):
        sr = self.sr
        rng = np.random.default_rng(99)
        self.rumble = self._noise(3 * sr, 1.0, 160.0, rng)
        self.air = self._noise(3 * sr, 250.0, 2600.0, rng)
        lp = int(0.05 * sr)
        t = np.arange(lp) / sr
        self.pop = (self._noise(lp, 300.0, 3500.0, rng) * np.exp(-t / 0.012) *
                    (1 - np.exp(-t / 0.0008))).astype(np.float32)

        def T(d):
            return np.arange(int(d * sr)) / sr

        # gear crunch: grinding teeth + metallic ring
        t = T(0.38)
        grind = self._noise(len(t), 700, 6500, rng) * (0.3 + 0.7 * (np.sin(TAU * 46 * t) > 0.1))
        ring = 0.3 * np.sin(TAU * 2150 * t) + 0.2 * np.sin(TAU * 3270 * t)
        env = (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.17)
        s = (grind + ring * np.exp(-t / 0.1)) * env
        self.shots["crunch"] = (s / np.max(np.abs(s)) * 0.9).astype(np.float32)

        # blow-off valve: breathy psssh with flutter
        t = T(1.0)
        n = self._noise(len(t), 2200, 9500, rng)
        flutter = 0.65 + 0.35 * np.sin(TAU * (26 - 10 * t) * t)
        env = (1 - np.exp(-t / 0.01)) * np.exp(-t / 0.30)
        low = self._noise(len(t), 150, 900, rng) * np.exp(-t / 0.12) * 0.4
        s = n * flutter * env + low
        self.shots["bov"] = (s / np.max(np.abs(s)) * 0.8).astype(np.float32)

        # backfire bangs (3 variants): thump + exhaust noise + cracks
        self.shots["backfire"] = []
        for k in range(3):
            t = T(0.30)
            thump = np.sin(TAU * (55 + 12 * k) * t * (1 + 0.5 * np.exp(-t / 0.03))) * np.exp(-t / 0.05)
            n = self._noise(len(t), 100, 1800, rng) * np.exp(-t / (0.03 + 0.01 * k))
            cr = np.zeros(len(t))
            for _ in range(rng.integers(3, 7)):
                p = int(rng.uniform(0.0, 0.2) * sr)
                seg = self._noise(240, 800, 6000, rng) * np.exp(-np.arange(240) / 50.0)
                cr[p:p + 240] += seg[:max(0, len(cr) - p)][:240] * rng.uniform(0.3, 0.8)
            s = np.tanh(2.2 * (n + 0.8 * thump + cr))
            self.shots["backfire"].append(s.astype(np.float32))

        # stall clunk
        t = T(0.35)
        s = np.sin(TAU * 48 * t) * np.exp(-t / 0.09) + 0.5 * self._noise(len(t), 40, 400, rng) * np.exp(-t / 0.05)
        self.shots["stall"] = (s / np.max(np.abs(s)) * 0.8).astype(np.float32)

        # engine failure: bang, rod rattle, ring, hiss
        t = T(2.0)
        bang = self._noise(len(t), 80, 3000, rng) * np.exp(-t / 0.12) * 1.5
        rattle = self._noise(len(t), 300, 5000, rng) * (np.sin(TAU * 22 * t) > 0) * np.exp(-t / 0.6) * 0.6
        ring = 0.3 * np.sin(TAU * 1400 * t) * np.exp(-t / 0.5)
        hiss = self._noise(len(t), 3000, 9000, rng) * np.exp(-t / 1.0) * 0.2
        s = np.tanh(1.5 * (bang + rattle + ring + hiss))
        self.shots["blow"] = s.astype(np.float32)

    def _reset_state(self):
        self.L = int(0.06 * self.sr)
        self.tail = np.zeros(self.L, dtype=np.float32)
        self.next_pulse = 0.0
        self.slot = 0
        self.hum_phase = 0.0
        self.prev_hz = 0.0
        self.gains = {}
        self.phases = {}
        self.freqs = {}
        self.active = []
        self.idx = {"rumble": 0, "air": 0, "gur": 40000, "breath": 20000}
        self.T = None

    # ----- per-engine pulse templates --------------------------------------- #
    def set_engine(self, cfg: EngineConfig):
        with self._lock:
            self.cfg = cfg
            self._reset_state()
            if self.enabled:
                self._build_templates()

    def _build_templates(self):
        cfg, sr, L, n = self.cfg, self.sr, self.L, self.cfg.cylinders
        t = np.arange(L) / sr
        atk = 1.0 - np.exp(-t / 0.0012)
        rng = np.random.default_rng(1234)
        T = np.zeros((3, 2, n, L), dtype=np.float32)
        for b, mult in enumerate(self.BAND_MULT):
            for c in range(n):
                detune = 1.0 + 0.04 * math.sin(c * 2.39 + 1.3)
                f0 = cfg.body_freq * mult * detune
                inst = f0 * (1.0 + 0.7 * np.exp(-t / 0.012))
                ph = TAU * np.cumsum(inst) / sr
                env = np.exp(-t / (cfg.body_decay / mult ** 0.6)) * atk
                body = np.sin(ph) * env + 0.4 * np.sin(2 * ph + 0.5) * env ** 1.5
                lo, hi = cfg.noise_band
                nz = self._noise(L, lo * mult ** 0.5, min(hi * mult ** 0.5, sr * 0.42), rng)
                nz = nz * np.exp(-t / 0.009) * atk
                for li, load in enumerate((0.0, 1.0)):
                    pulse = (0.55 + 0.55 * load) * body + cfg.noise_gain * (1.0 - 0.35 * load) * nz
                    d = 1.0 + (cfg.drive - 1.0) * (0.25 + 0.75 * load)
                    T[b, li, c] = np.tanh(d * pulse) * (0.6 + 0.4 * load)
        self.T = T.reshape(6, n, L)

    def trigger(self, name, amp=1.0):
        self.events.append((name, float(amp)))

    # ----- block rendering --------------------------------------------------- #
    def _ramp(self, name, target, n):
        prev = self.gains.get(name, 0.0)
        self.gains[name] = target
        return np.linspace(prev, target, n, dtype=np.float32)

    def _loop(self, name, loop, n):
        i, size = self.idx[name], len(loop)
        if i + n <= size:
            out = loop[i:i + n]
        else:
            out = np.concatenate((loop[i:], loop[:i + n - size]))
        self.idx[name] = (i + n) % size
        return out

    def _phase(self, name, f1, n):
        f0 = self.freqs.get(name, f1)
        fs = np.linspace(f0, f1, n)
        ph = self.phases.get(name, 0.0) + np.cumsum(fs * (TAU / self.sr))
        self.phases[name] = float(ph[-1] % TAU)
        self.freqs[name] = f1
        return ph

    def _render(self) -> np.ndarray:
        N, sr = CHUNK, self.sr
        cfg = self.cfg
        if cfg is None or self.T is None:
            return np.zeros(N, dtype=np.float32)
        rpm = max(0.0, float(self.rpm))
        thr = clamp(float(self.throttle), 0.0, 1.0)
        running, cut, crank = bool(self.running), bool(self.fuel_cut), bool(self.cranking)
        n, L = cfg.cylinders, self.L
        x = clamp(rpm / cfg.redline, 0.0, 1.1)
        pulse_hz = cfg.pulse_hz(rpm)

        ov = 0.0
        if running and not cut:
            ov = (1.0 - thr) * clamp((rpm - cfg.idle * 1.5) / (cfg.redline * 0.45), 0.0, 1.0)
        if running:
            level = (0.34 + 0.66 * thr) * (1.0 - 0.45 * ov) * (0.75 + 0.25 * x)
            if cut:
                level = 0.12
        else:
            level = 0.16 * clamp(rpm / 250.0, 0.0, 1.0)

        p = min(x, 1.0) * 2.0
        bi = min(int(p), 1)
        fr = p - bi
        wb = np.zeros(3, dtype=np.float32)
        wb[bi], wb[bi + 1] = 1.0 - fr, fr
        load = clamp((0.12 + 0.88 * thr) * (1.0 - 0.6 * ov), 0.0, 1.0)
        wl = np.array([1.0 - load, load], dtype=np.float32)
        tpl = np.tensordot(np.outer(wb, wl).ravel(), self.T, axes=1)

        buf = np.zeros(N + L, dtype=np.float32)
        buf[:L] += self.tail
        if pulse_hz > 4.0:
            base = max(sr / pulse_hz, 6.0)
            gnorm = (45.0 / max(45.0, pulse_hz)) ** 0.5
            pat, amps, order = cfg.interval_pattern, cfg.bank_amps, cfg.fire_order
            guard = 0
            while self.next_pulse < N and guard < 400:
                s = self.slot
                c = order[s] - 1
                o = int(self.next_pulse)
                a = gnorm * level * amps[cfg.bank_index[c] % len(amps)] * \
                    (0.92 + 0.16 * self.rng.random())
                buf[o:o + L] += tpl[c] * a
                self.next_pulse += base * pat[s % len(pat)]
                self.slot = (s + 1) % n
                guard += 1
            self.next_pulse -= N
        else:
            self.next_pulse = 0.0

        if running and ov > 0.3 and rpm > 2000 and self.rng.random() < 0.10 * ov:
            o = int(self.rng.integers(0, N))
            buf[o:o + len(self.pop)] += self.pop * float(self.rng.uniform(0.5, 1.0)) * 0.9

        out = buf[:N].copy()
        self.tail = buf[N:].copy()

        # phase-continuous harmonics locked to firing frequency
        fs = np.linspace(self.prev_hz, pulse_hz, N)
        ph = self.hum_phase + np.cumsum(fs * (TAU / sr))
        self.hum_phase = float(ph[-1] % TAU)
        self.prev_hz = pulse_hz
        hg = 0.0
        if running and not cut:
            hg = cfg.hum_gain * (0.25 + 0.75 * thr) * (0.35 + 0.65 * x) * (1.0 - 0.55 * ov)
        hum = np.zeros(N)
        for k, wk in enumerate(cfg.hum_weights):
            if (k + 1) * pulse_hz < sr * 0.45:
                hum += wk * np.sin((k + 1) * ph)
        out += hum.astype(np.float32) * self._ramp("hum", hg, N)

        rg = cfg.sub_gain * (0.12 + 0.88 * thr) * (0.35 + 0.65 * x) if running else 0.0
        out += self._loop("rumble", self.rumble, N) * self._ramp("rumble", rg, N)
        ag = cfg.noise_gain * 0.30 * thr * x ** 1.3 if running else 0.0
        out += self._loop("air", self.air, N) * self._ramp("air", ag, N)

        gg = 0.8 * ov * (0.4 + cfg.sub_gain)
        mod = (0.55 + 0.45 * np.sin(ph * 0.5)).astype(np.float32)
        out += self._loop("gur", self.rumble, N) * mod * self._ramp("gur", gg, N)

        # ---- forced induction ------------------------------------------------ #
        spool = clamp(float(self.spool), 0.0, 1.0)
        if self.induction == 1:        # turbo: spool whistle + breathy flow
            wf = 1500.0 + 7000.0 * spool
            wp = self._phase("whistle", wf, N)
            wg = 0.085 * spool ** 1.2 * (0.35 + 0.65 * thr) if running else 0.0
            out += (np.sin(wp) + 0.3 * np.sin(2 * wp)).astype(np.float32) * self._ramp("whistle", wg, N)
            bg = 0.10 * spool * (0.4 + 0.6 * thr) if running else 0.0
            out += self._loop("breath", self.air, N) * self._ramp("breath", bg, N)
        elif self.induction == 2:      # supercharger: gear-driven mechanical whine
            wf = max(40.0, rpm / 60.0 * 11.0)
            wp = self._phase("whine", wf, N)
            sg = (0.025 + 0.11 * clamp(float(self.boost) / 0.7, 0.0, 1.0)) if running else 0.0
            tone = np.sin(wp) + 0.5 * np.sin(2 * wp) + 0.25 * np.sin(3 * wp + 0.7)
            out += tone.astype(np.float32) * self._ramp("whine", sg, N)

        # ---- starter motor ------------------------------------------------------ #
        sp = self._phase("starter", 150.0 + 60.0 * math.sin(time.time() * 3.0), N)
        stg = 0.09 if crank else 0.0
        am = (0.7 + 0.3 * np.sin(sp * 0.06)).astype(np.float32)
        out += np.sin(sp).astype(np.float32) * am * self._ramp("starter", stg, N)

        # ---- one-shots: crunch, BOV, backfire, stall, failure -------------------- #
        while self.events:
            name, amp = self.events.popleft()
            arr = self.shots.get(name)
            if isinstance(arr, list):
                arr = arr[int(self.rng.integers(0, len(arr)))]
            if arr is not None:
                self.active.append([arr, 0, amp])
        for sh in list(self.active):
            arr, pos, g = sh
            seg = arr[pos:pos + N]
            out[:len(seg)] += seg * g
            sh[1] += N
            if sh[1] >= len(arr):
                self.active.remove(sh)

        mix = np.tanh(out * 1.6) * 0.9
        self.scope = mix
        return mix

    def _make_sound(self):
        with self._lock:
            mix = self._render()
        vol = 0.0 if self.muted else self.master
        pcm = (mix * 32767.0 * vol).astype(np.int16)
        if self.channels == 2:
            pcm = np.column_stack((pcm, pcm))
        return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))

    def _run(self):
        ch = pygame.mixer.Channel(0)
        while self._alive:
            try:
                if not ch.get_busy():
                    ch.play(self._make_sound())
                elif ch.get_queue() is None:
                    ch.queue(self._make_sound())
                else:
                    time.sleep(0.003)
            except pygame.error:
                break

    def start(self):
        if self.enabled and not self._alive:
            self._alive = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._alive = False
        if self._thread:
            self._thread.join(timeout=0.5)


# --------------------------------------------------------------------------- #
# Transmission (automatic + manual)
# --------------------------------------------------------------------------- #
class Transmission:
    """
    AUTO  : P / N / D with RPM/throttle shift logic, kick-down, rev-matched
            downshift blips and temporary paddle override.
    MANUAL: gear 0 (neutral) .. 7 chosen by the driver; the clutch is a pedal.
    """
    RATIOS = (3.62, 2.38, 1.72, 1.34, 1.08, 0.88, 0.73)
    SHIFT_TIME = 0.30

    def __init__(self):
        self.manual = False
        self.mode = "P"
        self.gear = 1
        self.target = 1
        self.shift_timer = 0.0
        self.shift_dir = 0
        self.cooldown = 0.0
        self.paddle_hold = 0.0
        self.flash = 0.0
        self.final = 3.3

    def configure(self, cfg: EngineConfig, wheel_r: float):
        w = cfg.redline * 0.97 * TAU / 60.0
        self.final = (w * wheel_r) / ((cfg.top_speed / 3.6) * self.RATIOS[-1])

    def reset(self, mode="P"):
        self.shift_timer, self.shift_dir, self.cooldown = 0.0, 0, 0.0
        self.paddle_hold = 0.0
        if self.manual:
            self.mode, self.gear, self.target = "N", 0, 0
        else:
            self.mode, self.gear, self.target = mode, 1, 1

    @property
    def shifting(self):
        return self.shift_timer > 0.0

    def wheel_rpm(self, speed, wheel_r, gear=None):
        g = self.gear if gear is None else gear
        return speed / wheel_r * self.final * self.RATIOS[max(1, g) - 1] * 60.0 / TAU

    def total_ratio(self):
        if self.manual:
            return 0.0 if self.gear == 0 else self.RATIOS[self.gear - 1] * self.final
        if self.mode != "D" or self.shifting:
            return 0.0
        return self.RATIOS[self.gear - 1] * self.final

    def engage_drive(self, speed, cfg, wheel_r):
        self.mode = "D"
        g = 1
        for k in range(7, 0, -1):
            if self.wheel_rpm(speed, wheel_r, k) >= 0.32 * cfg.redline:
                g = k
                break
        self.gear = self.target = g
        self.shift_timer, self.shift_dir = 0.0, 0

    def display(self):
        if self.manual:
            return "N" if self.gear == 0 else str(self.gear)
        if self.mode in ("P", "N"):
            return self.mode
        return str(self.target if self.shifting else self.gear)

    def update(self, dt, eng_rpm, speed, throttle, cfg, wheel_r):
        self.flash = max(0.0, self.flash - dt)
        if self.manual:
            self.shift_timer = 0.0
            return
        self.cooldown = max(0.0, self.cooldown - dt)
        self.paddle_hold = max(0.0, self.paddle_hold - dt)
        if self.shift_timer > 0.0:
            self.shift_timer -= dt
            if self.shift_timer <= 0.0:
                self.gear, self.shift_dir = self.target, 0
            return
        if self.mode != "D":
            self.gear = self.target = 1
            return
        if self.cooldown > 0.0 or self.paddle_hold > 0.0:
            return
        wrpm = self.wheel_rpm(speed, wheel_r)
        up = (0.36 + 0.58 * throttle) * cfg.redline
        down = (0.19 + 0.30 * throttle) * cfg.redline
        if self.gear < 7 and eng_rpm > up and wrpm > 0.6 * up:
            self.start_shift(+1)
        elif self.gear > 1 and wrpm < down:
            self.start_shift(-1)

    def start_shift(self, d):
        self.target = self.gear + d
        self.shift_dir = d
        self.shift_timer = self.SHIFT_TIME
        self.cooldown = 1.0


# --------------------------------------------------------------------------- #
# Vehicle: engine + clutch + chassis + fuel + thermals + induction
# --------------------------------------------------------------------------- #
class Vehicle:
    WHEEL_R = 0.33
    K_CLUTCH = 80.0
    IND_NAMES = ("NA", "TURBO", "SUPERCHARGER")
    IND_MAX_BOOST = (0.0, 1.1, 0.7)       # bar

    def __init__(self, tx: Transmission, fuel_pct=100.0):
        self.tx = tx
        self.cfg = None
        self.start_fuel_pct = fuel_pct
        self.omega = 0.0
        self.speed = 0.0
        self.throttle_cmd = self.throttle = self.thr_eff = 0.0
        self.brake_cmd = self.brake = 0.0
        self.clutch_cmd = self.clutch = 0.0          # pedal: 1 = pressed (disengaged)
        self.running = self.cranking = False
        self.fuel_cut = False                        # rev limiter
        self.no_fire = False                         # limiter or fuel starvation
        self.starve_t = 0.0
        self.crank_t = 0.0
        self.run_t = 99.0
        self.stall_t = 0.0
        self.rev_match = True
        self.rm_t = 0.0
        self.rm_target = 0.0
        self.induction = 0
        self.spool = self.boost = self.map_gauge = 0.0
        self.torque_mult = 1.0
        self.derate = 1.0
        self.fuel_l = 0.0
        self.refueling = False
        self.coolant = 60.0
        self.fan_on = False
        self.oil = 0.0
        self.damage = 0.0
        self.blown = False
        self.blown_reason = ""
        self.afr, self.lam, self.flow_lph, self.mdot_air = 99.9, 1.0, 0.0, 0.0
        self.eng_disp = 0.0
        self.prev_thr = 0.0
        self.bov_cd = 0.0
        self.burst, self.burst_t = 0, 0.0
        self.backfire_t = 0.0
        self.stalls = self.crunches = self.backfires = 0
        self.events = deque()

    # ----- convenience -------------------------------------------------------- #
    @property
    def rpm(self):
        return self.omega * 60.0 / TAU

    @property
    def fuel_pct(self):
        return 100.0 * self.fuel_l / self.cfg.tank

    @property
    def ind_name(self):
        return self.IND_NAMES[self.induction]

    def msg(self, text, color="warn"):
        self.events.append(("msg", (text, color)))

    def drain(self):
        while self.events:
            yield self.events.popleft()

    # ----- setup / commands ---------------------------------------------------- #
    def set_engine(self, cfg: EngineConfig):
        keep = self.running
        self.cfg = cfg
        self.speed = 0.0
        self.tx.configure(cfg, self.WHEEL_R)
        self.tx.reset("N" if keep else "P")
        self.omega = cfg.idle * TAU / 60.0 if keep else 0.0
        self.cranking = self.fuel_cut = self.no_fire = False
        self.run_t = 99.0
        self.fuel_l = cfg.tank * self.start_fuel_pct / 100.0
        self.start_fuel_pct = 100.0
        self.coolant = 82.0 if keep else 60.0
        self.damage, self.blown, self.refueling = 0.0, False, False
        self.spool = self.boost = 0.0
        self.clutch = self.clutch_cmd = 0.0

    def toggle(self):
        if self.running or self.cranking:
            self.running = self.cranking = False
            return
        if self.blown:
            return self.msg("ENGINE FAILED (" + self.blown_reason + ") - press K to repair", "bad")
        if self.refueling:
            return self.msg("Refuelling in progress...", "warn")
        if self.fuel_l <= 0.01:
            return self.msg("OUT OF FUEL - press R to refuel", "bad")
        if self.tx.manual and self.tx.gear != 0 and self.clutch < 0.8:
            return self.msg("Press clutch (Shift/C) or select neutral to start", "warn")
        self.cranking, self.crank_t = True, 0.0

    def request_drive(self):
        if self.tx.manual:
            return self.msg("Manual mode: use E / Q to change gear", "info")
        if self.running:
            self.tx.engage_drive(self.speed, self.cfg, self.WHEEL_R)

    def request_neutral(self):
        if self.tx.manual:
            self.tx.gear = 0
        elif self.tx.mode != "P":
            self.tx.mode = "N"

    def toggle_manual(self):
        tx = self.tx
        if tx.manual:
            tx.manual = False
            self.clutch_cmd = 0.0
            if tx.gear > 0 and self.running:
                tx.engage_drive(self.speed, self.cfg, self.WHEEL_R)
            else:
                tx.mode = "N"
            self.msg("AUTOMATIC transmission", "info")
        else:
            was_drive = tx.mode == "D"
            if tx.shifting:
                tx.gear, tx.shift_timer = tx.target, 0.0
            tx.manual = True
            tx.gear = tx.gear if was_drive else 0
            tx.mode = "N"
            self.msg("MANUAL transmission - clutch: Shift / C", "info")

    def toggle_induction(self):
        self.induction = (self.induction + 1) % 3
        self.spool = self.boost = 0.0
        self.msg("Induction: " + ("Naturally aspirated", "Turbocharger", "Supercharger")[self.induction], "info")

    def start_refuel(self):
        if self.running or self.cranking:
            return self.msg("Turn the engine off to refuel", "warn")
        if self.speed > 0.5:
            return self.msg("Stop the car to refuel", "warn")
        if self.fuel_l >= self.cfg.tank - 0.05:
            return self.msg("Tank already full", "info")
        self.refueling = True
        self.msg("REFUELLING...", "info")

    def repair(self):
        if self.running:
            return self.msg("Turn the engine off to repair", "warn")
        self.blown, self.damage = False, 0.0
        self.coolant = 60.0
        self.msg("Engine serviced - ready to start", "good")

    def paddle(self, d):
        tx = self.tx
        if tx.mode != "D" or tx.shifting:
            return self.msg("Paddle shift needs Drive (D)", "info")
        new = tx.gear + d
        if new < 1 or new > 7:
            return
        if d < 0 and tx.wheel_rpm(self.speed, self.WHEEL_R, new) > self.cfg.redline * 0.97:
            return self.msg("Downshift refused - over-rev protection", "warn")
        tx.start_shift(d)
        tx.paddle_hold = 4.0

    def shift(self, d):
        """E (+1) / Q (-1)."""
        tx = self.tx
        if not tx.manual:
            return self.paddle(d)
        new = int(clamp(tx.gear + d, 0, 7))
        if new == tx.gear:
            return
        if new > 0 and self.clutch < 0.75:       # synchros grind without the clutch
            self.crunches += 1
            self.events.append(("crunch", None))
            return
        tx.gear = new
        tx.flash = 0.35
        if self.rev_match and d < 0 and new > 0 and self.running:
            tgt = tx.wheel_rpm(self.speed, self.WHEEL_R, new)
            if self.cfg.idle * 1.1 < tgt < self.cfg.redline * 0.97:
                self.rm_target, self.rm_t = tgt, 0.7

    def blow(self, reason):
        self.running = self.cranking = False
        self.blown, self.blown_reason = True, reason
        self.events.append(("blown", reason))

    # ----- physics --------------------------------------------------------------- #
    def engine_torque(self, rpm, thr):
        c = self.cfg
        x = rpm / c.redline
        fr = c.fric * (0.45 + 0.9 * x)
        if self.induction == 2 and self.running:      # supercharger parasitic drag
            fr += 0.05 * c.peak_torque * x
        if self.cranking:
            return (c.fric * 3.5 if rpm < 260 else 0.0) - fr
        if not self.running:
            return -fr - c.fric * 0.6
        if self.no_fire:
            return -fr - c.fric * 1.3 * x
        pump = c.fric * 1.3 * x * (1.0 - thr)
        curve = max(0.42, 1.0 - 1.15 * ((rpm - c.peak_rpm) / c.redline) ** 2)
        T = c.peak_torque * curve * thr * self.torque_mult * self.derate
        if thr < 0.25:
            target = c.idle * (1.0 + 0.8 * max(0.0, 1.0 - self.run_t / 1.8))
            if rpm < target * 1.35:
                kp = 0.0006 * c.peak_torque
                T += (1.0 - thr) * (fr + c.fric * 1.3 * x +
                                    clamp(kp * (target - rpm), -0.05 * c.peak_torque,
                                          0.3 * c.peak_torque))
        return T - fr - pump

    def step(self, dt):
        c, tx, R = self.cfg, self.tx, self.WHEEL_R
        # --- pedals ----------------------------------------------------------- #
        self.throttle += clamp(self.throttle_cmd - self.throttle, -4.0 * dt, 2.8 * dt)
        self.brake += clamp(self.brake_cmd - self.brake, -6.0 * dt, 7.0 * dt)
        if tx.manual:
            self.clutch += clamp(self.clutch_cmd - self.clutch, -1.6 * dt, 8.0 * dt)
        else:
            self.clutch = 0.0

        # --- engine state machine -------------------------------------------- #
        if self.cranking:
            self.crank_t += dt
            if self.crank_t >= 0.9:
                self.cranking = False
                if self.fuel_l > 0.01 and not self.blown:
                    self.running, self.run_t, self.stall_t = True, 0.0, 0.0
                else:
                    self.events.append(("nofuel", None))
        elif self.running:
            self.run_t += dt
        if self.refueling:
            self.fuel_l = min(c.tank, self.fuel_l + 18.0 * dt)
            if self.fuel_l >= c.tank - 1e-6:
                self.refueling = False
                self.msg("Tank full", "good")

        # --- transmission mode logic -------------------------------------------- #
        if not tx.manual:
            if not self.running and not self.cranking:
                if tx.mode == "D":
                    tx.mode = "N"
                if self.speed < 0.3:
                    tx.mode = "P"
            if self.running and tx.mode == "P":
                tx.mode = "N"
        tx.update(dt, self.rpm, self.speed, self.throttle, c, R)

        thr = self.throttle
        if tx.manual:
            if self.rm_t > 0.0:                      # rev-match blip on downshift
                self.rm_t -= dt
                if self.clutch > 0.5 and self.rpm < self.rm_target:
                    thr = max(thr, 0.65)
        elif tx.shifting:
            thr = max(thr, 0.55) if tx.shift_dir < 0 else thr * 0.25
        self.thr_eff = thr

        # --- limiter, starvation ------------------------------------------------- #
        if self.running:
            if self.rpm >= c.redline:
                self.fuel_cut = True
            elif self.rpm < c.redline - 250:
                self.fuel_cut = False
            if self.fuel_l < 0.03 * c.tank and random.random() < dt * 6.0:
                self.starve_t = 0.12
        else:
            self.fuel_cut = False
        self.starve_t = max(0.0, self.starve_t - dt)
        self.no_fire = self.fuel_cut or self.starve_t > 0.0

        # --- induction ----------------------------------------------------------- #
        rpm0 = self.rpm
        alive = self.running and not self.no_fire
        if self.induction == 1:
            tgt = thr * smoothstep((rpm0 - 1800.0) / (0.45 * c.redline)) if alive else 0.0
            tau = 0.9 if tgt > self.spool else 0.30
            self.spool += (tgt - self.spool) * (1.0 - math.exp(-dt / tau))
        elif self.induction == 2:
            tgt = thr * clamp(rpm0 / (0.45 * c.redline), 0.0, 1.0) if alive else 0.0
            self.spool += (tgt - self.spool) * (1.0 - math.exp(-dt / 0.12))
        else:
            self.spool = 0.0
        self.boost = self.IND_MAX_BOOST[self.induction] * self.spool
        if self.induction == 0:
            self.torque_mult = 1.0
        elif self.induction == 1:
            self.torque_mult = 0.88 + 0.5 * self.boost
        else:
            self.torque_mult = 1.0 + 0.45 * self.boost
        self.derate = 1.0 - clamp((self.coolant - 108.0) / 30.0, 0.0, 0.5)

        # blow-off valve on throttle lift
        self.bov_cd = max(0.0, self.bov_cd - dt)
        if self.induction == 1 and self.running and self.prev_thr >= 0.25 > thr \
                and self.boost > 0.3 and self.bov_cd <= 0.0:
            self.events.append(("bov", clamp(self.boost / 1.1, 0.35, 1.0)))
            self.spool *= 0.45
            self.bov_cd = 0.7

        # --- physics substeps ------------------------------------------------------ #
        n = max(8, int(math.ceil(dt * 960)))
        h = dt / n
        G = tx.total_ratio()
        m = c.mass
        fb = 14000.0 * self.brake + (16000.0 if (tx.mode == "P" and not tx.manual) else 0.0)
        last_eng = 0.0
        for _ in range(n):
            rpm = self.omega * 60.0 / TAU
            Te = self.engine_torque(rpm, thr)
            Tc = 0.0
            if G > 0.0:
                ww = self.speed * G / R
                rw = ww * 60.0 / TAU
                if tx.manual:
                    eng = smoothstep(((1.0 - self.clutch) - 0.25) / 0.6)    # bite curve
                elif rw >= 1300.0:
                    eng = 1.0
                else:
                    eng = max(smoothstep((rpm - (c.idle + 100.0)) / 900.0), rw / 1300.0)
                cap = min(c.peak_torque * 1.6, c.traction * m * 9.81 * R / G) * eng
                Tc = clamp(self.K_CLUTCH * (self.omega - ww), -cap, cap)
                last_eng = eng
            self.omega = max(0.0, self.omega + (Te - Tc) / c.inertia * h)
            F = Tc * G / R if G > 0.0 else 0.0
            drag = 0.5 * 1.2 * c.cda * self.speed ** 2 + (0.012 * m * 9.81 if self.speed > 0.01 else 0.0)
            self.speed += F / m * h
            self.speed = max(0.0, self.speed - (drag + fb) / m * h)
        self.eng_disp = (1.0 - self.clutch) if tx.manual else last_eng

        rpm = self.rpm
        x = rpm / c.redline

        # --- stalling --------------------------------------------------------------- #
        if self.running and self.run_t > 1.0 and rpm < 0.35 * c.idle:
            self.stall_t += dt
            if self.stall_t > 0.15:
                self.running = False
                self.stalls += 1
                self.events.append(("stall", None))
        else:
            self.stall_t = 0.0

        # --- over-rev (money shift) ----------------------------------------------- #
        if rpm > c.redline * 1.18:
            self.damage += dt * 3.0
            if self.damage >= 1.0 and not self.blown:
                self.blow("OVER-REV")

        # --- manifold, air, AFR, fuel flow ------------------------------------------- #
        if self.running:
            self.map_gauge = -0.72 * (1.0 - thr) + self.boost * thr
        elif self.cranking:
            self.map_gauge = -0.25
        else:
            self.map_gauge = 0.0
        pr = (101.3 + self.map_gauge * 100.0) / 101.3
        self.mdot_air = 1.2 * (c.displacement / 1000.0) * 0.85 * (rpm / 120.0) * pr if self.running else 0.0
        overrun = self.running and thr < 0.08 and rpm > 1500 and not self.no_fire
        lam = 1.0
        if thr > 0.55:
            lam -= 0.17 * smoothstep((thr - 0.55) / 0.4)          # power enrichment
        if self.boost > 0.3:
            lam -= 0.06
        if self.coolant < 60:
            lam -= 0.06 * clamp((60 - self.coolant) / 35.0, 0, 1)   # cold enrichment
        mdot_fuel = self.mdot_air / (STOICH * lam)
        if self.no_fire:
            mdot_fuel = 0.0
        elif overrun:
            mdot_fuel = self.mdot_air / STOICH * 0.06                # crackle trickle
        self.lam = lam
        self.afr = clamp(self.mdot_air / mdot_fuel, 0.0, 99.9) if mdot_fuel > 1e-9 else 99.9
        self.flow_lph = mdot_fuel / FUEL_DENSITY * 3600.0
        if self.running:
            self.fuel_l = max(0.0, self.fuel_l - mdot_fuel / FUEL_DENSITY * dt * FUEL_TIME_SCALE)
            if self.fuel_l <= 0.0:
                self.running = False
                self.events.append(("nofuel", None))

        # --- backfire (off-throttle at high RPM) ------------------------------------- #
        self.backfire_t = max(0.0, self.backfire_t - dt)
        if self.running and thr < 0.1 and not self.no_fire and rpm > 0.45 * c.redline and self.fuel_l > 0:
            heat = clamp((rpm - 0.45 * c.redline) / (0.5 * c.redline), 0.0, 1.0)
            if self.prev_thr >= 0.3 and rpm > 0.55 * c.redline:        # lift-off burst
                self.burst, self.burst_t = random.randint(2, 5), 0.0
            fire = False
            if self.burst > 0:
                self.burst_t -= dt
                if self.burst_t <= 0.0:
                    fire = True
                    self.burst -= 1
                    self.burst_t = random.uniform(0.05, 0.14)
            elif random.random() < 3.0 * heat * dt:
                fire = True
            if fire:
                amp = (0.45 + 0.55 * heat) * (1.25 if self.induction else 1.0)
                self.events.append(("backfire", clamp(amp, 0.2, 1.3)))
                self.backfire_t = 0.14
                self.backfires += 1
        else:
            self.burst = 0
        self.prev_thr = thr

        # --- thermals (accelerated) --------------------------------------------------- #
        fuel_power = mdot_fuel * 44e6
        friction_w = (c.fric * (0.45 + 0.9 * x) + c.fric * 1.3 * x * (1.0 - thr)) * self.omega if self.running else 0.0
        stress_w = c.displacement * 36e3 * smoothstep((x - 0.8) / 0.2) ** 1.2 if self.running else 0.0
        heat_w = 0.30 * fuel_power + 0.8 * friction_w + stress_w     # stress: high-RPM bearing/oil shear heat
        T = self.coolant
        opened = smoothstep((T - 86.0) / 12.0)                     # thermostat 86..98 C
        if T > 102.0:
            self.fan_on = True
        elif T < 96.0:
            self.fan_on = False
        k = c.displacement * (20.0 + opened * (300.0 + 40.0 * self.speed) + (200.0 if self.fan_on else 0.0))
        cool_w = k * (T - AMBIENT)
        self.coolant += THERMAL_SCALE * (heat_w - cool_w) / (14000.0 * c.displacement) * dt
        if self.coolant > 118.0:
            self.damage += (self.coolant - 118.0) / 10.0 * dt
            if self.damage >= 1.0 and not self.blown:
                self.blow("OVERHEAT")
        elif self.coolant < 110.0 and rpm < c.redline * 1.1:
            self.damage = max(0.0, self.damage - 0.02 * dt)

        # --- oil pressure ----------------------------------------------------------------- #
        if self.blown or not (self.running or self.cranking):
            tgt = 0.0
        else:
            tgt = clamp((0.9 + 0.00105 * rpm) * (1.1 - 0.004 * (self.coolant - 60.0)), 0.0, 5.5)
        self.oil += (tgt - self.oil) * (1.0 - math.exp(-dt / 0.4))


# --------------------------------------------------------------------------- #
# UI widgets
# --------------------------------------------------------------------------- #
class Button:
    def __init__(self, rect, label, on_click=None, hold=None, active=None, enabled=None):
        self.rect = pygame.Rect(rect)
        self.label, self.on_click, self.hold = label, on_click, hold
        self.active, self.enabled = active, enabled

    def draw(self, surf, font, mouse, held):
        enabled = self.enabled() if self.enabled else True
        active = (self.active() if self.active else False) or (self.hold is not None and held == self.hold)
        hover = self.rect.collidepoint(mouse) and enabled
        if not enabled:
            bg, fg = (28, 31, 40), (90, 96, 112)
        elif active:
            bg, fg = (210, 100, 25), (255, 255, 255)
        elif hover:
            bg, fg = (62, 72, 100), (245, 248, 255)
        else:
            bg, fg = (40, 47, 66), (215, 222, 238)
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, (85, 96, 125), self.rect, 1, border_radius=8)
        label = self.label() if callable(self.label) else self.label
        img = font.render(label, True, fg)
        surf.blit(img, img.get_rect(center=self.rect.center))


# --------------------------------------------------------------------------- #
# MainApp
# --------------------------------------------------------------------------- #
class MainApp:
    W, H = 1440, 840
    BG, PANEL, BORDER = (13, 15, 21), (21, 25, 35), (52, 60, 82)
    TEXT, DIM = (222, 228, 242), (128, 138, 160)
    ORANGE, RED, GREEN = (255, 140, 30), (230, 45, 45), (70, 205, 125)
    AMBER, BLUE = (245, 190, 40), (80, 170, 255)
    ALERT_COL = {"info": (120, 190, 255), "warn": (255, 190, 50), "bad": (255, 70, 60), "good": (90, 220, 130)}

    def __init__(self, fuel_pct=100.0):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption("Procedural Engine Simulator v2")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        pick = "dejavusans,arial,helvetica"
        self.f_title = pygame.font.SysFont(pick, 24, bold=True)
        self.f_huge = pygame.font.SysFont(pick, 50, bold=True)
        self.f_big = pygame.font.SysFont(pick, 28, bold=True)
        self.f_mid = pygame.font.SysFont(pick, 17, bold=True)
        self.f_small = pygame.font.SysFont(pick, 14)
        self.f_smallb = pygame.font.SysFont(pick, 14, bold=True)
        self.f_tiny = pygame.font.SysFont(pick, 12)

        self.catalog = build_catalog()
        self.engine_idx = 0
        self.variant_idx = [0] * len(self.catalog)
        self.units = "kmh"
        self.held_ui = None
        self.disp_rpm = 0.0
        self.alerts: List[list] = []
        self.flame_t = 0.0

        self.tx = Transmission()
        self.vehicle = Vehicle(self.tx, fuel_pct)
        self.synth = AudioSynthesizer()

        self.cyl_rect = pygame.Rect(20, 62, 620, 340)
        self.gauge_rect = pygame.Rect(650, 62, 770, 340)
        self.scope_rect = pygame.Rect(20, 410, 620, 152)
        self.hud_rect = pygame.Rect(650, 410, 770, 152)
        self.glow_layer = pygame.Surface(self.cyl_rect.size, pygame.SRCALPHA)
        self.fire_phase, self.fire_slot = 0.0, 0
        self.cylinders: List[Cylinder] = []
        self.buttons = self._build_buttons()
        self.select_engine(0)

    # ----- engine management ------------------------------------------------ #
    @property
    def cfg(self) -> EngineConfig:
        return self.catalog[self.engine_idx][self.variant_idx[self.engine_idx]]

    def select_engine(self, idx):
        self.engine_idx = idx
        self._apply_engine()

    def cycle_variant(self):
        variants = self.catalog[self.engine_idx]
        if len(variants) > 1:
            self.variant_idx[self.engine_idx] = (self.variant_idx[self.engine_idx] + 1) % len(variants)
            self._apply_engine()

    def _apply_engine(self):
        cfg = self.cfg
        self.vehicle.set_engine(cfg)
        self.synth.set_engine(cfg)
        self.cylinders = cfg.make_cylinders()
        self.fire_phase, self.fire_slot = 0.0, 0
        self._layout_cylinders()

    def _layout_cylinders(self):
        cfg, rect = self.cfg, self.cyl_rect
        inner = rect.inflate(-70, -120)
        inner.move_ip(0, 10)
        B = len(cfg.banks)
        cols = max(len(b) for b in cfg.banks)
        px, py = inner.w / cols, inner.h / B
        radius = int(min(px * 0.38, py * 0.40, 46))
        self.crank_y, self.inner = inner.centery, inner
        self.num_font = pygame.font.SysFont("dejavusans,arial", max(11, int(radius * 0.6)), bold=True)
        for bi, bank in enumerate(cfg.banks):
            y = inner.top + py * (bi + 0.5)
            off = (0.11 * px if bi % 2 else -0.11 * px) if B > 1 else 0.0
            for ci, num in enumerate(bank):
                cyl = self.cylinders[num - 1]
                cyl.pos = (inner.left + px * (ci + 0.5) + off, y)
                cyl.radius = radius

    def _build_buttons(self):
        b = []
        v, tx = self.vehicle, self.tx
        names = ["I4", "V6", "V8", "V10", "V12", "W16", "Flat-6"]
        gap, w = 6, (620 - 6 * 6) // 7
        for i, nm in enumerate(names):
            b.append(Button((20 + i * (w + gap), 596, w, 40), f"{i + 1} {nm}",
                            on_click=lambda i=i: self.select_engine(i),
                            active=lambda i=i: self.engine_idx == i))
        # row A (right): engine / transmission
        wA = (760 - 3 * 8) // 4
        rowA = [
            (lambda: "STOP" if (v.running or v.cranking) else "START (Space)", v.toggle,
             lambda: v.running or v.cranking, None),
            (lambda: "MANUAL (M)" if tx.manual else "AUTO (M)", v.toggle_manual, lambda: tx.manual, None),
            ("DRIVE (D)", v.request_drive, lambda: (not tx.manual) and tx.mode == "D",
             lambda: v.running or tx.manual),
            ("NEUTRAL (N)", v.request_neutral,
             lambda: (tx.gear == 0) if tx.manual else tx.mode == "N", None),
        ]
        for i, (lab, cb, act, en) in enumerate(rowA):
            b.append(Button((660 + i * (wA + 8), 596, wA, 40), lab, on_click=cb, active=act, enabled=en))
        # row B (left): shifting
        wB = (620 - 2 * 8) // 3
        rowB = [
            ("SHIFT DOWN (Q)", lambda: v.shift(-1), None, None),
            ("SHIFT UP (E)", lambda: v.shift(+1), None, None),
            (lambda: "REV-MATCH: ON (H)" if v.rev_match else "REV-MATCH: OFF (H)",
             lambda: setattr(v, "rev_match", not v.rev_match), lambda: v.rev_match, None),
        ]
        for i, (lab, cb, act, en) in enumerate(rowB):
            b.append(Button((20 + i * (wB + 8), 644, wB, 40), lab, on_click=cb, active=act, enabled=en))
        # row B (right): systems
        wC = (760 - 5 * 8) // 6
        rowC = [
            (lambda: {"NA": "NA (T)", "TURBO": "TURBO (T)", "SUPERCHARGER": "SUPERCHG (T)"}[v.ind_name],
             v.toggle_induction, lambda: v.induction > 0, None),
            ("REFUEL (R)", v.start_refuel, lambda: v.refueling, None),
            ("REPAIR (K)", v.repair, None, lambda: v.blown),
            ("VARIANT (V)", self.cycle_variant, None, lambda: len(self.catalog[self.engine_idx]) > 1),
            (lambda: "UNITS: km/h" if self.units == "kmh" else "UNITS: mph", self.toggle_units, None, None),
            (lambda: "SOUND: OFF" if self.synth.muted else "SOUND: ON (X)", self.toggle_mute,
             lambda: self.synth.muted, None),
        ]
        for i, (lab, cb, act, en) in enumerate(rowC):
            b.append(Button((660 + i * (wC + 8), 644, wC, 40), lab, on_click=cb, active=act, enabled=en))
        # pedals
        b.append(Button((20, 694, 340, 80), "CLUTCH  [ Shift / C ]  hold", hold="clutch",
                        enabled=lambda: tx.manual))
        b.append(Button((370, 694, 340, 80), "BRAKE  [ S / Down ]  hold", hold="brake"))
        b.append(Button((720, 694, 700, 80), "THROTTLE  [ W / Up ]  hold", hold="throttle"))
        return b

    def toggle_units(self):
        self.units = "mph" if self.units == "kmh" else "kmh"

    def toggle_mute(self):
        self.synth.muted = not self.synth.muted

    def alert(self, text, kind="warn", dur=2.8):
        self.alerts.append([text, self.ALERT_COL.get(kind, self.ALERT_COL["warn"]), dur])
        self.alerts = self.alerts[-4:]

    # ----- update ------------------------------------------------------------ #
    def update(self, dt):
        keys = pygame.key.get_pressed()
        v, tx, s = self.vehicle, self.tx, self.synth
        v.throttle_cmd = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP] or self.held_ui == "throttle") else 0.0
        v.brake_cmd = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN] or self.held_ui == "brake") else 0.0
        clutch_down = keys[pygame.K_c] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or self.held_ui == "clutch"
        v.clutch_cmd = 1.0 if (tx.manual and clutch_down) else 0.0
        v.step(dt)

        for ev, payload in v.drain():
            if ev == "crunch":
                s.trigger("crunch")
                self.alert("GEAR CRUNCH!  Press the clutch fully first", "bad")
            elif ev == "stall":
                s.trigger("stall")
                self.alert("ENGINE STALLED", "bad")
            elif ev == "bov":
                s.trigger("bov", payload)
            elif ev == "backfire":
                s.trigger("backfire", payload)
                self.flame_t = 0.14
            elif ev == "blown":
                s.trigger("blow")
                self.alert("ENGINE FAILURE: " + payload, "bad", 5.0)
            elif ev == "nofuel":
                self.alert("OUT OF FUEL - engine stopped" if v.fuel_l <= 0.01 else "Engine did not start", "bad", 4.0)
            elif ev == "msg":
                self.alert(payload[0], payload[1])

        cfg = self.cfg
        if v.running and not v.no_fire:
            self.fire_phase += cfg.pulse_hz(v.rpm) * dt
            guard = 0
            while self.fire_phase >= 1.0 and guard < 64:
                self.fire_phase -= 1.0
                self.cylinders[cfg.fire_order[self.fire_slot] - 1].fire()
                self.fire_slot = (self.fire_slot + 1) % cfg.cylinders
                guard += 1
        for c in self.cylinders:
            c.update(dt)
        self.flame_t = max(0.0, self.flame_t - dt)
        for a in self.alerts:
            a[2] -= dt
        self.alerts = [a for a in self.alerts if a[2] > 0]

        s.rpm, s.throttle = v.rpm, v.thr_eff
        s.running, s.cranking, s.fuel_cut = v.running, v.cranking, v.no_fire
        s.induction, s.spool, s.boost = v.induction, v.spool, v.boost
        self.disp_rpm += (v.rpm - self.disp_rpm) * min(1.0, dt * 20.0)

    # ----- drawing helpers --------------------------------------------------- #
    def text(self, s, font, color, pos, anchor="center"):
        img = font.render(s, True, color)
        self.screen.blit(img, img.get_rect(**{anchor: pos}))

    def panel(self, rect):
        pygame.draw.rect(self.screen, self.PANEL, rect, border_radius=12)
        pygame.draw.rect(self.screen, self.BORDER, rect, 1, border_radius=12)

    @staticmethod
    def _pt(cx, cy, r, ang):
        a = math.radians(ang)
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    def draw_header(self):
        cfg, v = self.cfg, self.vehicle
        self.text("PROCEDURAL ENGINE SIMULATOR", self.f_title, self.TEXT, (20, 12), "topleft")
        desc = f"{cfg.label}  \u2022  {cfg.displacement:.1f} L  \u2022  {v.ind_name}  \u2022  {cfg.description}"
        self.text(desc, self.f_small, self.DIM, (20, 42), "topleft")
        if v.blown:
            st, col = "ENGINE FAILED", self.RED
        elif v.cranking:
            st, col = "CRANKING", self.ORANGE
        elif v.running:
            st, col = ("REV LIMITER", self.RED) if v.fuel_cut else ("RUNNING", self.GREEN)
        else:
            st, col = "ENGINE OFF", self.DIM
        pygame.draw.circle(self.screen, col, (self.W - 160, 26), 8)
        self.text(st, self.f_mid, col, (self.W - 144, 26), "midleft")
        if self.alerts:
            text, color, left = self.alerts[-1]
            if left > 0.6 or int(left * 8) % 2 == 0:
                self.text(text, self.f_big, color, (self.W // 2 + 120, 22))

    def draw_cylinders(self):
        s, rect, cfg = self.screen, self.cyl_rect, self.cfg
        self.panel(rect)
        self.text("CYLINDER BANKS  \u2013  top view, crank axis horizontal", self.f_tiny, self.DIM,
                  (rect.left + 16, rect.top + 12), "topleft")
        pygame.draw.line(s, (60, 68, 90), (self.inner.left - 8, self.crank_y),
                         (self.inner.right + 8, self.crank_y), 2)
        gl = self.glow_layer
        gl.fill((0, 0, 0, 0))
        for c in self.cylinders:
            if c.glow > 0.03:
                px, py = c.pos[0] - rect.x, c.pos[1] - rect.y
                for scale, a in ((1.0 + 0.9 * c.glow, 45), (1.0 + 0.55 * c.glow, 80), (1.0 + 0.25 * c.glow, 120)):
                    pygame.draw.circle(gl, (255, 110, 20, int(a * c.glow)), (px, py), int(c.radius * scale))
        s.blit(gl, rect.topleft)
        for bi in range(len(cfg.banks)):
            first = self.cylinders[cfg.banks[bi][0] - 1]
            self.text(chr(65 + bi), self.f_mid, self.DIM, (self.inner.left - 24, first.pos[1]))
        for c in self.cylinders:
            x, y = int(c.pos[0]), int(c.pos[1])
            pygame.draw.circle(s, (90, 100, 125), (x, y), c.radius + 2)
            pygame.draw.circle(s, Cylinder.glow_color(c.glow), (x, y), c.radius)
            pygame.draw.circle(s, lerp_color((60, 68, 88), (255, 220, 120), c.glow), (x, y),
                               int(c.radius * 0.62), 2)
            self.text(str(c.number), self.num_font,
                      lerp_color((170, 180, 200), (255, 255, 255), c.glow), (x, y))
        ix, iy = rect.right - 52, rect.top + 52
        for a in cfg.bank_angles:
            r = math.radians(a)
            pygame.draw.line(s, (150, 160, 185), (ix, iy), (ix + 34 * math.sin(r), iy - 34 * math.cos(r)), 5)
        pygame.draw.circle(s, self.ORANGE, (ix, iy), 5)
        self.text("end view", self.f_tiny, self.DIM, (ix, iy + 44))
        # exhaust tip + backfire flame
        ex, ey = rect.right - 22, rect.bottom - 52
        pygame.draw.rect(s, (80, 88, 110), (ex - 14, ey - 7, 22, 14), border_radius=4)
        if self.flame_t > 0:
            f = self.flame_t / 0.14
            L = int(26 + 34 * f * random.random() + 14)
            pts = [(ex + 8, ey - 8), (ex + 8 + L, ey - 2 + random.randint(-5, 5)),
                   (ex + 8 + L * 0.55, ey + 2), (ex + 8 + L * 0.9, ey + 8), (ex + 8, ey + 8)]
            pygame.draw.polygon(s, (255, 150, 30), pts)
            pygame.draw.polygon(s, (255, 235, 140), [(ex + 8, ey - 4), (ex + 8 + L * 0.5, ey), (ex + 8, ey + 4)])
        order = "-".join(str(n) for n in cfg.fire_order)
        self.text(f"Firing order: {order}", self.f_small, self.TEXT, (rect.centerx - 20, rect.bottom - 18))

    def draw_gauge(self, cx, cy, r, value, vmin, vmax, major, minor, divisor=1.0, redline=None,
                   title="", fmt=None):
        s = self.screen
        pygame.draw.circle(s, (8, 10, 15), (cx, cy), r + 9)
        pygame.draw.circle(s, (70, 80, 105), (cx, cy), r + 9, 3)
        pygame.draw.circle(s, (17, 20, 29), (cx, cy), r)
        A0, SW = 135.0, 270.0

        def ang(val):
            return A0 + SW * clamp((val - vmin) / (vmax - vmin), 0.0, 1.0)

        if redline is not None:
            pts = [self._pt(cx, cy, r * 0.90, ang(redline + (vmax - redline) * i / 30)) for i in range(31)]
            pygame.draw.lines(s, (190, 28, 28), False, pts, 7)
        steps = int(round((vmax - vmin) / minor))
        for i in range(steps + 1):
            val = vmin + i * minor
            q = (val - vmin) / major
            is_major = abs(q - round(q)) < 1e-6
            col = (236, 240, 248) if is_major else (115, 124, 148)
            if redline is not None and val >= redline:
                col = (255, 95, 75)
            r1 = r * (0.78 if is_major else 0.86)
            pygame.draw.line(s, col, self._pt(cx, cy, r1, ang(val)), self._pt(cx, cy, r * 0.95, ang(val)),
                             3 if is_major else 1)
            if is_major:
                lx, ly = self._pt(cx, cy, r * 0.63, ang(val))
                lab = fmt(val) if fmt else str(int(round(val / divisor)))
                self.text(lab, self.f_smallb if r < 100 else self.f_mid, self.TEXT, (lx, ly))
        self.text(title, self.f_tiny, self.DIM, (cx, cy - r * 0.30))
        a = ang(value)
        tip, tail = self._pt(cx, cy, r * 0.86, a), self._pt(cx, cy, r * 0.14, a + 180)
        p1, p2 = self._pt(cx, cy, 5, a + 90), self._pt(cx, cy, 5, a - 90)
        pygame.draw.polygon(s, (255, 85, 40), [tip, p1, tail, p2])
        pygame.draw.circle(s, (45, 52, 70), (cx, cy), 10)
        pygame.draw.circle(s, (255, 85, 40), (cx, cy), 4)

    def draw_chip(self, x, y, w, text, col):
        r = pygame.Rect(x, y, w, 26)
        pygame.draw.rect(self.screen, (28, 33, 46), r, border_radius=8)
        pygame.draw.rect(self.screen, col, r, 2, border_radius=8)
        self.text(text, self.f_smallb, col, r.center)

    def draw_gauges(self):
        self.panel(self.gauge_rect)
        cfg, v, tx = self.cfg, self.vehicle, self.tx
        tach_max = max(9000, int(math.ceil((cfg.redline + 800) / 1000.0)) * 1000)
        r = 128
        tcx, tcy, scx, scy = 790, 208, 1075, 208
        self.draw_gauge(tcx, tcy, r, self.disp_rpm, 0, tach_max, 1000, 500, 1000,
                        redline=cfg.redline, title="RPM x1000")
        self.text(f"{int(self.disp_rpm):d}", self.f_big, self.TEXT, (tcx, tcy + 48))
        gcol = self.ORANGE if not tx.manual else self.BLUE
        self.text(tx.display(), self.f_huge, gcol, (tcx, tcy + 92))
        spd = v.speed * (3.6 if self.units == "kmh" else 2.23694)
        vmax = 400 if self.units == "kmh" else 250
        self.draw_gauge(scx, scy, r, spd, 0, vmax, 50, 10, 1, title="SPEED")
        self.text(f"{int(spd):d}", self.f_big, self.TEXT, (scx, scy + 52))
        self.text("km/h" if self.units == "kmh" else "mph", self.f_mid, self.DIM, (scx, scy + 86))

        # boost / manifold gauge
        bcx, bcy = 1330, 150
        self.draw_gauge(bcx, bcy, 72, v.map_gauge, -1.0, 2.0, 1.0, 0.25, redline=1.4,
                        title="", fmt=lambda val: f"{val:.0f}")
        self.text(f"{v.map_gauge:+.2f}", self.f_smallb, self.TEXT, (bcx, bcy + 42))
        self.text("MAP bar", self.f_tiny, self.DIM, (bcx, bcy + 58))

        # status chips
        x, y = 1252, 238
        self.draw_chip(x, y, 160, "TRANS: MANUAL" if tx.manual else "TRANS: AUTO",
                       self.BLUE if tx.manual else self.GREEN)
        mode = ("GEAR N" if tx.gear == 0 else f"GEAR {tx.gear}") if tx.manual else \
               ("PARK" if tx.mode == "P" else "NEUTRAL" if tx.mode == "N" else f"DRIVE  {tx.display()}")
        self.draw_chip(x, y + 31, 160, mode, self.ORANGE)
        self.draw_chip(x, y + 62, 160, "REV-MATCH " + ("ON" if v.rev_match else "OFF"),
                       self.GREEN if v.rev_match else self.DIM)
        self.draw_chip(x, y + 93, 160, v.ind_name,
                       self.AMBER if v.induction else self.DIM)
        self.draw_chip(x, y + 124, 160, f"CLUTCH {int(v.eng_disp * 100):d}%",
                       self.GREEN if v.eng_disp > 0.9 else self.AMBER if v.eng_disp > 0.1 else self.RED)

        # gear strip
        strip = pygame.Rect(670, 360, 560, 30)
        pygame.draw.rect(self.screen, (13, 16, 24), strip, border_radius=8)
        labels = (["N"] if tx.manual else ["P", "N"]) + [str(i) for i in range(1, 8)]
        cw = strip.w / len(labels)
        cur = tx.display()
        for i, lab in enumerate(labels):
            cell = pygame.Rect(strip.x + i * cw, strip.y, cw, strip.h)
            if lab == cur:
                pygame.draw.rect(self.screen, gcol, cell.inflate(-6, -6), border_radius=6)
                self.text(lab, self.f_mid, (20, 20, 20), cell.center)
            else:
                self.text(lab, self.f_mid, self.DIM, cell.center)

    def hud_bar(self, col, row, label, frac, text, color, marks=()):
        x0 = 664 + col * 380
        y = 420 + row * 28
        self.text(label, self.f_tiny, self.DIM, (x0, y + 10), "midleft")
        bar = pygame.Rect(x0 + 62, y + 2, 170, 16)
        pygame.draw.rect(self.screen, (13, 16, 24), bar, border_radius=5)
        w = int(bar.w * clamp(frac, 0.0, 1.0))
        if w > 0:
            pygame.draw.rect(self.screen, color, (bar.x, bar.y, w, bar.h), border_radius=5)
        for m in marks:
            mx = bar.x + int(bar.w * clamp(m, 0, 1))
            pygame.draw.line(self.screen, (255, 255, 255), (mx, bar.y - 2), (mx, bar.bottom + 2), 2)
        pygame.draw.rect(self.screen, self.BORDER, bar, 1, border_radius=5)
        self.text(text, self.f_small, self.TEXT, (bar.right + 10, y + 10), "midleft")

    def draw_hud(self):
        self.panel(self.hud_rect)
        v, cfg, tx = self.vehicle, self.cfg, self.tx
        blink = int(time.time() * 4) % 2 == 0
        # fuel
        fp = v.fuel_pct
        fcol = self.GREEN if fp > 25 else self.AMBER if fp > 10 else (self.RED if blink else (120, 30, 30))
        self.hud_bar(0, 0, "FUEL", fp / 100.0, f"{v.fuel_l:.1f} L  {fp:.0f}%", fcol)
        # coolant
        T = v.coolant
        ccol = self.BLUE if T < 70 else self.GREEN if T < 105 else self.AMBER if T < 118 else \
            (self.RED if blink else (120, 30, 30))
        self.hud_bar(0, 1, "COOLANT", (T - 40.0) / 95.0, f"{T:.0f} \u00b0C" + ("  FAN" if v.fan_on else ""), ccol)
        # oil
        ocol = self.RED if (v.running and v.oil < 0.8) else self.GREEN
        self.hud_bar(0, 2, "OIL PRESS", v.oil / 6.0, f"{v.oil:.1f} bar", ocol)
        # AFR
        if v.running and not v.no_fire and v.afr < 99:
            acol = self.ORANGE if v.afr < 13.5 else self.GREEN if v.afr < 15.5 else self.BLUE
            atxt = f"{v.afr:.1f}:1  \u03bb{(v.afr / 14.7):.2f}"
            afrac = (v.afr - 9.0) / 15.0
        else:
            acol, atxt, afrac = self.DIM, ("CUT" if v.running else "--"), 0.0
        self.hud_bar(0, 3, "AFR", afrac, atxt, acol, marks=((14.7 - 9) / 15.0,))
        # consumption
        kmh = v.speed * 3.6
        if v.running and kmh > 8:
            ftxt = f"{v.flow_lph / kmh * 100:.1f} L/100km"
        else:
            ftxt = f"{v.flow_lph:.1f} L/h"
        self.hud_bar(0, 4, "FLOW", v.flow_lph / 150.0, ftxt, self.AMBER)
        # right column
        manual = tx.manual
        ecol = self.GREEN if v.eng_disp > 0.9 else self.AMBER if v.eng_disp > 0.1 else self.RED
        self.hud_bar(1, 0, "CLUTCH", v.eng_disp, f"{v.eng_disp * 100:.0f}%  " + ("PEDAL" if manual else "AUTO"), ecol)
        self.hud_bar(1, 1, "THROTTLE", v.throttle, f"{v.throttle * 100:.0f}%", self.GREEN)
        self.hud_bar(1, 2, "BRAKE", v.brake, f"{v.brake * 100:.0f}%", self.RED)
        mabs = 101.3 + v.map_gauge * 100.0
        self.hud_bar(1, 3, "MAP", mabs / 300.0, f"{mabs:.0f} kPa abs", self.AMBER if v.boost > 0.05 else self.BLUE)
        if manual:
            R = v.WHEEL_R
            g = tx.gear
            dn = tx.wheel_rpm(v.speed, R, max(1, g - 1)) if g > 1 else (tx.wheel_rpm(v.speed, R, 1) if g == 0 else 0)
            up = tx.wheel_rpm(v.speed, R, g + 1) if 0 < g < 7 else 0
            marks = [m / cfg.redline for m in (dn, up) if 0 < m < cfg.redline * 1.05]
            txt = f"dn {int(dn):d}  up {int(up):d}" if (dn or up) else "--"
            self.hud_bar(1, 4, "MATCH", v.rpm / cfg.redline, txt, self.BLUE, marks=marks)
        elif v.induction:
            self.hud_bar(1, 4, "BOOST", v.boost / 1.2, f"{v.boost:.2f} bar", self.AMBER)
        else:
            self.hud_bar(1, 4, "TORQUE", v.rpm / cfg.redline if v.running else 0.0,
                         f"{cfg.peak_torque:.0f} Nm pk", self.DIM)

    def draw_scope_and_help(self):
        s, v, cfg = self.screen, self.vehicle, self.cfg
        rect = self.scope_rect
        self.panel(rect)
        self.text("SYNTH OUTPUT", self.f_tiny, self.DIM, (rect.x + 12, rect.y + 8), "topleft")
        hz = cfg.pulse_hz(v.rpm)
        self.text(f"Hz = (RPM/60) x (Cyl/2) = ({int(v.rpm)}/60) x ({cfg.cylinders}/2) = {hz:.0f} Hz",
                  self.f_small, self.TEXT, (rect.right - 12, rect.y + 8), "topright")
        mid = rect.centery + 10
        pygame.draw.line(s, (45, 52, 72), (rect.x + 8, mid), (rect.right - 8, mid), 1)
        if self.synth.enabled:
            wave = self.synth.scope[::3]
            pts = [(rect.x + 10 + i * (rect.w - 20) / max(1, len(wave) - 1), mid - float(a) * 52)
                   for i, a in enumerate(wave)]
            if len(pts) > 1:
                pygame.draw.lines(s, (80, 220, 255), False, pts, 1)
        else:
            self.text("audio device unavailable", self.f_small, self.DIM, rect.center)
        stats = f"stalls {v.stalls}   gear crunches {v.crunches}   backfires {v.backfires}"
        self.text(stats, self.f_tiny, self.DIM, (rect.x + 12, rect.bottom - 8), "bottomleft")
        self.text("ENGINE", self.f_tiny, self.DIM, (20, 576), "topleft")
        self.text("TRANSMISSION / SYSTEMS", self.f_tiny, self.DIM, (660, 576), "topleft")
        self.text("W/Up throttle  S/Down brake  Shift/C clutch  Space start/stop  1-7 engine  V variant  U units  X mute",
                  self.f_small, self.DIM, (20, 786), "topleft")
        self.text("M manual/auto  E shift up  Q shift down  D drive  N neutral  H rev-match  T induction  R refuel  K repair  Esc quit",
                  self.f_small, self.DIM, (20, 806), "topleft")
        self.text(f"demo clocks: fuel x{FUEL_TIME_SCALE:.0f}, thermal x{THERMAL_SCALE:.0f}",
                  self.f_tiny, self.DIM, (self.W - 16, 822), "bottomright")

    def draw(self):
        self.screen.fill(self.BG)
        self.draw_header()
        self.draw_cylinders()
        self.draw_gauges()
        self.draw_hud()
        self.draw_scope_and_help()
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            font = self.f_mid if b.hold else self.f_small
            b.draw(self.screen, font, mouse, self.held_ui)

    # ----- events / main loop ------------------------------------------------ #
    def on_mouse_down(self, pos):
        for b in self.buttons:
            if b.rect.collidepoint(pos) and (b.enabled() if b.enabled else True):
                if b.hold:
                    self.held_ui = b.hold
                elif b.on_click:
                    b.on_click()
                return

    def on_key(self, k):
        v = self.vehicle
        if k == pygame.K_SPACE:
            v.toggle()
        elif pygame.K_1 <= k <= pygame.K_7:
            self.select_engine(k - pygame.K_1)
        elif k == pygame.K_v:
            self.cycle_variant()
        elif k == pygame.K_d:
            v.request_drive()
        elif k == pygame.K_n:
            v.request_neutral()
        elif k == pygame.K_m:
            v.toggle_manual()
        elif k == pygame.K_e:
            v.shift(+1)
        elif k == pygame.K_q:
            v.shift(-1)
        elif k == pygame.K_h:
            v.rev_match = not v.rev_match
            self.alert("Rev-match " + ("ON" if v.rev_match else "OFF"), "info")
        elif k == pygame.K_t:
            v.toggle_induction()
        elif k == pygame.K_r:
            v.start_refuel()
        elif k == pygame.K_k:
            v.repair()
        elif k == pygame.K_u:
            self.toggle_units()
        elif k == pygame.K_x:
            self.toggle_mute()
        elif k in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT) and not self.tx.manual:
            self.alert("Automatic clutch - press M for manual", "info", 1.8)

    def run(self):
        self.synth.start()
        alive = True
        while alive:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        alive = False
                    else:
                        self.on_key(e.key)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.on_mouse_down(e.pos)
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    self.held_ui = None
            self.update(dt)
            self.draw()
            pygame.display.flip()
        self.synth.stop()
        pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procedural engine simulator")
    ap.add_argument("--fuel", type=float, default=100.0, help="starting fuel level in percent")
    args = ap.parse_args()
    MainApp(fuel_pct=clamp(args.fuel, 0.5, 100.0)).run()