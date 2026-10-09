#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural Car Engine Simulator
===============================
Real-time engine physics + procedural (NumPy) audio synthesis + live cylinder
visualisation + dashboard gauges, rendered with pygame.

SETUP
-----
    pip install pygame numpy
    python engine_simulator.py

CONTROLS
--------
    W / Up      Throttle (hold)         S / Down    Brake (hold)
    LShift / C  Clutch (hold, manual)   Space       Start / Stop engine
    E / Q       Shift up / down (manual gearbox)
    M           Toggle manual <-> automatic gearbox
    T           Cycle induction: NA / turbocharger / supercharger
    R           Refuel                  X           Rev-match assist on/off
    F           Fast fuel drain (demo)  B           Mute
    D           Drive (auto gearbox)    N           Neutral (free-rev)
    1-7         Engine: I4 V6 V8 V10 V12 W16 Flat-6
    V           Cycle variant           U           km/h <-> mph
    Esc         Quit
    Every key also has an on-screen button (hold the pedal buttons).

SIMULATED SYSTEMS
-----------------
* Manual gearbox: hold the clutch (LShift/C), select gears with E/Q, release
  the clutch gently to launch. Dump it at low rpm and the engine stalls.
  Shift without the clutch and you get a gear crunch.
* Fuel: air-mass model with live AFR (rich under load, lean on cruise,
  enriched when cold). Tank empties -> engine sputters and dies; R refuels.
* Cooling: coolant warms up (cold engines make less power); ram air + fan
  cool it. Sustained high rpm / low speed overheats -> engine failure
  (recovers once it cools below 95 C).
* Induction: NA, turbo (lag, boost gauge, BOV dump on throttle lift) or
  supercharger (mechanical whine, near-instant boost).
* Backfires: overrun crackles + loud bangs with exhaust flames when you
  lift off at high rpm.
* Money-shift protection: over-revving past redline on a bad downshift can
  destroy the engine. Stalling while rolling in gear can bump-start it.

Sound model: Hz = (RPM / 60) * (Cylinders / 2) firing pulses per second. Every
pulse is a pre-synthesised burst (pitch-dropping sine + band-passed noise +
tanh overdrive, per cylinder, per RPM band, per load) that is overlap-added at
the exact firing instants, plus phase-continuous harmonics, a rumble/intake bed
and overrun gurgle/pops. BOV dumps, gear crunches, backfire bangs and stall
sputters are pre-rendered one-shot samples mixed live.
"""

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pygame

TAU = 2.0 * math.pi
CHUNK = 768            # audio samples rendered per block (~17 ms)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def smoothstep(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------- #
# Cylinder & EngineConfig
# --------------------------------------------------------------------------- #
class Cylinder:
    """One cylinder: layout position + firing glow state."""
    FADE_RATE = 11.0   # 1/s exponential fade of the combustion flash

    def __init__(self, number: int, bank: int, col: int):
        self.number = number
        self.bank = bank
        self.col = col
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
        cold = (38, 44, 58)
        red = (205, 38, 28)
        orange = (255, 160, 35)
        if g < 0.5:
            return lerp_color(cold, red, g * 2.0)
        return lerp_color(red, orange, (g - 0.5) * 2.0)


@dataclass
class EngineConfig:
    # identity / layout
    key: str
    name: str
    variant: str
    description: str
    fire_order: List[int]
    banks: List[List[int]]            # cylinder numbers per bank (front -> back)
    bank_angles: List[float]          # degrees from vertical (end-view icon)
    # powertrain
    peak_torque: float                # Nm
    peak_rpm: float
    redline: float
    idle: float
    top_speed: float                  # km/h (sets final drive ratio)
    mass: float = 1500.0
    cda: float = 0.64
    traction: float = 0.9
    # acoustic "voice"
    body_freq: float = 100.0          # Hz, thump resonance of each burst
    body_decay: float = 0.025         # s
    noise_band: Tuple[float, float] = (300.0, 3000.0)
    noise_gain: float = 0.5
    drive: float = 1.5                # overdrive amount
    hum_gain: float = 0.2             # phase-continuous harmonic layer
    hum_weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.15)
    sub_gain: float = 0.3             # bass / rumble bed
    interval_pattern: Tuple[float, ...] = (1.0,)   # uneven firing intervals
    bank_amps: Tuple[float, ...] = (1.0,)          # per-bank exhaust loudness

    def __post_init__(self):
        n = len(self.fire_order)
        assert sorted(self.fire_order) == list(range(1, n + 1)), self.name
        assert sorted(c for b in self.banks for c in b) == list(range(1, n + 1))
        self.cylinders = n
        self.fric = 0.06 * self.peak_torque
        self.inertia = self.peak_torque / 550.0
        # estimated displacement (litres) for the fuel / AFR model
        self.disp = clamp(self.peak_torque / 110.0, 1.6, 9.0)
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
    """Returns one list of variants per engine (index = key 1..7)."""
    E = EngineConfig
    i4 = [E("I4", "Inline-4", "", "Buzzy, high-revving four. Firing 1-3-4-2.",
            [1, 3, 4, 2], [[1, 2, 3, 4]], [0],
            250, 4800, 7500, 850, 215, mass=1250,
            body_freq=150, body_decay=0.016, noise_band=(700, 4000), noise_gain=0.55,
            drive=1.3, hum_gain=0.20, hum_weights=(1, .6, .35, .2), sub_gain=0.12)]

    v6_common = dict(peak_torque=400, peak_rpm=5500, redline=7500, idle=800, top_speed=260,
                     mass=1450, body_freq=125, body_decay=0.019, noise_band=(500, 3500),
                     noise_gain=0.5, drive=1.4, hum_gain=0.22,
                     hum_weights=(1, .55, .3, .18), sub_gain=0.18)
    v6 = [E("V6", "V6", "60\u00b0", "Smooth, even-fire 60\u00b0 V6. Firing 1-2-5-6-3-4.",
            [1, 2, 5, 6, 3, 4], [[1, 3, 5], [2, 4, 6]], [-30, 30], **v6_common),
          E("V6", "V6", "90\u00b0", "Odd-fire 90\u00b0 V6 with uneven pulses. Firing 1-2-5-6-3-4.",
            [1, 2, 5, 6, 3, 4], [[1, 3, 5], [2, 4, 6]], [-45, 45],
            interval_pattern=(1.25, 0.75), bank_amps=(1.0, 0.85), **v6_common)]

    v8_common = dict(peak_torque=600, peak_rpm=5000, redline=7200, idle=750, top_speed=300,
                     mass=1600)
    v8 = [E("V8", "V8", "Cross-plane", "Deep American burble, uneven exhaust pulses. Firing 1-8-4-3-6-5-7-2.",
            [1, 8, 4, 3, 6, 5, 7, 2], [[1, 3, 5, 7], [2, 4, 6, 8]], [-45, 45],
            body_freq=72, body_decay=0.036, noise_band=(180, 1800), noise_gain=0.55,
            drive=1.9, hum_gain=0.20, hum_weights=(1, .7, .35, .12), sub_gain=0.55,
            bank_amps=(1.0, 0.82), **v8_common),
          E("V8", "V8", "Flat-plane", "Higher-pitched, raspy exotic V8. Firing 1-8-4-3-6-5-7-2.",
            [1, 8, 4, 3, 6, 5, 7, 2], [[1, 3, 5, 7], [2, 4, 6, 8]], [-45, 45],
            body_freq=112, body_decay=0.021, noise_band=(500, 5000), noise_gain=0.6,
            drive=1.5, hum_gain=0.22, hum_weights=(1, .5, .45, .3), sub_gain=0.25,
            **v8_common)]

    v10 = [E("V10", "V10", "", "Screaming 90\u00b0 V10 with uneven firing. Firing 1-6-5-10-2-7-3-8-4-9.",
             [1, 6, 5, 10, 2, 7, 3, 8, 4, 9], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [-45, 45],
             520, 7000, 8500, 900, 320, mass=1550,
             body_freq=120, body_decay=0.018, noise_band=(900, 7000), noise_gain=0.75,
             drive=1.5, hum_gain=0.30, hum_weights=(1, .7, .55, .4), sub_gain=0.2,
             interval_pattern=(1.18, 0.82))]

    v12 = [E("V12", "V12", "", "Silky, howling 60\u00b0 V12. Firing 1-12-4-9-2-11-6-7-3-10-5-8.",
             [1, 12, 4, 9, 2, 11, 6, 7, 3, 10, 5, 8],
             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [-30, 30],
             700, 6500, 8500, 900, 340, mass=1750,
             body_freq=100, body_decay=0.020, noise_band=(700, 5000), noise_gain=0.35,
             drive=1.2, hum_gain=0.55, hum_weights=(1, .8, .6, .45), sub_gain=0.22)]

    w16 = [E("W16", "W16", "", "Quad-bank 8.0 L monster. Firing 1-14-9-4-7-12-15-6-13-8-3-10-11-2-5-16.",
             [1, 14, 9, 4, 7, 12, 15, 6, 13, 8, 3, 10, 11, 2, 5, 16],
             [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
             [-40, -13, 13, 40],
             1500, 4500, 7000, 750, 400, mass=1950, cda=0.60, traction=1.1,
             body_freq=55, body_decay=0.048, noise_band=(120, 1400), noise_gain=0.6,
             drive=2.1, hum_gain=0.25, hum_weights=(1, .65, .4, .2), sub_gain=0.85,
             interval_pattern=(1.07, 0.93), bank_amps=(1.0, 0.8, 0.92, 0.72))]

    flat6 = [E("Flat-6", "Flat-6", "", "Boxer six, metallic rasp. Firing 1-6-2-4-3-5.",
               [1, 6, 2, 4, 3, 5], [[1, 2, 3], [4, 5, 6]], [-90, 90],
               450, 6200, 8000, 900, 290, mass=1400,
               body_freq=130, body_decay=0.020, noise_band=(450, 3500), noise_gain=0.55,
               drive=1.6, hum_gain=0.22, hum_weights=(1, .6, .4, .25), sub_gain=0.22,
               bank_amps=(1.0, 0.88))]
    return [i4, v6, v8, v10, v12, w16, flat6]


# --------------------------------------------------------------------------- #
# Procedural audio
# --------------------------------------------------------------------------- #
class AudioSynthesizer:
    """
    Streams procedurally generated audio through a pygame Channel.

    Shared (written by the main thread): rpm, throttle, running, fuel_cut,
    boost, induction.  A worker thread renders CHUNK-sample blocks and queues
    them as Sounds.  One-shot events (BOV dump, gear crunch, backfire bang,
    stall sputter) are posted from the main thread and mixed by the worker.
    """
    BAND_MULT = (0.75, 1.1, 1.6)     # burst pitch/brightness at low / mid / high RPM

    def __init__(self):
        self.enabled = False
        self.muted = False
        self.master = 0.8
        self.rpm = 0.0
        self.throttle = 0.0
        self.running = False
        self.fuel_cut = False
        self.induction = "NA"
        self.boost = 0.0
        self.cfg = None
        self.scope = np.zeros(CHUNK, dtype=np.float32)
        self.rng = np.random.default_rng(7)
        self._lock = threading.Lock()
        self._elock = threading.Lock()
        self.events = []
        self._alive = False
        self._thread = None
        self.sr = 44100
        self.channels = 1
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 1, 512, allowedchanges=0)
            freq, _fmt, ch = pygame.mixer.get_init()
            self.sr, self.channels = int(freq), int(ch)
            self.enabled = True
        except pygame.error as exc:
            print(f"[audio] disabled: {exc}")
        if self.enabled:
            self._build_beds()
        self._reset_state()

    # ----- one-time noise beds / one-shot samples ---------------------------- #
    def _bandpass(self, x, lo, hi):
        n = len(x)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        mask = (1.0 / (1.0 + (f / max(hi, 1.0)) ** 4)) * \
               (1.0 / (1.0 + (lo / np.maximum(f, 1.0)) ** 4))
        y = np.fft.irfft(X * mask, n)
        m = np.max(np.abs(y))
        return (y / m if m > 0 else y).astype(np.float32)

    def _build_beds(self):
        sr = self.sr
        rng = np.random.default_rng(99)
        self.rumble = self._bandpass(rng.standard_normal(3 * sr), 1.0, 160.0)
        self.air = self._bandpass(rng.standard_normal(3 * sr), 250.0, 2600.0)
        lp = int(0.05 * sr)
        t = np.arange(lp) / sr
        pop = self._bandpass(rng.standard_normal(lp), 300.0, 3500.0)
        self.pop = (pop * np.exp(-t / 0.012) * (1 - np.exp(-t / 0.0008))).astype(np.float32)

        # ---- blow-off valve dump ("pshhh", slight flutter) ----
        n = int(0.45 * sr)
        t = np.arange(n) / sr
        hiss = self._bandpass(rng.standard_normal(n), 1100.0, 9500.0)
        env = np.exp(-t / 0.18) * (1 - np.exp(-t / 0.003))
        flut = 0.82 + 0.18 * np.sin(TAU * 26.0 * t)
        bov = (hiss * env * flut * 0.85).astype(np.float32)

        # ---- gear crunch (grit + wavering squeal, grinding AM) ----
        n = int(0.34 * sr)
        t = np.arange(n) / sr
        grit = self._bandpass(rng.standard_normal(n), 480.0, 4200.0)
        f = 950.0 + 320.0 * np.sin(TAU * 5.5 * t)
        sq = np.sin(TAU * np.cumsum(f) / sr)
        am = 0.5 + 0.5 * np.sin(TAU * 64.0 * t)
        crunch = ((grit * 0.95 + sq * 0.30) * am * np.exp(-t / 0.30) *
                  (1 - np.exp(-t / 0.004))).astype(np.float32)

        # ---- exhaust backfire bang (pitch-dropping boom + crack) ----
        n = int(0.30 * sr)
        t = np.arange(n) / sr
        boom = np.sin(TAU * np.cumsum(72.0 + 48.0 * np.exp(-t / 0.015)) / sr)
        crack = self._bandpass(rng.standard_normal(n), 400.0, 7000.0)
        tail = self._bandpass(rng.standard_normal(n), 110.0, 1900.0)
        bang = (np.tanh(2.6 * (boom * 1.5 * np.exp(-t / 0.07) +
                               crack * 1.6 * np.exp(-t / 0.012) +
                               tail * 0.6 * np.exp(-t / 0.10))) *
                (1 - np.exp(-t / 0.0004))).astype(np.float32)

        # ---- engine stall sputter (dying thumps) ----
        n = int(0.7 * sr)
        p = self.pop
        t5 = np.arange(len(p)) / sr
        thump = (np.sin(TAU * np.cumsum(95.0 + 55.0 * np.exp(-t5 / 0.012)) / sr) *
                 np.exp(-t5 / 0.035) * (1 - np.exp(-t5 / 0.001)))
        sput = np.zeros(n, dtype=np.float32)
        for delay, g in ((0.0, 1.0), (0.14, 0.75), (0.26, 0.55), (0.36, 0.40)):
            o = int(delay * sr)
            sput[o:o + len(p)] += (thump * 1.2 + p * 0.5) * g
        sput = sput.astype(np.float32)

        self.samples = {"pop": self.pop, "bov": bov, "crunch": crunch,
                        "bang": bang, "sputter": sput}

    def _reset_state(self):
        self.L = int(0.06 * self.sr)
        self.tail = np.zeros(self.L, dtype=np.float32)
        self.next_pulse = 0.0
        self.slot = 0
        self.hum_phase = 0.0
        self.prev_hz = 0.0
        self.whine_phase = 0.0
        self.prev_whz = 0.0
        self.turbo_phase = 0.0
        self.prev_thz = 0.0
        self.gains = {}
        self.idx = {"rumble": 0, "air": 0, "air2": 21000, "gur": 40000}
        self.T = None
        self._one = np.zeros(0, dtype=np.float32)

    # ----- per-engine pulse templates --------------------------------------- #
    def set_engine(self, cfg: EngineConfig):
        with self._lock:
            self.cfg = cfg
            self._reset_state()
            with self._elock:
                self.events = []
            if self.enabled:
                self._build_templates()

    def _build_templates(self):
        """T[band*2+load, cylinder, sample]: sine burst + band-passed noise + overdrive."""
        cfg, sr, L, n = self.cfg, self.sr, self.L, self.cfg.cylinders
        t = np.arange(L) / sr
        atk = 1.0 - np.exp(-t / 0.0012)
        rng = np.random.default_rng(1234)
        T = np.zeros((3, 2, n, L), dtype=np.float32)
        for b, mult in enumerate(self.BAND_MULT):
            for c in range(n):
                detune = 1.0 + 0.04 * math.sin(c * 2.39 + 1.3)     # each cylinder differs
                f0 = cfg.body_freq * mult * detune
                inst = f0 * (1.0 + 0.7 * np.exp(-t / 0.012))        # pitch drop
                ph = TAU * np.cumsum(inst) / sr
                env = np.exp(-t / (cfg.body_decay / mult ** 0.6)) * atk
                body = np.sin(ph) * env + 0.4 * np.sin(2 * ph + 0.5) * env ** 1.5
                lo, hi = cfg.noise_band
                nz = self._bandpass(rng.standard_normal(L), lo * mult ** 0.5,
                                    min(hi * mult ** 0.5, sr * 0.42))
                nz = nz * np.exp(-t / 0.009) * atk
                for li, load in enumerate((0.0, 1.0)):
                    pulse = (0.55 + 0.55 * load) * body + cfg.noise_gain * (1.0 - 0.35 * load) * nz
                    d = 1.0 + (cfg.drive - 1.0) * (0.25 + 0.75 * load)   # subtle harmonic overdrive
                    T[b, li, c] = np.tanh(d * pulse) * (0.6 + 0.4 * load)
        self.T = T.reshape(6, n, L)

    # ----- one-shot event queue ---------------------------------------------- #
    def post_event(self, name, strength=1.0):
        """Main thread: queue a one-shot sample ('bov'/'crunch'/'bang'/'sputter')."""
        if not self.enabled:
            return
        with self._elock:
            self.events.append((name, float(strength)))

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

    def _render(self) -> np.ndarray:
        N, sr = CHUNK, self.sr
        cfg = self.cfg
        if cfg is None or self.T is None:
            return np.zeros(N, dtype=np.float32)
        rpm = max(0.0, float(self.rpm))
        thr = clamp(float(self.throttle), 0.0, 1.0)
        running, cut = bool(self.running), bool(self.fuel_cut)
        boost = clamp(float(self.boost), 0.0, 2.0)
        ind = self.induction
        fi = boost if ind in ("TURBO", "SUPER") else 0.0
        n, L = cfg.cylinders, self.L
        x = clamp(rpm / cfg.redline, 0.0, 1.1)
        pulse_hz = cfg.pulse_hz(rpm)

        # engine-braking overrun amount (throttle closed while revving)
        ov = 0.0
        if running and not cut:
            ov = (1.0 - thr) * clamp((rpm - cfg.idle * 1.5) / (cfg.redline * 0.45), 0.0, 1.0)

        if running:
            level = (0.34 + 0.66 * thr) * (1.0 - 0.45 * ov) * (0.75 + 0.25 * x)
            if cut:
                level = 0.12                      # rev-limiter: fuel cut, compression thumps only
            level *= 1.0 + 0.22 * fi              # forced induction is louder
        else:
            level = 0.16 * clamp(rpm / 250.0, 0.0, 1.0)     # cranking / spin-down thumps

        # blend burst templates by RPM band and load (boost fattens the load blend)
        p = min(x, 1.0) * 2.0
        bi = min(int(p), 1)
        fr = p - bi
        wb = np.zeros(3, dtype=np.float32)
        wb[bi], wb[bi + 1] = 1.0 - fr, fr
        load = clamp((0.12 + 0.88 * thr) * (1.0 - 0.6 * ov) + 0.22 * fi * thr, 0.0, 1.0)
        wl = np.array([1.0 - load, load], dtype=np.float32)
        tpl = np.tensordot(np.outer(wb, wl).ravel(), self.T, axes=1)      # (n, L)

        # overlap-add firing pulses at exact firing instants
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

        # small overrun crackle / pops
        if running and ov > 0.3 and rpm > 2000 and self.rng.random() < 0.10 * ov:
            o = int(self.rng.integers(0, N))
            buf[o:o + len(self.pop)] += self.pop * float(self.rng.uniform(0.5, 1.0)) * 0.9

        out = buf[:N].copy()
        self.tail = buf[N:].copy()

        # phase-continuous harmonics locked to firing frequency (pitch scales with RPM)
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

        # bass rumble bed (throttle boosts bass) + intake/exhaust roar
        rg = cfg.sub_gain * (0.12 + 0.88 * thr) * (0.35 + 0.65 * x) if running else 0.0
        out += self._loop("rumble", self.rumble, N) * self._ramp("rumble", rg, N)
        ag = cfg.noise_gain * 0.30 * thr * x ** 1.3 * (1.0 + 0.9 * fi) if running else 0.0
        out += self._loop("air", self.air, N) * self._ramp("air", ag, N)

        # deceleration: pulsing low gurgle (engine-braking overrun)
        gg = 0.8 * ov * (0.4 + cfg.sub_gain)
        mod = (0.55 + 0.45 * np.sin(ph * 0.5)).astype(np.float32)
        out += self._loop("gur", self.rumble, N) * mod * self._ramp("gur", gg, N)

        # ---- supercharger mechanical whine (rotor harmonics, phase-continuous) --
        if running and ind == "SUPER":
            whz = 230.0 + rpm * 0.34
            fsw = np.linspace(self.prev_whz, whz, N)
            phw = self.whine_phase + np.cumsum(fsw) * (TAU / sr)
            self.whine_phase = float(phw[-1] % TAU)
            self.prev_whz = whz
            w = np.sin(phw) + 0.5 * np.sin(2.0 * phw + 0.3) + 0.22 * np.sin(3.0 * phw + 0.7)
            wg = (0.045 + 0.15 * thr) * (0.35 + 0.65 * x)
            out += (w * self._ramp("whine", wg, N)).astype(np.float32)
        else:
            self._ramp("whine", 0.0, N)

        # ---- turbo whistle + spool hiss ----
        tg = 0.0
        sg = 0.0
        if running and ind == "TURBO":
            thz = 900.0 + rpm * 0.6
            fst = np.linspace(self.prev_thz, thz, N)
            pht = self.turbo_phase + np.cumsum(fst) * (TAU / sr)
            self.turbo_phase = float(pht[-1] % TAU)
            self.prev_thz = thz
            w = np.sin(pht) + 0.35 * np.sin(2.0 * pht + 0.5)
            tg = 0.012 + 0.05 * clamp(boost / 1.4, 0.0, 1.0)
            out += (w * self._ramp("whistle", tg, N)).astype(np.float32)
            sg = 0.07 * clamp(boost / 1.4, 0.0, 1.0)
        else:
            self._ramp("whistle", 0.0, N)
        out += self._loop("air2", self.air, N) * self._ramp("spool", sg, N)

        # ---- one-shot events (BOV dump / gear crunch / backfire / sputter) ----
        with self._elock:
            evs, self.events = self.events, []
        for name, g in evs:
            smp = self.samples.get(name)
            if smp is None:
                continue
            off = int(self.rng.integers(0, max(1, N - 64)))
            need = off + len(smp)
            if len(self._one) < need:
                self._one = np.concatenate(
                    (self._one, np.zeros(need - len(self._one), dtype=np.float32)))
            self._one[off:off + len(smp)] += smp * np.float32(g)
        if len(self._one) > 0:
            k = min(len(self._one), N)
            out[:k] += self._one[:k]
            self._one = self._one[k:].copy()

        mix = np.tanh(out * 1.6) * 0.9
        self.scope = mix
        return mix

    def _make_sound(self):
        with self._lock:
            mix = self._render()
        vol = 0.0 if self.muted else self.master
        pcm = (mix * 32767.0 * vol).astype(np.int16)
        if self.channels == 2:
            pcm = np.ascontiguousarray(np.column_stack((pcm, pcm)))
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
# Transmission
# --------------------------------------------------------------------------- #
class Transmission:
    """7-speed gearbox: automatic (P/N/D) or manual sequential (E/Q + clutch)."""
    RATIOS = (3.62, 2.38, 1.72, 1.34, 1.08, 0.88, 0.73)
    AUTO_SHIFT_TIME = 0.30
    MANUAL_SHIFT_TIME = 0.20

    def __init__(self):
        self.mode = "P"
        self.manual = False
        self.revmatch_assist = True
        self.gear = 1
        self.target = 1
        self.shift_timer = 0.0
        self.shift_dir = 0
        self.cooldown = 0.0
        self.final = 3.3

    def configure(self, cfg: EngineConfig, wheel_r: float):
        """Final drive so top gear reaches ~97 % of redline at the car's top speed."""
        w = cfg.redline * 0.97 * TAU / 60.0
        self.final = (w * wheel_r) / ((cfg.top_speed / 3.6) * self.RATIOS[-1])

    def reset(self, mode="P"):
        self.mode, self.gear, self.target = mode, 1, 1
        self.shift_timer, self.shift_dir, self.cooldown = 0.0, 0, 0.0

    @property
    def shifting(self):
        return self.shift_timer > 0.0

    def wheel_rpm(self, speed, wheel_r, gear=None):
        g = self.gear if gear is None else gear
        return speed / wheel_r * self.final * self.RATIOS[g - 1] * 60.0 / TAU

    def total_ratio(self):
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

    def begin_shift(self, new, d):
        """Manual shift (clutch was verified by the caller)."""
        self.target = new
        self.shift_dir = d
        self.shift_timer = self.MANUAL_SHIFT_TIME
        self.cooldown = 0.30

    def display(self):
        if self.mode in ("P", "N"):
            return self.mode
        return str(self.target if self.shifting else self.gear)

    def update(self, dt, eng_rpm, speed, throttle, cfg, wheel_r):
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.shift_timer > 0.0:
            self.shift_timer -= dt
            if self.shift_timer <= 0.0:
                self.gear, self.shift_dir = self.target, 0
            return
        if self.mode != "D":
            self.gear = self.target = 1
            return
        if self.manual:                 # driver does the shifting
            return
        if self.cooldown > 0.0:
            return
        wrpm = self.wheel_rpm(speed, wheel_r)
        up = (0.36 + 0.58 * throttle) * cfg.redline
        down = (0.19 + 0.30 * throttle) * cfg.redline
        if self.gear < 7 and eng_rpm > up and wrpm > 0.6 * up:
            self._shift(+1)
        elif self.gear > 1 and wrpm < down:
            self._shift(-1)

    def _shift(self, d):
        self.target = self.gear + d
        self.shift_dir = d
        self.shift_timer = self.AUTO_SHIFT_TIME
        self.cooldown = 1.0


# --------------------------------------------------------------------------- #
# Vehicle physics (engine + clutch + chassis + fuel + thermals + induction)
# --------------------------------------------------------------------------- #
class Vehicle:
    WHEEL_R = 0.33
    K_CLUTCH = 80.0        # Nm per rad/s of slip
    FUEL_CAP = 60.0        # litres
    FUEL_DENSITY = 0.75    # kg/L
    STALL_FRAC = 0.42      # stall below this fraction of idle rpm
    BOOST_GAIN = 0.6       # extra torque per bar of boost

    def __init__(self, tx: Transmission):
        self.tx = tx
        self.cfg = None
        self.omega = 0.0           # crank speed, rad/s
        self.speed = 0.0           # m/s
        self.throttle_cmd = self.throttle = self.thr_eff = 0.0
        self.brake_cmd = self.brake = 0.0
        self.clutch_cmd = self.clutch_pos = 0.0
        self.clutch_eng = 0.0       # 0 = disengaged, 1 = fully engaged
        self.clutch_disp = 0.0      # engagement shown on the HUD
        self.running = self.cranking = self.fuel_cut = False
        self.crank_t = 0.0
        self.run_t = 99.0
        # fuel / AFR
        self.fuel = self.FUEL_CAP
        self.fuel_scale = 1.0       # demo fast-drain multiplier
        self.fuel_lph = 0.0
        self._lph_real = 0.0
        self.afr = 0.0
        self.decel_cut = False
        # thermals
        self.coolant = 20.0
        self.warm = 0.0
        self.oil_p = 0.0
        self.engine_failed = False
        self.fail_reason = ""
        self.overheat_t = 0.0
        self.overrev_t = 0.0
        # induction
        self.induction = "NA"
        self.boost = 0.0
        self.map_bar = 0.0
        self.prev_thr = 0.0
        self.bov_flash = 0.0
        self.bov_cd = 0.0
        # gearbox helpers
        self.blip_t = 0.0
        self.blip_target = 0.0
        self.stall_flash = 0.0
        self.stall_ign = 0.0
        self.bump_t = 0.0
        self.trip = 0.0
        self.pending = []           # ("audio", (name, gain)) / ("toast", (text, kind))

    @property
    def rpm(self):
        return self.omega * 60.0 / TAU

    # ----- setup ------------------------------------------------------------- #
    def set_engine(self, cfg: EngineConfig):
        keep = self.running
        self.cfg = cfg
        self.speed = 0.0
        self.tx.configure(cfg, self.WHEEL_R)
        self.tx.reset("N" if keep else "P")
        self.omega = cfg.idle * TAU / 60.0 if keep else 0.0
        self.cranking = self.fuel_cut = False
        self.run_t = 99.0
        self.fuel = self.FUEL_CAP
        self.fuel_lph = self._lph_real = 0.0
        self.afr = 0.0
        self.decel_cut = False
        self.coolant = 20.0
        self.warm = 0.0
        self.oil_p = 0.0
        self.engine_failed = False
        self.fail_reason = ""
        self.overheat_t = self.overrev_t = 0.0
        self.boost = 0.0
        self.map_bar = 0.0
        self.blip_t = 0.0
        self.blip_target = 0.0
        self.stall_flash = 0.0
        self.stall_ign = 0.0
        self.bump_t = 0.0
        self.trip = 0.0
        self.clutch_pos = self.clutch_eng = self.clutch_disp = 0.0
        self.pending = []

    # ----- driver commands --------------------------------------------------- #
    def toggle(self):
        if self.engine_failed:
            self.pending.append(
                ("toast", (f"ENGINE FAILED ({self.fail_reason}) - let it cool", "bad")))
            return
        if self.running or self.cranking:
            self.running = self.cranking = False
            self.stall_ign = 0.0
        else:
            if self.fuel <= 0.0:
                self.pending.append(("toast", ("tank is dry - press R to refuel", "warn")))
            self.cranking, self.crank_t = True, 0.0

    def request_drive(self):
        if self.running:
            self.tx.engage_drive(self.speed, self.cfg, self.WHEEL_R)

    def request_neutral(self):
        if self.tx.mode != "P":
            self.tx.mode = "N"

    def refuel(self):
        self.fuel = self.FUEL_CAP
        self.pending.append(("toast", (f"refuelled {self.FUEL_CAP:.0f} L", "good")))

    def cycle_induction(self):
        order = ("NA", "TURBO", "SUPER")
        self.induction = order[(order.index(self.induction) + 1) % len(order)]
        self.boost = 0.0
        label = {"NA": "naturally aspirated",
                 "TURBO": "turbocharger (BOV armed)",
                 "SUPER": "supercharger (whine)"}[self.induction]
        self.pending.append(("toast", (f"induction: {label}", "info")))

    def shift(self, d):
        """Manual up/down shift (E / Q). Crunches without the clutch."""
        tx = self.tx
        if not tx.manual or tx.shifting or tx.cooldown > 0.0:
            return
        clutch_ok = self.clutch_eng < 0.25
        if tx.mode == "N":
            if d > 0:                                   # engage 1st from neutral
                if clutch_ok:
                    tx.mode, tx.gear, tx.target = "D", 1, 1
                else:
                    self._crunch()
            return
        if tx.mode != "D":
            return
        new = tx.gear + d
        if new < 1:                                     # down from 1st -> neutral
            if clutch_ok:
                tx.mode = "N"
            else:
                self._crunch()
            return
        if new > len(tx.RATIOS):
            return
        if not clutch_ok:
            self._crunch()
            return
        tx.begin_shift(new, d)
        if d < 0 and tx.revmatch_assist and self.running:
            target = tx.wheel_rpm(self.speed, self.WHEEL_R, new) * 1.03
            if target > self.rpm + 100.0:               # auto rev-match blip
                self.blip_target = target
                self.blip_t = 0.5

    def _crunch(self):
        self.pending += [("audio", ("crunch", 1.0)),
                         ("toast", ("GEAR CRUNCH - hold CLUTCH (Shift/C) while shifting", "bad"))]

    def _stall(self, reason):
        self.running = False
        self.stall_flash = 2.0
        self.stall_ign = 8.0            # ignition still on: bump start possible
        self.pending += [("audio", ("sputter", 0.9)),
                         ("toast", (f"ENGINE STALLED - {reason}", "bad"))]

    def fail_engine(self, reason):
        self.running = False
        self.cranking = False
        self.engine_failed = True
        self.fail_reason = reason
        self.stall_flash = 3.0
        self.blip_t = 0.0
        self.pending += [("audio", ("crunch", 0.9)), ("audio", ("bang", 0.5)),
                         ("toast", (f"ENGINE FAILURE - {reason}", "bad"))]

    def pop_events(self):
        ev = self.pending
        self.pending = []
        return ev

    # ----- engine torque ----------------------------------------------------- #
    def power_mult(self):
        """Cold engines are weak, overheating engines derate, boost adds power."""
        cold = 0.78 + 0.22 * self.warm
        hot = 1.0 - 0.45 * clamp((self.coolant - 105.0) / 17.0, 0.0, 1.0)
        fi = 1.0 + self.boost * self.BOOST_GAIN
        return cold * hot * fi

    def engine_torque(self, rpm, thr):
        c = self.cfg
        x = rpm / c.redline
        fr = c.fric * (0.45 + 0.9 * x)                     # friction
        if self.cranking:
            return (c.fric * 3.5 if rpm < 260 else 0.0) - fr
        if not self.running:
            return -fr - c.fric * 0.6                      # compression drag on spin-down
        if self.fuel_cut or self.fuel <= 0.0 or self.engine_failed:
            return -fr - c.fric * 1.3 * x                  # no fuel: pure engine braking
        pump = c.fric * 1.3 * x * (1.0 - thr)              # throttle-closed pumping loss
        curve = max(0.42, 1.0 - 1.15 * ((rpm - c.peak_rpm) / c.redline) ** 2)
        T = c.peak_torque * curve * thr * self.power_mult()
        if thr < 0.25:                                     # idle controller (+ start-up flare)
            target = c.idle * (1.0 + 0.8 * max(0.0, 1.0 - self.run_t / 1.8))
            if rpm < target * 1.35:
                kp = 0.0006 * c.peak_torque
                T += (1.0 - thr) * (fr + c.fric * 1.3 * x +
                                    clamp(kp * (target - rpm), -0.05 * c.peak_torque,
                                          0.3 * c.peak_torque))
        return T - fr - pump

    # ----- fuel / AFR -------------------------------------------------------- #
    def step_fuel(self, dt):
        c = self.cfg
        thr = clamp(self.thr_eff, 0.0, 1.0)
        rpm = self.rpm
        self.decel_cut = (self.running and thr < 0.05 and rpm > c.idle * 1.35
                          and self.fuel > 0.0)
        self.afr = 0.0
        self.fuel_lph = 0.0
        self._lph_real = 0.0
        if self.running and not self.fuel_cut and self.fuel > 0.0 and not self.decel_cut:
            x = clamp(rpm / c.redline, 0.0, 1.05)
            ve = 0.88 - 0.18 * x                                   # volumetric efficiency
            air = (rpm / 60.0) * (c.disp / 2000.0) * ve * 1.2 * (1.0 + self.boost)  # kg/s
            afr = 12.6 + 2.6 * (1.0 - thr) ** 0.8                  # rich -> lean with load
            afr -= 1.4 * (1.0 - self.warm)                         # cold-start enrichment
            afr = clamp(afr, 10.5, 16.5)
            fuel_kgs = air / afr
            self.afr = afr
            self._lph_real = fuel_kgs * 3600.0 / self.FUEL_DENSITY
            self.fuel_lph = self._lph_real * self.fuel_scale
            self.fuel = max(0.0, self.fuel - self.fuel_lph / 3600.0 * dt)

    # ----- thermals ---------------------------------------------------------- #
    def step_thermal(self, dt):
        heat = 0.0
        if self.running:
            heat = 0.16 + 0.06 * self._lph_real          # burn heat (unscaled)
        fan = 0.55 if (self.running and self.coolant > 92.0) else 0.0
        sf = 0.55 + min(1.6, self.speed * 3.6 / 90.0) + fan   # ram air + fan
        if self.coolant > 85.0:
            cool = 0.106 * sf * (self.coolant - 85.0)   # thermostat open
        else:
            cool = 0.004 * (self.coolant - 20.0)        # natural cool-down
        if not self.running:
            cool = min(cool, 0.35)
        self.coolant = clamp(self.coolant + (heat - cool) * dt, 15.0, 130.0)
        self.warm = smoothstep((self.coolant - 20.0) / 70.0)
        # oil pressure (rpm-driven, thinner when hot)
        tgt = (0.9 + self.rpm / 1000.0 * 0.6) * (1.05 - 0.15 * self.warm) if self.running else 0.0
        self.oil_p += (tgt - self.oil_p) * min(1.0, dt * 8.0)

    # ----- induction --------------------------------------------------------- #
    def update_induction(self, dt):
        c = self.cfg
        thr = clamp(self.thr_eff, 0.0, 1.0)
        x = clamp(self.rpm / c.redline, 0.0, 1.0)
        if not self.running:
            self.boost *= math.exp(-5.0 * dt)
        elif self.induction == "TURBO":
            spool = clamp((x - 0.18) / 0.45, 0.0, 1.0)          # needs rpm to spool
            target = 1.4 * (thr ** 1.4) * spool
            tau = 0.55 - 0.38 * spool                           # lag shrinks with rpm
            if target < self.boost:
                tau = 0.08                                      # wastegate / dump falls fast
            if (thr < 0.15 and self.prev_thr >= 0.30 and self.boost > 0.25
                    and self.bov_cd <= 0.0):
                self.pending.append(("audio", ("bov", clamp(self.boost / 1.1, 0.6, 1.3))))
                self.bov_flash = 1.0
                self.bov_cd = 0.5
                self.boost *= 0.2                               # BOV vents the charge
            self.boost += (target - self.boost) * clamp(dt / max(tau, 0.02), 0.0, 1.0)
        elif self.induction == "SUPER":
            target = 0.85 * clamp(x / 0.95, 0.0, 1.0) * (0.30 + 0.70 * thr)
            self.boost += (target - self.boost) * clamp(dt / 0.08, 0.0, 1.0)
        else:
            self.boost *= math.exp(-6.0 * dt)
        self.boost = clamp(self.boost, 0.0, 1.6)
        self.prev_thr = thr
        # manifold pressure for the gauge (vacuum at closed throttle)
        vac = 0.80 * (1.0 - thr) * (0.25 + 0.75 * x) if self.running else 0.0
        self.map_bar = self.boost - vac

    # ----- main step --------------------------------------------------------- #
    def step(self, dt: float):
        c, tx, R = self.cfg, self.tx, self.WHEEL_R
        self.throttle += clamp(self.throttle_cmd - self.throttle, -4.0 * dt, 2.8 * dt)
        self.brake += clamp(self.brake_cmd - self.brake, -6.0 * dt, 7.0 * dt)
        self.clutch_pos += clamp(self.clutch_cmd - self.clutch_pos, -9.0 * dt, 7.0 * dt)
        self.clutch_eng = smoothstep((0.62 - self.clutch_pos) / 0.45)   # bite zone

        self.stall_flash = max(0.0, self.stall_flash - dt)
        self.stall_ign = max(0.0, self.stall_ign - dt)
        self.bov_flash = max(0.0, self.bov_flash - 3.0 * dt)
        self.bov_cd = max(0.0, self.bov_cd - dt)

        # ------- starter / ignition state machine -------
        if self.cranking:
            self.crank_t += dt
            if self.fuel <= 0.0 or self.engine_failed:
                if self.crank_t >= 0.8:
                    self.cranking = False
                    msg = ("NO FUEL - press R to refuel" if self.fuel <= 0.0
                           else "ENGINE FAILED - let it cool")
                    self.pending.append(("toast", (msg, "warn")))
            elif self.crank_t >= 0.9 and self.rpm > 120.0:
                self.cranking, self.running, self.run_t = False, True, 0.0
                self.omega = max(self.omega, c.idle * 1.15 * TAU / 60.0)   # start-up flare
            elif self.crank_t >= 3.0:
                self.cranking = False
                if self.rpm <= 120.0:
                    self.pending.append(
                        ("toast", ("CLUTCH ENGAGED - hold Shift/C to crank freely", "warn")))
        elif self.running:
            self.run_t += dt

        if not self.running and not self.cranking:
            if tx.mode == "D" and not tx.manual:
                tx.mode = "N"
            if tx.mode == "N" and self.speed < 0.3 and not tx.manual:
                tx.mode = "P"
        if self.running and tx.mode == "P":
            tx.mode = "N"
        if self.engine_failed and self.coolant < 95.0:
            self.engine_failed = False
            self.pending.append(("toast", ("engine cooled - press SPACE to restart", "good")))

        # ------- rev limiter -------
        if self.running:
            if self.rpm >= c.redline:
                self.fuel_cut = True
            elif self.rpm < c.redline - 250.0:
                self.fuel_cut = False
        else:
            self.fuel_cut = False

        tx.update(dt, self.rpm, self.speed, self.throttle, c, R)

        # ------- effective throttle (auto-shift blip / rev-match blip) -------
        thr = self.throttle
        if tx.shifting and not tx.manual:
            thr = max(thr, 0.55) if tx.shift_dir < 0 else thr * 0.25
        if self.blip_t > 0.0:
            self.blip_t -= dt
            if self.blip_target > self.rpm + 80.0:
                thr = max(thr, 0.85)             # blip up toward the target rpm
            elif self.blip_target < self.rpm - 200.0:
                thr = min(thr, 0.20)
        self.thr_eff = clamp(thr, 0.0, 1.0)

        # ------- fuel / AFR / thermals / induction -------
        self.step_thermal(dt)
        self.step_fuel(dt)
        self.update_induction(dt)

        # ------- drivetrain physics -------
        n = max(8, int(math.ceil(dt * 960)))
        h = dt / n
        G = tx.total_ratio()
        m = c.mass
        fb = 14000.0 * self.brake + (16000.0 if tx.mode == "P" else 0.0)
        fi = 1.0 + self.boost * self.BOOST_GAIN
        eng_last = 0.0
        for _ in range(n):
            rpm = self.omega * 60.0 / TAU
            Te = self.engine_torque(rpm, thr)
            Tc = 0.0
            if G > 0.0:
                ww = self.speed * G / R
                if tx.manual:
                    eng = self.clutch_eng                       # real clutch pedal
                else:
                    rw = ww * 60.0 / TAU                        # torque-converter style
                    eng = 1.0 if rw >= 1300.0 else \
                        max(smoothstep((rpm - (c.idle + 100.0)) / 900.0), rw / 1300.0)
                eng_last = eng
                cap = min(c.peak_torque * 2.4, c.traction * m * 9.81 * R / G) * eng * fi
                Tc = clamp(self.K_CLUTCH * (self.omega - ww), -cap, cap)
                # bump start: rolling in gear with the clutch out spins the engine up
                if (not self.running and not self.cranking and not self.engine_failed
                        and self.fuel > 0.0 and self.stall_ign > 0.0 and eng > 0.4
                        and rpm > c.idle * 0.95):
                    self.bump_t += h
                    if self.bump_t > 0.25:
                        self.running, self.run_t, self.bump_t = True, 0.0, 0.0
                        self.stall_ign = 0.0
                        self.pending.append(("toast", ("BUMP STARTED", "good")))
                else:
                    self.bump_t = 0.0
            self.omega = max(0.0, self.omega + (Te - Tc) / c.inertia * h)
            F = Tc * G / R if G > 0.0 else 0.0
            drag = 0.5 * 1.2 * c.cda * self.speed ** 2 + \
                (0.012 * m * 9.81 if self.speed > 0.01 else 0.0)
            self.speed += F / m * h
            self.speed = max(0.0, self.speed - (drag + fb) / m * h)
        self.clutch_disp = eng_last if G > 0.0 else 0.0

        # ------- stall / failure checks -------
        if self.running:
            if self.fuel <= 0.0:
                self._stall("OUT OF FUEL - press R")
            elif self.rpm < c.idle * self.STALL_FRAC:
                self._stall("clutch dumped under load")
        if self.rpm > c.redline * 1.1:            # money shift over-rev
            self.overrev_t += dt
            if self.overrev_t > 1.0 and not self.engine_failed:
                self.fail_engine("OVER-REV (money shift)")
        else:
            self.overrev_t = max(0.0, self.overrev_t - 2.0 * dt)
        if self.coolant >= 122.0:                 # sustained high-rpm overheating
            self.overheat_t += dt
            if self.overheat_t > 2.0 and self.running and not self.engine_failed:
                self.fail_engine("OVERHEAT")
        else:
            self.overheat_t = max(0.0, self.overheat_t - dt)

        self.trip += self.speed * dt


# --------------------------------------------------------------------------- #
# UI widgets
# --------------------------------------------------------------------------- #
class Button:
    def __init__(self, rect, label, on_click=None, hold=None, active=None, enabled=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.hold = hold
        self.active = active
        self.enabled = enabled

    def draw(self, surf, font, mouse, held):
        enabled = self.enabled() if self.enabled else True
        active = (self.active() if self.active else False) or \
                 (self.hold is not None and held == self.hold)
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
    W, H = 1440, 900
    BG, PANEL, BORDER = (13, 15, 21), (21, 25, 35), (52, 60, 82)
    TEXT, DIM = (222, 228, 242), (128, 138, 160)
    ORANGE, RED, GREEN = (255, 140, 30), (230, 45, 45), (70, 205, 125)

    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption("Procedural Engine Simulator - manual box, fuel, thermals, boost")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        pick = "dejavusans,arial,helvetica"
        self.f_title = pygame.font.SysFont(pick, 24, bold=True)
        self.f_huge = pygame.font.SysFont(pick, 54, bold=True)
        self.f_big = pygame.font.SysFont(pick, 30, bold=True)
        self.f_mid = pygame.font.SysFont(pick, 17, bold=True)
        self.f_small = pygame.font.SysFont(pick, 14)
        self.f_tiny = pygame.font.SysFont(pick, 12)

        self.catalog = build_catalog()
        self.engine_idx = 0
        self.variant_idx = [0] * len(self.catalog)
        self.units = "kmh"
        self.held_ui = None
        self.disp_rpm = 0.0
        self.bf_flash = 0.0        # backfire flame flash
        self.bf_t = 0.0            # backfire scheduler
        self.msg, self.msg_col, self.msg_t = "", self.TEXT, 0.0

        self.tx = Transmission()
        self.vehicle = Vehicle(self.tx)
        self.synth = AudioSynthesizer()

        self.cyl_rect = pygame.Rect(20, 62, 620, 356)
        self.gauge_rect = pygame.Rect(650, 62, 770, 452)
        self.data_rect = pygame.Rect(650, 522, 770, 166)
        self.help_rect = pygame.Rect(650, 696, 770, 178)
        self.scope_rect = pygame.Rect(20, 694, 620, 110)
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
            self.variant_idx[self.engine_idx] = \
                (self.variant_idx[self.engine_idx] + 1) % len(variants)
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
        self.crank_y = inner.centery
        self.inner = inner
        for bi, bank in enumerate(cfg.banks):
            y = inner.top + py * (bi + 0.5)
            off = 0.0
            if B > 1:
                off = 0.11 * px if bi % 2 else -0.11 * px     # bank stagger
            for ci, num in enumerate(bank):
                cyl = self.cylinders[num - 1]
                cyl.pos = (inner.left + px * (ci + 0.5) + off, y)
                cyl.radius = radius

    # ----- buttons ----------------------------------------------------------- #
    def _build_buttons(self):
        b = []
        names = ["I4", "V6", "V8", "V10", "V12", "W16", "Flat-6"]
        gap, x0, w = 6, 20, (620 - 6 * 6) // 7
        for i, nm in enumerate(names):
            b.append(Button((x0 + i * (w + gap), 470, w, 42), f"{i + 1} {nm}",
                            on_click=lambda i=i: self.select_engine(i),
                            active=lambda i=i: self.engine_idx == i))
        w2 = 96
        row1 = [
            (lambda: "STOP" if (self.vehicle.running or self.vehicle.cranking) else "START",
             self.vehicle.toggle, lambda: self.vehicle.running or self.vehicle.cranking, None),
            ("DRIVE  D", self.vehicle.request_drive, lambda: self.tx.mode == "D",
             lambda: self.vehicle.running),
            ("NEUTRAL  N", self.vehicle.request_neutral, lambda: self.tx.mode == "N", None),
            (lambda: "MANUAL  M" if self.tx.manual else "AUTO  M",
             self.toggle_manual, lambda: self.tx.manual, None),
            (lambda: "UNITS: km/h" if self.units == "kmh" else "UNITS: mph",
             self.toggle_units, None, None),
            ("VARIANT  V", self.cycle_variant, None,
             lambda: len(self.catalog[self.engine_idx]) > 1),
        ]
        for i, (lab, cb, act, en) in enumerate(row1):
            b.append(Button((20 + i * (w2 + 8), 522, w2, 42), lab,
                            on_click=cb, active=act, enabled=en))
        row2 = [
            ("UPSHIFT  E", lambda: self.do_shift(1), None, lambda: self.tx.manual),
            ("DOWNSHIFT  Q", lambda: self.do_shift(-1), None, lambda: self.tx.manual),
            (lambda: f"{self.vehicle.induction}  T", self.vehicle.cycle_induction,
             lambda: self.vehicle.induction != "NA", None),
            ("REFUEL  R", self.vehicle.refuel, None, None),
            ("REV-MATCH  X", self.toggle_revmatch, lambda: self.tx.revmatch_assist,
             lambda: self.tx.manual),
            (lambda: "SOUND: OFF" if self.synth.muted else "SOUND: ON",
             self.toggle_mute, lambda: self.synth.muted, None),
        ]
        for i, (lab, cb, act, en) in enumerate(row2):
            b.append(Button((20 + i * (w2 + 8), 572, w2, 42), lab,
                            on_click=cb, active=act, enabled=en))
        b.append(Button((20, 622, 201, 64), "THROTTLE  [ W / Up ]", hold="throttle"))
        b.append(Button((229, 622, 201, 64), "BRAKE  [ S / Down ]", hold="brake"))
        b.append(Button((438, 622, 201, 64), "CLUTCH  [ LShift / C ]", hold="clutch"))
        return b

    # ----- small command helpers --------------------------------------------- #
    def toast(self, text, kind="info"):
        colors = {"info": self.TEXT, "warn": self.ORANGE, "bad": self.RED, "good": self.GREEN}
        self.msg = text
        self.msg_col = colors.get(kind, self.TEXT)
        self.msg_t = 2.8

    def toggle_units(self):
        self.units = "mph" if self.units == "kmh" else "kmh"

    def toggle_mute(self):
        self.synth.muted = not self.synth.muted

    def toggle_manual(self):
        tx, v = self.tx, self.vehicle
        tx.manual = not tx.manual
        if tx.manual and tx.mode == "D" and v.speed < 3.0:
            tx.mode = "N"                       # dip the clutch for the driver
            tx.gear = tx.target = 1
        if tx.manual:
            self.toast("gearbox: MANUAL - hold LShift/C (clutch), shift with E/Q", "info")
        else:
            self.toast("gearbox: AUTOMATIC", "info")

    def toggle_revmatch(self):
        self.tx.revmatch_assist = not self.tx.revmatch_assist
        self.toast(f"rev-match assist {'ON' if self.tx.revmatch_assist else 'OFF'}", "info")

    def toggle_fast_fuel(self):
        v = self.vehicle
        v.fuel_scale = 8.0 if v.fuel_scale == 1.0 else 1.0
        self.toast(f"fuel drain x{v.fuel_scale:.0f}", "warn")

    def do_shift(self, d):
        if not self.tx.manual:
            self.toast("gearbox is AUTOMATIC - press M for manual", "warn")
            return
        self.vehicle.shift(d)

    # ----- update ------------------------------------------------------------ #
    def update(self, dt):
        keys = pygame.key.get_pressed()
        v = self.vehicle
        v.throttle_cmd = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP] or
                                 self.held_ui == "throttle") else 0.0
        v.brake_cmd = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN] or
                              self.held_ui == "brake") else 0.0
        v.clutch_cmd = 1.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or
                               keys[pygame.K_c] or self.held_ui == "clutch") else 0.0
        v.step(dt)

        # drain vehicle events -> audio one-shots / toast messages
        for kind, data in v.pop_events():
            if kind == "audio":
                self.synth.post_event(data[0], data[1])
            else:
                self.toast(data[0], data[1])

        # overrun backfires: big bangs + exhaust flames on high-rpm off-throttle
        cfg = self.cfg
        if v.running and v.thr_eff < 0.06 and v.rpm > 0.6 * cfg.redline:
            self.bf_t -= dt
            if self.bf_t <= 0.0:
                self.synth.post_event("bang", random.uniform(0.45, 1.0))
                self.bf_flash = 1.0
                self.bf_t = random.uniform(0.05, 0.22)
        else:
            self.bf_t = 0.1
        self.bf_flash = max(0.0, self.bf_flash - 5.0 * dt)
        self.msg_t = max(0.0, self.msg_t - dt)

        # visual firing (same order & rate as the audio)
        if v.running and not v.fuel_cut:
            self.fire_phase += cfg.pulse_hz(v.rpm) * dt
            guard = 0
            while self.fire_phase >= 1.0 and guard < 64:
                self.fire_phase -= 1.0
                self.cylinders[cfg.fire_order[self.fire_slot] - 1].fire()
                self.fire_slot = (self.fire_slot + 1) % cfg.cylinders
                guard += 1
        for c in self.cylinders:
            c.update(dt)

        s = self.synth
        s.rpm, s.throttle = v.rpm, v.thr_eff
        s.running, s.fuel_cut = v.running, v.fuel_cut
        s.boost, s.induction = v.boost, v.induction
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

    # ----- header + warning pills -------------------------------------------- #
    def draw_header(self):
        cfg, v, tx = self.cfg, self.vehicle, self.tx
        self.text("PROCEDURAL ENGINE SIMULATOR", self.f_title, self.TEXT, (20, 14), "topleft")
        self.text(f"{cfg.label}  \u2022  {cfg.description}", self.f_small, self.DIM,
                  (20, 42), "topleft")
        if v.engine_failed:
            st, col = "ENGINE FAILED", self.RED
        elif v.cranking:
            st, col = "CRANKING", self.ORANGE
        elif v.running:
            st, col = ("REV LIMITER" if v.fuel_cut else "RUNNING"), \
                      (self.RED if v.fuel_cut else self.GREEN)
        elif v.fuel <= 0.0:
            st, col = "NO FUEL", self.ORANGE
        else:
            st, col = "ENGINE OFF", self.DIM
        pygame.draw.circle(self.screen, col, (self.W - 160, 24), 8)
        self.text(st, self.f_mid, col, (self.W - 144, 24), "midleft")

        # warning / mode pills (drawn right-to-left)
        items = [("MANUAL" if tx.manual else "AUTO",
                  (90, 170, 255) if tx.manual else self.DIM)]
        if v.fuel <= 0.0:
            items.append(("NO FUEL", self.RED))
        elif v.fuel < 0.15 * v.FUEL_CAP:
            items.append(("LOW FUEL", self.ORANGE))
        if v.engine_failed:
            items.append(("ENGINE FAILED", self.RED))
        if v.coolant >= 108.0:
            items.append(("OVERHEAT", self.RED if v.coolant >= 118.0 else self.ORANGE))
        if v.running and v.oil_p < 0.75:
            items.append(("OIL PRESS", self.RED))
        if v.fuel_cut:
            items.append(("LIMITER", self.RED))
        if v.blip_t > 0.0:
            items.append(("REV-MATCH", self.GREEN))
        if v.bov_flash > 0.05:
            items.append(("BOV DUMP", (90, 220, 255)))
        if v.stall_flash > 0.0:
            items.append(("STALLED", self.ORANGE))
        x = self.W - 330
        for label, col in reversed(items):
            w = self.f_tiny.size(label)[0] + 18
            x -= w
            rect = pygame.Rect(x, 13, w, 22)
            pygame.draw.rect(self.screen, (24, 28, 38), rect, border_radius=11)
            pygame.draw.rect(self.screen, col, rect, 1, border_radius=11)
            self.text(label, self.f_tiny, col, rect.center)
            x -= 8

    # ----- cylinder view + exhaust flames ------------------------------------- #
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
                for scale, a in ((1.0 + 0.9 * c.glow, 45), (1.0 + 0.55 * c.glow, 80),
                                 (1.0 + 0.25 * c.glow, 120)):
                    pygame.draw.circle(gl, (255, 110, 20, int(a * c.glow)),
                                       (px, py), int(c.radius * scale))
        s.blit(gl, rect.topleft)
        for bi in range(len(cfg.banks)):
            first = self.cylinders[cfg.banks[bi][0] - 1]
            self.text(chr(65 + bi), self.f_mid, self.DIM, (self.inner.left - 24, first.pos[1]))
        font = pygame.font.SysFont("dejavusans,arial",
                                   max(11, int(self.cylinders[0].radius * 0.6)), bold=True)
        for c in self.cylinders:
            x, y = int(c.pos[0]), int(c.pos[1])
            pygame.draw.circle(s, (90, 100, 125), (x, y), c.radius + 2)
            pygame.draw.circle(s, Cylinder.glow_color(c.glow), (x, y), c.radius)
            pygame.draw.circle(s, lerp_color((60, 68, 88), (255, 220, 120), c.glow),
                               (x, y), int(c.radius * 0.62), 2)
            self.text(str(c.number), font,
                      lerp_color((170, 180, 200), (255, 255, 255), c.glow), (x, y))
        ix, iy = rect.right - 52, rect.top + 52
        for a in cfg.bank_angles:
            r = math.radians(a)
            pygame.draw.line(s, (150, 160, 185), (ix, iy),
                             (ix + 34 * math.sin(r), iy - 34 * math.cos(r)), 5)
        pygame.draw.circle(s, self.ORANGE, (ix, iy), 5)
        self.text("end view", self.f_tiny, self.DIM, (ix, iy + 44))
        order = "-".join(str(n) for n in cfg.fire_order)
        self.text(f"Firing order: {order}", self.f_small, self.TEXT,
                  (rect.centerx, rect.bottom - 18))

        # exhaust tips + backfire flames
        tips = [(rect.left + 46, rect.bottom - 34), (rect.right - 46, rect.bottom - 34)]
        for px, py in tips:
            pygame.draw.rect(s, (74, 82, 104), (px - 9, py - 4, 18, 10), border_radius=3)
            self.text("EXH", self.f_tiny, self.DIM, (px, rect.bottom - 14))
        if self.bf_flash > 0.03:
            f = self.bf_flash
            tix = pygame.time.get_ticks() * 0.06
            for px, py in tips:
                fl = 0.8 + 0.35 * math.sin(tix + px * 0.13)
                rad = int((10 + 30 * f) * fl)
                pygame.draw.circle(s, (255, 80, 18), (px, py - 6), rad)
                pygame.draw.circle(s, (255, 170, 40), (px, py - 8), int(rad * 0.62))
                pygame.draw.circle(s, (255, 240, 170), (px, py - 10), max(2, int(rad * 0.3)))

    # ----- gauges ------------------------------------------------------------- #
    def draw_gauge(self, cx, cy, r, value, vmax, major, minor, divisor, redline=None,
                   title="", vmin=0.0, font=None, redlow=None, label_map=None):
        s = self.screen
        font = font or self.f_mid
        pygame.draw.circle(s, (8, 10, 15), (cx, cy), r + 9)
        pygame.draw.circle(s, (70, 80, 105), (cx, cy), r + 9, 3)
        pygame.draw.circle(s, (17, 20, 29), (cx, cy), r)
        A0, SW = 135.0, 270.0
        span = float(vmax - vmin)

        def ang(v):
            return A0 + SW * clamp((v - vmin) / span, 0.0, 1.0)

        if redline is not None:
            pts = [self._pt(cx, cy, r * 0.90, ang(redline + (vmax - redline) * i / 30))
                   for i in range(31)]
            pygame.draw.lines(s, (190, 28, 28), False, pts, 7)
        if redlow is not None:
            pts = [self._pt(cx, cy, r * 0.90, ang(vmin + (redlow - vmin) * i / 30))
                   for i in range(31)]
            pygame.draw.lines(s, (190, 28, 28), False, pts, 7)
        steps = int(round(span / minor))
        for i in range(steps + 1):
            val = vmin + i * minor
            off = i * minor
            is_major = (off % major) < 1e-9 or (major - off % major) < 1e-9
            col = (236, 240, 248) if is_major else (115, 124, 148)
            if (redline is not None and val >= redline) or \
               (redlow is not None and val <= redlow + 1e-9):
                col = (255, 95, 75)
            r1 = r * (0.78 if is_major else 0.86)
            pygame.draw.line(s, col, self._pt(cx, cy, r1, ang(val)),
                             self._pt(cx, cy, r * 0.95, ang(val)), 3 if is_major else 1)
            if is_major:
                txt = None
                if label_map:
                    for k, v_ in label_map.items():
                        if abs(val - k) < 1e-6:
                            txt = v_
                            break
                if txt is None:
                    txt = str(int(round(val / divisor)))
                lx, ly = self._pt(cx, cy, r * 0.63, ang(val))
                self.text(txt, font, self.TEXT, (lx, ly))
        self.text(title, self.f_tiny, self.DIM, (cx, cy - r * 0.32))
        a = ang(value)
        tip, tail = self._pt(cx, cy, r * 0.86, a), self._pt(cx, cy, r * 0.14, a + 180)
        p1, p2 = self._pt(cx, cy, 5, a + 90), self._pt(cx, cy, 5, a - 90)
        pygame.draw.polygon(s, (255, 85, 40), [tip, (cx + (p1[0] - cx), cy + (p1[1] - cy)),
                                               tail, (cx + (p2[0] - cx), cy + (p2[1] - cy))])
        pygame.draw.circle(s, (45, 52, 70), (cx, cy), 11)
        pygame.draw.circle(s, (255, 85, 40), (cx, cy), 5)

    def draw_gauges(self):
        self.panel(self.gauge_rect)
        cfg, v, tx = self.cfg, self.vehicle, self.tx
        tach_max = max(9000, int(math.ceil((cfg.redline + 800) / 1000.0)) * 1000)
        tcx, tcy, scx, scy, r = 845, 205, 1240, 205, 132
        self.draw_gauge(tcx, tcy, r, self.disp_rpm, tach_max, 1000, 500, 1000,
                        redline=cfg.redline, title="RPM x1000")
        self.text(f"{int(self.disp_rpm):d}", self.f_big, self.TEXT, (tcx, tcy + 64))
        self.text(tx.display(), self.f_huge, self.ORANGE, (tcx, tcy + 112))
        self.text("SEQUENTIAL" if tx.manual else "AUTOMATIC", self.f_tiny,
                  (140, 220, 255) if tx.manual else self.DIM, (tcx, tcy + 142))
        spd = v.speed * (3.6 if self.units == "kmh" else 2.23694)
        vmax = 400 if self.units == "kmh" else 250
        self.draw_gauge(scx, scy, r, spd, vmax, 50, 10, 1, title="SPEED")
        self.text(f"{int(spd):d}", self.f_huge, self.TEXT, (scx, scy + 70))
        self.text("km/h" if self.units == "kmh" else "mph", self.f_mid, self.DIM,
                  (scx, scy + 112))

        # mini gauges: boost/manifold, coolant, fuel, oil pressure
        my = 390
        self.draw_gauge(735, my, 52, v.map_bar, 2.0, 1.0, 0.25, 1, redline=1.45,
                        vmin=-1.0, font=self.f_tiny, title="BOOST bar")
        self.text(f"{v.map_bar:+.2f}", self.f_small,
                  (255, 95, 75) if v.map_bar > 1.45 else self.TEXT, (735, 452))
        self.draw_gauge(935, my, 52, v.coolant, 130.0, 40.0, 10.0, 1, vmin=20.0,
                        redline=112.0, font=self.f_tiny, title="COOLANT \u00b0C")
        tcol = self.RED if v.coolant >= 108 else (self.ORANGE if v.coolant >= 96 else self.TEXT)
        self.text(f"{v.coolant:4.0f}", self.f_small, tcol, (935, 452))
        self.draw_gauge(1135, my, 52, v.fuel / v.FUEL_CAP, 1.0, 0.5, 0.25, 1,
                        redlow=0.12, font=self.f_tiny, title="FUEL",
                        label_map={0.0: "E", 0.5: "1/2", 1.0: "F"})
        fcol = self.RED if v.fuel <= 0.0 else \
            (self.ORANGE if v.fuel < 0.15 * v.FUEL_CAP else self.TEXT)
        self.text(f"{v.fuel:4.1f}L", self.f_small, fcol, (1135, 452))
        self.draw_gauge(1335, my, 52, v.oil_p, 8.0, 2.0, 1.0, 1, redlow=1.0,
                        font=self.f_tiny, title="OIL bar")
        ocol = self.RED if (v.running and v.oil_p < 0.75) else self.TEXT
        self.text(f"{v.oil_p:4.1f}", self.f_small, ocol, (1335, 452))

        # gear / mode strip
        strip = pygame.Rect(672, 472, 726, 30)
        pygame.draw.rect(self.screen, (13, 16, 24), strip, border_radius=8)
        labels = ["MT" if tx.manual else "AT", "P", "N"] + [str(i) for i in range(1, 8)]
        cw = strip.w / len(labels)
        cur = tx.display()
        for i, lab in enumerate(labels):
            cell = pygame.Rect(int(strip.x + i * cw), strip.y, int(cw) + 1, strip.h)
            if i == 0:
                if tx.manual:
                    pygame.draw.rect(self.screen, (36, 90, 175),
                                     cell.inflate(-6, -6), border_radius=6)
                    self.text(lab, self.f_tiny, (215, 232, 255), cell.center)
                else:
                    self.text(lab, self.f_tiny, self.DIM, cell.center)
            elif lab == cur:
                pygame.draw.rect(self.screen, self.ORANGE, cell.inflate(-6, -6),
                                 border_radius=6)
                self.text(lab, self.f_mid, (20, 20, 20), cell.center)
            else:
                self.text(lab, self.f_mid, self.DIM, cell.center)

    # ----- engine data panel -------------------------------------------------- #
    def draw_data(self):
        rect = self.data_rect
        self.panel(rect)
        v, tx = self.vehicle, self.tx

        self.text("ENGINE DATA", self.f_tiny, self.DIM, (rect.x + 14, rect.y + 8), "topleft")

        # AFR bar (scale 10:1 .. 17:1, stoich marker at 14.7)
        ax, ay, aw, ah = rect.x + 52, rect.y + 26, 330, 15
        self.text("AFR", self.f_small, self.DIM, (rect.x + 14, ay + 8), "midleft")
        zones = ((10.0, 13.0, (150, 62, 46)), (13.0, 15.0, (52, 118, 74)),
                 (15.0, 17.0, (56, 90, 140)))
        for lo, hi, col in zones:
            x0 = ax + (lo - 10.0) / 7.0 * aw
            pygame.draw.rect(self.screen, col, (int(x0), ay, int((hi - lo) / 7.0 * aw), ah))
        pygame.draw.rect(self.screen, self.BORDER, (ax, ay, aw, ah), 1)
        mx = ax + (14.7 - 10.0) / 7.0 * aw
        pygame.draw.line(self.screen, (235, 240, 250), (mx, ay - 3), (mx, ay + ah + 3), 1)
        if v.afr > 0.0:
            nx = ax + clamp((v.afr - 10.0) / 7.0, 0.0, 1.0) * aw
            pygame.draw.rect(self.screen, (255, 200, 60), (int(nx) - 2, ay - 3, 4, ah + 6))
            if v.afr < 13.2:
                atxt, acol = "RICH", self.ORANGE
            elif v.afr < 15.0:
                atxt, acol = "STOICH", self.GREEN
            else:
                atxt, acol = "LEAN", (120, 170, 255)
            val, acol = f"{v.afr:4.1f} : 1  {atxt}", acol
        else:
            state = "CUT" if (v.fuel_cut or v.decel_cut) else "off"
            val, acol = f" \u2014  : 1  {state}", self.DIM
        self.text(val, self.f_small, acol, (ax + aw + 16, ay + 8), "midleft")

        kmh = v.speed * 3.6
        if kmh > 3.0 and v.fuel_lph > 0.0:
            cons = f"{v.fuel_lph / kmh * 100:4.1f} L/100km"
        else:
            cons = "    \u2014     L/100km"
        fast = "   [DRAIN x8]" if v.fuel_scale > 1.5 else ""
        fcol = self.RED if v.fuel <= 0.0 else \
            (self.ORANGE if v.fuel < 0.15 * v.FUEL_CAP else self.TEXT)
        tcol = self.RED if v.coolant >= 108 else \
            (self.ORANGE if v.coolant >= 96 else self.TEXT)

        def row(y, label, value, col=None):
            self.text(label, self.f_small, self.DIM, (rect.x + 14, y), "midleft")
            self.text(value, self.f_small, col or self.TEXT, (rect.x + 150, y), "midleft")

        y0 = rect.y + 52
        row(y0, "FUEL",
            f"{v.fuel:5.1f} / {v.FUEL_CAP:.0f} L    {v.fuel_lph:5.1f} L/h    {cons}{fast}", fcol)
        row(y0 + 19, "COOLANT", f"{v.coolant:5.1f} \u00b0C    OIL {v.oil_p:4.2f} bar", tcol)
        row(y0 + 38, "INDUCTION",
            f"{v.induction:6s}  boost {v.boost:+.2f} bar   manifold {v.map_bar:+.2f} bar")
        # clutch row with a mini engagement bar
        cy = y0 + 57
        self.text("CLUTCH", self.f_small, self.DIM, (rect.x + 14, cy), "midleft")
        bar = pygame.Rect(rect.x + 150, cy - 7, 160, 14)
        pygame.draw.rect(self.screen, (24, 28, 40), bar, border_radius=4)
        eng = v.clutch_disp
        if eng > 0.003:
            pygame.draw.rect(self.screen,
                             lerp_color((230, 70, 60), (60, 205, 110), eng),
                             (bar.x, bar.y, max(2, int(bar.w * eng)), bar.h), border_radius=4)
        pygame.draw.rect(self.screen, self.BORDER, bar, 1, border_radius=4)
        self.text(f"{int(eng * 100):3d}% engaged", self.f_small, self.TEXT,
                  (bar.right + 12, cy), "midleft")
        row(y0 + 76, "GEARBOX",
            f"{'MANUAL sequential' if tx.manual else 'AUTOMATIC'}    "
            f"gear {tx.display()}    rev-match {'ON' if tx.revmatch_assist else 'OFF'}")
        row(y0 + 95, "TRIP",
            f"{v.trip / 1000.0:6.2f} km    cooling fan "
            f"{'ON' if (v.running and v.coolant > 92.0) else 'off'}")

    # ----- help panel --------------------------------------------------------- #
    def draw_help(self):
        rect = self.help_rect
        self.panel(rect)
        self.text("CONTROLS", self.f_tiny, self.DIM, (rect.x + 14, rect.y + 8), "topleft")
        entries = [
            ("W / Up", "throttle (hold)"), ("S / Down", "brake (hold)"),
            ("LShift / C", "clutch (hold)"), ("Space", "start / stop engine"),
            ("E / Q", "shift up / down"), ("M", "manual / automatic gearbox"),
            ("T", "induction: NA / turbo / super"), ("R", "refuel"),
            ("X", "rev-match assist"), ("D / N", "drive / neutral"),
            ("1-7", "select engine"), ("V", "cycle variant"),
            ("U", "km/h <-> mph"), ("B", "sound on / off"),
            ("F", "fast fuel drain (demo)"), ("Esc", "quit"),
        ]
        for i, (key, desc) in enumerate(entries):
            col, rowi = i // 8, i % 8
            x = rect.x + 16 + col * 386
            y = rect.y + 26 + rowi * 18
            self.text(key, self.f_small, self.TEXT, (x, y), "midleft")
            self.text(desc, self.f_small, self.DIM, (x + 118, y), "midleft")
        self.text("tip: press M, hold the clutch (LShift/C), shift with E/Q, "
                  "release the clutch gently to launch",
                  self.f_tiny, self.DIM, (rect.x + 16, rect.bottom - 12), "topleft")

    # ----- bars / scope / info lines ------------------------------------------ #
    def draw_bars(self):
        v = self.vehicle
        bars = [
            (20, v.throttle, "THROTTLE", (60, 205, 110)),
            (232, v.brake, "BRAKE", (230, 70, 60)),
            (444, v.clutch_disp, "CLUTCH ENGAGEMENT",
             lerp_color((230, 70, 60), (60, 205, 110), v.clutch_disp)),
        ]
        for x, val, label, col in bars:
            r = pygame.Rect(x, 428, 196, 22)
            pygame.draw.rect(self.screen, (24, 28, 40), r, border_radius=6)
            if val > 0.003:
                pygame.draw.rect(self.screen, col,
                                 (r.x, r.y, max(2, int(r.w * val)), r.h), border_radius=6)
            pygame.draw.rect(self.screen, self.BORDER, r, 1, border_radius=6)
            self.text(f"{label} {int(val * 100):d}%", self.f_tiny,
                      (255, 255, 255), r.center)
        self.text("ENGINE SELECT", self.f_tiny, self.DIM, (20, 456), "topleft")

    def draw_scope(self):
        rect = self.scope_rect
        s = self.screen
        self.panel(rect)
        self.text("SYNTH OUTPUT", self.f_tiny, self.DIM, (rect.x + 12, rect.y + 8), "topleft")
        mid = rect.centery + 8
        pygame.draw.line(s, (45, 52, 72), (rect.x + 8, mid), (rect.right - 8, mid), 1)
        if self.synth.enabled:
            wave = self.synth.scope[::3]
            pts = [(rect.x + 10 + i * (rect.w - 20) / max(1, len(wave) - 1),
                    mid - float(a) * 38) for i, a in enumerate(wave)]
            if len(pts) > 1:
                pygame.draw.lines(s, (80, 220, 255), False, pts, 1)
        else:
            self.text("audio device unavailable", self.f_small, self.DIM, rect.center)

    def _status_tip(self):
        v, tx = self.vehicle, self.tx
        if v.engine_failed:
            return f"ENGINE FAILED ({v.fail_reason}) - let it cool below 95 \u00b0C"
        if not v.running and not v.cranking:
            return "SPACE to start the engine"
        if v.fuel <= 0.0:
            return "OUT OF FUEL - press R"
        if v.coolant >= 108.0:
            return "OVERHEATING - lift off and cool down!"
        if tx.manual and v.running and v.clutch_disp > 0.05 and v.rpm < self.cfg.idle * 1.5:
            return "clutch slipping - add throttle or release the pedal"
        return "T induction  \u00b7  M gearbox mode  \u00b7  F fast fuel drain  \u00b7  X rev-match"

    def draw_info_lines(self):
        v, cfg = self.vehicle, self.cfg
        hz = cfg.pulse_hz(v.rpm)
        lines = [
            f"Pulse rate  Hz = (RPM/60) x (Cyl/2) = {hz:.0f} Hz      "
            f"Manifold {v.map_bar:+.2f} bar      Boost {v.boost:+.2f} bar",
            f"Redline {int(cfg.redline)} rpm   Idle {int(cfg.idle)} rpm   "
            f"Peak {int(cfg.peak_torque)} Nm @ {int(cfg.peak_rpm)}   ~{cfg.disp:.1f} L   "
            f"mass {int(cfg.mass)} kg",
            self._status_tip(),
        ]
        for i, ln in enumerate(lines):
            self.text(ln, self.f_small, self.TEXT if i < 2 else self.ORANGE,
                      (20, 812 + i * 22), "topleft")

    def draw(self):
        self.screen.fill(self.BG)
        self.draw_header()
        self.draw_cylinders()
        self.draw_gauges()
        self.draw_data()
        self.draw_help()
        self.draw_bars()
        self.draw_scope()
        self.draw_info_lines()
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            b.draw(self.screen, self.f_mid if b.hold or b.rect.w > 100 else self.f_small,
                   mouse, self.held_ui)
        if self.msg_t > 0.0:
            self.text(self.msg, self.f_mid, self.msg_col, (self.W // 2, self.H - 14))

    # ----- events / main loop ------------------------------------------------ #
    def on_mouse_down(self, pos):
        for b in self.buttons:
            if b.rect.collidepoint(pos) and (b.enabled() if b.enabled else True):
                if b.hold:
                    self.held_ui = b.hold
                elif b.on_click:
                    b.on_click()
                return

    def run(self):
        self.synth.start()
        alive = True
        while alive:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.KEYDOWN:
                    k = e.key
                    if k == pygame.K_ESCAPE:
                        alive = False
                    elif k == pygame.K_SPACE:
                        self.vehicle.toggle()
                    elif pygame.K_1 <= k <= pygame.K_7:
                        self.select_engine(k - pygame.K_1)
                    elif k == pygame.K_v:
                        self.cycle_variant()
                    elif k == pygame.K_d:
                        self.vehicle.request_drive()
                    elif k == pygame.K_n:
                        self.vehicle.request_neutral()
                    elif k == pygame.K_m:
                        self.toggle_manual()
                    elif k == pygame.K_e:
                        self.do_shift(1)
                    elif k == pygame.K_q:
                        self.do_shift(-1)
                    elif k == pygame.K_t:
                        self.vehicle.cycle_induction()
                    elif k == pygame.K_r:
                        self.vehicle.refuel()
                    elif k == pygame.K_x:
                        self.toggle_revmatch()
                    elif k == pygame.K_f:
                        self.toggle_fast_fuel()
                    elif k == pygame.K_b:
                        self.toggle_mute()
                    elif k == pygame.K_u:
                        self.toggle_units()
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
    MainApp().run()