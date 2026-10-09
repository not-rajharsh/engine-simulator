#!/usr/bin/env python3
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
    W / Up      Throttle (hold)         S / Down   Brake (hold)
    Space       Start / Stop engine     1-7        Engine: I4 V6 V8 V10 V12 W16 Flat-6
    D           Drive (auto gearbox)    N          Neutral (free-rev the engine)
    V           Cycle variant (V6 60/90 deg, V8 cross-/flat-plane)
    U           km/h <-> mph            M          Mute
    Esc         Quit
    Every key also has an on-screen button (hold the THROTTLE / BRAKE buttons).

TIP: start the engine, stay in Neutral and blip the throttle to hear each engine
rev freely; press D to drive. Lift off at speed to hear engine-braking overrun.

Sound model: Hz = (RPM / 60) * (Cylinders / 2) firing pulses per second. Every
pulse is a pre-synthesised burst (pitch-dropping sine + band-passed noise +
tanh overdrive, per cylinder, per RPM band, per load) that is overlap-added at
the exact firing instants, plus phase-continuous harmonics, a rumble/intake bed
and overrun gurgle/pops.
"""

import math
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

    Shared (written by the main thread): rpm, throttle, running, fuel_cut.
    A worker thread renders CHUNK-sample blocks and queues them as Sounds.
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
        self.cfg = None
        self.scope = np.zeros(CHUNK, dtype=np.float32)
        self.rng = np.random.default_rng(7)
        self._lock = threading.Lock()
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

    # ----- one-time noise beds / pop template ------------------------------- #
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

    def _reset_state(self):
        self.L = int(0.06 * self.sr)
        self.tail = np.zeros(self.L, dtype=np.float32)
        self.next_pulse = 0.0
        self.slot = 0
        self.hum_phase = 0.0
        self.prev_hz = 0.0
        self.gains = {}
        self.idx = {"rumble": 0, "air": 0, "gur": 40000}
        self.T = None

    # ----- per-engine pulse templates --------------------------------------- #
    def set_engine(self, cfg: EngineConfig):
        with self._lock:
            self.cfg = cfg
            self._reset_state()
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
                level = 0.12                      # rev-limiter: fuel cut, only compression thumps
        else:
            level = 0.16 * clamp(rpm / 250.0, 0.0, 1.0)     # cranking / spin-down thumps

        # blend burst templates by RPM band and load (throttle boosts the heavy/bass version)
        p = min(x, 1.0) * 2.0
        bi = min(int(p), 1)
        fr = p - bi
        wb = np.zeros(3, dtype=np.float32)
        wb[bi], wb[bi + 1] = 1.0 - fr, fr
        load = clamp((0.12 + 0.88 * thr) * (1.0 - 0.6 * ov), 0.0, 1.0)
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

        # overrun crackle / pops
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
        ag = cfg.noise_gain * 0.30 * thr * x ** 1.3 if running else 0.0
        out += self._loop("air", self.air, N) * self._ramp("air", ag, N)

        # deceleration: pulsing low gurgle (engine-braking overrun)
        gg = 0.8 * ov * (0.4 + cfg.sub_gain)
        mod = (0.55 + 0.45 * np.sin(ph * 0.5)).astype(np.float32)
        out += self._loop("gur", self.rumble, N) * mod * self._ramp("gur", gg, N)

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
    """7-speed automatic with P / N / D, kick-down and rev-matched downshifts."""
    RATIOS = (3.62, 2.38, 1.72, 1.34, 1.08, 0.88, 0.73)
    SHIFT_TIME = 0.30

    def __init__(self):
        self.mode = "P"
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
        self.shift_timer = self.SHIFT_TIME
        self.cooldown = 1.0


# --------------------------------------------------------------------------- #
# Vehicle physics (engine + clutch + chassis)
# --------------------------------------------------------------------------- #
class Vehicle:
    WHEEL_R = 0.33
    K_CLUTCH = 80.0        # Nm per rad/s of slip

    def __init__(self, tx: Transmission):
        self.tx = tx
        self.cfg = None
        self.omega = 0.0           # crank speed, rad/s
        self.speed = 0.0           # m/s
        self.throttle_cmd = self.throttle = self.thr_eff = 0.0
        self.brake_cmd = self.brake = 0.0
        self.running = self.cranking = self.fuel_cut = False
        self.crank_t = 0.0
        self.run_t = 99.0

    @property
    def rpm(self):
        return self.omega * 60.0 / TAU

    def set_engine(self, cfg: EngineConfig):
        keep = self.running
        self.cfg = cfg
        self.speed = 0.0
        self.tx.configure(cfg, self.WHEEL_R)
        self.tx.reset("N" if keep else "P")
        self.omega = cfg.idle * TAU / 60.0 if keep else 0.0
        self.cranking = self.fuel_cut = False
        self.run_t = 99.0

    def toggle(self):
        if self.running or self.cranking:
            self.running = self.cranking = False
        else:
            self.cranking, self.crank_t = True, 0.0

    def request_drive(self):
        if self.running:
            self.tx.engage_drive(self.speed, self.cfg, self.WHEEL_R)

    def request_neutral(self):
        if self.tx.mode != "P":
            self.tx.mode = "N"

    def engine_torque(self, rpm, thr):
        c = self.cfg
        x = rpm / c.redline
        fr = c.fric * (0.45 + 0.9 * x)                     # friction
        if self.cranking:
            return (c.fric * 3.5 if rpm < 260 else 0.0) - fr
        if not self.running:
            return -fr - c.fric * 0.6                      # compression drag on spin-down
        if self.fuel_cut:
            return -fr - c.fric * 1.3 * x
        pump = c.fric * 1.3 * x * (1.0 - thr)              # throttle-closed pumping loss
        curve = max(0.42, 1.0 - 1.15 * ((rpm - c.peak_rpm) / c.redline) ** 2)
        T = c.peak_torque * curve * thr
        if thr < 0.25:                                     # idle controller (+ start-up flare)
            target = c.idle * (1.0 + 0.8 * max(0.0, 1.0 - self.run_t / 1.8))
            if rpm < target * 1.35:
                kp = 0.0006 * c.peak_torque
                T += (1.0 - thr) * (fr + c.fric * 1.3 * x +
                                    clamp(kp * (target - rpm), -0.05 * c.peak_torque,
                                          0.3 * c.peak_torque))
        return T - fr - pump

    def step(self, dt):
        c, tx, R = self.cfg, self.tx, self.WHEEL_R
        self.throttle += clamp(self.throttle_cmd - self.throttle, -4.0 * dt, 2.8 * dt)
        self.brake += clamp(self.brake_cmd - self.brake, -6.0 * dt, 7.0 * dt)

        if self.cranking:
            self.crank_t += dt
            if self.crank_t >= 0.9:
                self.cranking, self.running, self.run_t = False, True, 0.0
        elif self.running:
            self.run_t += dt
        if not self.running and not self.cranking:
            if tx.mode == "D":
                tx.mode = "N"
            if self.speed < 0.3:
                tx.mode = "P"
        if self.running and tx.mode == "P":
            tx.mode = "N"

        tx.update(dt, self.rpm, self.speed, self.throttle, c, R)
        thr = self.throttle
        if tx.shifting:
            thr = max(thr, 0.55) if tx.shift_dir < 0 else thr * 0.25   # blip / lift
        self.thr_eff = thr

        if self.running:
            if self.rpm >= c.redline:
                self.fuel_cut = True
            elif self.rpm < c.redline - 250:
                self.fuel_cut = False
        else:
            self.fuel_cut = False

        n = max(8, int(math.ceil(dt * 960)))
        h = dt / n
        G = tx.total_ratio()
        m = c.mass
        fb = 14000.0 * self.brake + (16000.0 if tx.mode == "P" else 0.0)
        for _ in range(n):
            rpm = self.omega * 60.0 / TAU
            Te = self.engine_torque(rpm, thr)
            Tc = 0.0
            if G > 0.0:
                ww = self.speed * G / R
                rw = ww * 60.0 / TAU
                if rw >= 1300.0:
                    eng = 1.0
                else:   # clutch bites as engine revs / wheels turn
                    eng = max(smoothstep((rpm - (c.idle + 100.0)) / 900.0), rw / 1300.0)
                cap = min(c.peak_torque * 2.2, c.traction * m * 9.81 * R / G) * eng
                Tc = clamp(self.K_CLUTCH * (self.omega - ww), -cap, cap)
            self.omega = max(0.0, self.omega + (Te - Tc) / c.inertia * h)
            F = Tc * G / R if G > 0.0 else 0.0
            drag = 0.5 * 1.2 * c.cda * self.speed ** 2 + (0.012 * m * 9.81 if self.speed > 0.01 else 0.0)
            self.speed += F / m * h
            self.speed = max(0.0, self.speed - (drag + fb) / m * h)


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
    W, H = 1280, 760
    BG, PANEL, BORDER = (13, 15, 21), (21, 25, 35), (52, 60, 82)
    TEXT, DIM = (222, 228, 242), (128, 138, 160)
    ORANGE, RED, GREEN = (255, 140, 30), (230, 45, 45), (70, 205, 125)

    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption("Procedural Engine Simulator")
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

        self.tx = Transmission()
        self.vehicle = Vehicle(self.tx)
        self.synth = AudioSynthesizer()

        self.cyl_rect = pygame.Rect(20, 62, 620, 370)
        self.gauge_rect = pygame.Rect(650, 62, 610, 370)
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

    def _build_buttons(self):
        b = []
        names = ["I4", "V6", "V8", "V10", "V12", "W16", "Flat-6"]
        gap, x0, w = 6, 20, (620 - 6 * 6) // 7
        for i, nm in enumerate(names):
            b.append(Button((x0 + i * (w + gap), 546, w, 44), f"{i + 1} {nm}",
                            on_click=lambda i=i: self.select_engine(i),
                            active=lambda i=i: self.engine_idx == i))
        w2 = (620 - 5 * 8) // 6
        row = [
            (lambda: "STOP" if (self.vehicle.running or self.vehicle.cranking) else "START",
             self.vehicle.toggle, lambda: self.vehicle.running or self.vehicle.cranking, None),
            ("DRIVE (D)", self.vehicle.request_drive, lambda: self.tx.mode == "D",
             lambda: self.vehicle.running),
            ("NEUTRAL (N)", self.vehicle.request_neutral, lambda: self.tx.mode == "N", None),
            (lambda: "UNITS: km/h" if self.units == "kmh" else "UNITS: mph",
             self.toggle_units, None, None),
            ("VARIANT (V)", self.cycle_variant, None,
             lambda: len(self.catalog[self.engine_idx]) > 1),
            (lambda: "SOUND: OFF" if self.synth.muted else "SOUND: ON",
             self.toggle_mute, lambda: self.synth.muted, None),
        ]
        for i, (lab, cb, act, en) in enumerate(row):
            b.append(Button((20 + i * (w2 + 8), 602, w2, 44), lab, on_click=cb, active=act, enabled=en))
        b.append(Button((20, 658, 306, 72), "THROTTLE  [ W / Up ]  hold", hold="throttle"))
        b.append(Button((334, 658, 306, 72), "BRAKE  [ S / Down ]  hold", hold="brake"))
        return b

    def toggle_units(self):
        self.units = "mph" if self.units == "kmh" else "kmh"

    def toggle_mute(self):
        self.synth.muted = not self.synth.muted

    # ----- update ------------------------------------------------------------ #
    def update(self, dt):
        keys = pygame.key.get_pressed()
        v = self.vehicle
        v.throttle_cmd = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP] or self.held_ui == "throttle") else 0.0
        v.brake_cmd = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN] or self.held_ui == "brake") else 0.0
        v.step(dt)

        # visual firing (same order & rate as the audio)
        cfg = self.cfg
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
        self.text("PROCEDURAL ENGINE SIMULATOR", self.f_title, self.TEXT, (20, 14), "topleft")
        self.text(f"{cfg.label}  \u2022  {cfg.description}", self.f_small, self.DIM, (20, 42), "topleft")
        if v.cranking:
            st, col = "CRANKING", self.ORANGE
        elif v.running:
            st, col = ("REV LIMITER" if v.fuel_cut else "RUNNING"), (self.RED if v.fuel_cut else self.GREEN)
        else:
            st, col = "ENGINE OFF", self.DIM
        pygame.draw.circle(self.screen, col, (self.W - 160, 26), 8)
        self.text(st, self.f_mid, col, (self.W - 144, 26), "midleft")

    def draw_cylinders(self):
        s, rect, cfg = self.screen, self.cyl_rect, self.cfg
        self.panel(rect)
        self.text("CYLINDER BANKS  \u2013  top view, crank axis horizontal", self.f_tiny, self.DIM,
                  (rect.left + 16, rect.top + 12), "topleft")
        # crankshaft
        pygame.draw.line(s, (60, 68, 90), (self.inner.left - 8, self.crank_y),
                         (self.inner.right + 8, self.crank_y), 2)
        # glow layer
        gl = self.glow_layer
        gl.fill((0, 0, 0, 0))
        for c in self.cylinders:
            if c.glow > 0.03:
                px, py = c.pos[0] - rect.x, c.pos[1] - rect.y
                for k, (scale, a) in enumerate(((1.0 + 0.9 * c.glow, 45), (1.0 + 0.55 * c.glow, 80),
                                                (1.0 + 0.25 * c.glow, 120))):
                    pygame.draw.circle(gl, (255, 110, 20, int(a * c.glow)), (px, py), int(c.radius * scale))
        s.blit(gl, rect.topleft)
        # bodies
        for bi in range(len(cfg.banks)):
            first = self.cylinders[cfg.banks[bi][0] - 1]
            self.text(chr(65 + bi), self.f_mid, self.DIM, (self.inner.left - 24, first.pos[1]))
        font = pygame.font.SysFont("dejavusans,arial", max(11, int(self.cylinders[0].radius * 0.6)), bold=True)
        for c in self.cylinders:
            x, y = int(c.pos[0]), int(c.pos[1])
            pygame.draw.circle(s, (90, 100, 125), (x, y), c.radius + 2)
            pygame.draw.circle(s, Cylinder.glow_color(c.glow), (x, y), c.radius)
            pygame.draw.circle(s, lerp_color((60, 68, 88), (255, 220, 120), c.glow),
                               (x, y), int(c.radius * 0.62), 2)
            self.text(str(c.number), font, lerp_color((170, 180, 200), (255, 255, 255), c.glow), (x, y))
        # end-view bank-angle icon
        ix, iy = rect.right - 52, rect.top + 52
        for a in cfg.bank_angles:
            r = math.radians(a)
            pygame.draw.line(s, (150, 160, 185), (ix, iy), (ix + 34 * math.sin(r), iy - 34 * math.cos(r)), 5)
        pygame.draw.circle(s, self.ORANGE, (ix, iy), 5)
        self.text("end view", self.f_tiny, self.DIM, (ix, iy + 44))
        order = "-".join(str(n) for n in cfg.fire_order)
        self.text(f"Firing order: {order}", self.f_small, self.TEXT, (rect.centerx, rect.bottom - 18))

    def draw_gauge(self, cx, cy, r, value, vmax, major, minor, divisor, redline=None, title=""):
        s = self.screen
        pygame.draw.circle(s, (8, 10, 15), (cx, cy), r + 9)
        pygame.draw.circle(s, (70, 80, 105), (cx, cy), r + 9, 3)
        pygame.draw.circle(s, (17, 20, 29), (cx, cy), r)
        A0, SW = 135.0, 270.0

        def ang(v):
            return A0 + SW * clamp(v / vmax, 0.0, 1.0)

        if redline is not None:
            pts = [self._pt(cx, cy, r * 0.90, ang(redline + (vmax - redline) * i / 30)) for i in range(31)]
            pygame.draw.lines(s, (190, 28, 28), False, pts, 7)
        steps = int(round(vmax / minor))
        for i in range(steps + 1):
            val = i * minor
            is_major = abs(val / major - round(val / major)) < 1e-6
            col = (236, 240, 248) if is_major else (115, 124, 148)
            if redline is not None and val >= redline:
                col = (255, 95, 75)
            r1 = r * (0.78 if is_major else 0.86)
            pygame.draw.line(s, col, self._pt(cx, cy, r1, ang(val)), self._pt(cx, cy, r * 0.95, ang(val)),
                             3 if is_major else 1)
            if is_major:
                lx, ly = self._pt(cx, cy, r * 0.63, ang(val))
                self.text(str(int(round(val / divisor))), self.f_mid, self.TEXT, (lx, ly))
        self.text(title, self.f_tiny, self.DIM, (cx, cy - r * 0.32))
        a = ang(value)
        tip, tail = self._pt(cx, cy, r * 0.86, a), self._pt(cx, cy, r * 0.14, a + 180)
        p1, p2 = self._pt(cx, cy, 5, a + 90), self._pt(cx, cy, 5, a - 90)
        pygame.draw.polygon(s, (255, 85, 40), [tip, (cx + (p1[0] - cx), cy + (p1[1] - cy)), tail,
                                              (cx + (p2[0] - cx), cy + (p2[1] - cy))])
        pygame.draw.circle(s, (45, 52, 70), (cx, cy), 11)
        pygame.draw.circle(s, (255, 85, 40), (cx, cy), 5)

    def draw_gauges(self):
        self.panel(self.gauge_rect)
        cfg, v = self.cfg, self.vehicle
        tach_max = max(9000, int(math.ceil((cfg.redline + 800) / 1000.0)) * 1000)
        tcx, tcy, scx, scy, r = 800, 245, 1110, 245, 148
        self.draw_gauge(tcx, tcy, r, self.disp_rpm, tach_max, 1000, 500, 1000,
                        redline=cfg.redline, title="RPM x1000")
        self.text(f"{int(self.disp_rpm):d}", self.f_big, self.TEXT, (tcx, tcy + 62))
        self.text(self.tx.display(), self.f_huge, self.ORANGE, (tcx, tcy + 112))
        spd = v.speed * (3.6 if self.units == "kmh" else 2.23694)
        vmax = 400 if self.units == "kmh" else 250
        self.draw_gauge(scx, scy, r, spd, vmax, 50, 10, 1, title="SPEED")
        self.text(f"{int(spd):d}", self.f_huge, self.TEXT, (scx, scy + 70))
        self.text("km/h" if self.units == "kmh" else "mph", self.f_mid, self.DIM, (scx, scy + 112))

        # gear strip
        strip = pygame.Rect(670, 394, 570, 30)
        pygame.draw.rect(self.screen, (13, 16, 24), strip, border_radius=8)
        labels = ["P", "N"] + [str(i) for i in range(1, 8)]
        cw = strip.w / len(labels)
        cur = self.tx.display()
        for i, lab in enumerate(labels):
            cell = pygame.Rect(strip.x + i * cw, strip.y, cw, strip.h)
            if lab == cur:
                pygame.draw.rect(self.screen, self.ORANGE, cell.inflate(-6, -6), border_radius=6)
                self.text(lab, self.f_mid, (20, 20, 20), cell.center)
            else:
                self.text(lab, self.f_mid, self.DIM, cell.center)

    def draw_bars_and_scope(self):
        v, s = self.vehicle, self.screen
        for x, val, label, col in ((20, v.throttle, "THROTTLE", self.GREEN), (340, v.brake, "BRAKE", self.RED)):
            r = pygame.Rect(x, 498, 300, 22)
            pygame.draw.rect(s, (24, 28, 40), r, border_radius=6)
            pygame.draw.rect(s, col, (r.x, r.y, int(r.w * val), r.h), border_radius=6)
            pygame.draw.rect(s, self.BORDER, r, 1, border_radius=6)
            self.text(f"{label} {int(val * 100):d}%", self.f_tiny, (255, 255, 255), r.center)
        self.text("ENGINE SELECT", self.f_tiny, self.DIM, (20, 530), "topleft")

        # oscilloscope
        rect = pygame.Rect(650, 496, 610, 118)
        self.panel(rect)
        self.text("SYNTH OUTPUT", self.f_tiny, self.DIM, (rect.x + 12, rect.y + 8), "topleft")
        mid = rect.centery + 6
        pygame.draw.line(s, (45, 52, 72), (rect.x + 8, mid), (rect.right - 8, mid), 1)
        if self.synth.enabled:
            wave = self.synth.scope[::3]
            pts = [(rect.x + 10 + i * (rect.w - 20) / max(1, len(wave) - 1), mid - float(a) * 42)
                   for i, a in enumerate(wave)]
            if len(pts) > 1:
                pygame.draw.lines(s, (80, 220, 255), False, pts, 1)
        else:
            self.text("audio device unavailable", self.f_small, self.DIM, rect.center)

        cfg = self.cfg
        hz = cfg.pulse_hz(v.rpm)
        lines = [
            f"Pulse rate  Hz = (RPM/60) x (Cyl/2) = ({int(v.rpm)}/60) x ({cfg.cylinders}/2) = {hz:.0f} Hz",
            f"Redline {int(cfg.redline)} rpm   Idle {int(cfg.idle)} rpm   Peak torque {int(cfg.peak_torque)} Nm @ {int(cfg.peak_rpm)}",
            "W/Up throttle   S/Down brake   Space start/stop   1-7 engine   V variant",
            "D drive   N neutral   U units   M mute   Esc quit   (try: N + blip throttle, then D)",
        ]
        for i, ln in enumerate(lines):
            self.text(ln, self.f_small, self.TEXT if i < 2 else self.DIM, (656, 626 + i * 24), "topleft")

    def draw(self):
        self.screen.fill(self.BG)
        self.draw_header()
        self.draw_cylinders()
        self.draw_gauges()
        self.draw_bars_and_scope()
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            b.draw(self.screen, self.f_mid if b.hold or b.rect.w > 100 else self.f_small, mouse, self.held_ui)

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
                    elif k == pygame.K_u:
                        self.toggle_units()
                    elif k == pygame.K_m:
                        self.toggle_mute()
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
