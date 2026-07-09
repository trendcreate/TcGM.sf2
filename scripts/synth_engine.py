"""
TcGM.sf2 additive (sine-summation) synthesis engine.

Every sound in this soundfont is built exclusively from summed sine
oscillators (pure partials). No recorded/sampled waveforms and no
noise generators are used - "noisy" timbres (hats, snares, breath,
etc.) are approximated by clusters of many detuned/inharmonic sine
partials (a sinusoidal-modeling approach), which is still 100%
additive sine synthesis.
"""
import numpy as np

SR = 44100
TWO_PI = 2.0 * np.pi


def midi_to_freq(note, cents=0.0):
    return 440.0 * 2.0 ** ((note - 69 + cents / 100.0) / 12.0)


def _time_axis(duration, sr=SR):
    n = max(1, int(round(duration * sr)))
    return np.arange(n, dtype=np.float64) / sr


def env_ad(t, attack, decay_to, decay_tau):
    """Simple attack + exponential decay-to-sustain envelope, 0..1 domain."""
    a = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    a = a * a * (3 - 2 * a)  # smoothstep attack (click-free)
    decay = decay_to + (1.0 - decay_to) * np.exp(-np.maximum(t - attack, 0.0) / max(decay_tau, 1e-6))
    return a * decay


def rng_for(seed):
    return np.random.default_rng(seed)


def additive_partials(freq, duration, partials, sr=SR, seed=0):
    """
    partials: list of dicts, each with:
      ratio        - frequency ratio to fundamental (float, may be inharmonic)
      amp          - peak linear amplitude (0..1 scale, will be normalised later)
      attack       - seconds
      decay_to     - sustain level fraction of peak (0..1)
      decay_tau    - seconds, exponential time constant toward decay_to
      detune_cents - static detune in cents (for chorus-like richness)
      phase        - optional starting phase (radians), random if omitted
    Returns float64 mono buffer (not yet normalised).
    """
    t = _time_axis(duration, sr)
    rng = rng_for(seed)
    out = np.zeros_like(t)
    for p in partials:
        ratio = p["ratio"]
        f = freq * ratio * (2.0 ** (p.get("detune_cents", 0.0) / 1200.0))
        if f <= 0 or f >= sr * 0.5:
            continue
        phase = p.get("phase", None)
        if phase is None:
            phase = rng.uniform(0, TWO_PI)
        env = env_ad(t, p.get("attack", 0.005), p.get("decay_to", 1.0), p.get("decay_tau", 1.0))
        out += p["amp"] * env * np.sin(TWO_PI * f * t + phase)
    return out


def noise_cluster(center_freq, bandwidth_ratio, n_partials, duration, decay_tau,
                   sr=SR, seed=0, spectral_tilt=0.0, attack=0.001, decay_to=0.0):
    """
    Approximate a noisy/inharmonic burst (hats, snare body, breath, cymbals...)
    using many independent sine partials with randomised frequency, phase and
    per-partial decay -- a purely additive stand-in for a noise generator.
    """
    rng = rng_for(seed)
    t = _time_axis(duration, sr)
    out = np.zeros_like(t)
    lo = center_freq * (1.0 - bandwidth_ratio)
    hi = center_freq * (1.0 + bandwidth_ratio)
    lo = max(lo, 20.0)
    freqs = rng.uniform(lo, hi, n_partials)
    phases = rng.uniform(0, TWO_PI, n_partials)
    taus = decay_tau * rng.uniform(0.5, 1.5, n_partials)
    for f, ph, tau in zip(freqs, phases, taus):
        tilt = (f / center_freq) ** spectral_tilt
        env = env_ad(t, attack, decay_to, tau)
        out += tilt * env * np.sin(TWO_PI * f * t + ph)
    if n_partials > 0:
        out /= np.sqrt(n_partials)
    return out


def apply_formants(freq, partials, formant_bands):
    """
    formant_bands: list of (center_hz, bandwidth_hz, gain) -- purely a static
    amplitude-weighting of the additive partials (a spectral envelope), so it
    stays inside "sine-summation only" (no time-domain filters involved).
    Mutates and returns partials.
    """
    for p in partials:
        f = freq * p["ratio"]
        gain = 1.0
        for c, bw, g in formant_bands:
            gain *= 1.0 + (g - 1.0) * np.exp(-0.5 * ((f - c) / bw) ** 2)
        p["amp"] *= gain
    return partials


def normalize(buf, peak=0.92):
    m = np.max(np.abs(buf)) if buf.size else 0.0
    if m < 1e-9:
        return buf
    return buf * (peak / m)


def find_loop_points(buf, freq, sr=SR, min_loop_ms=30.0, search_ms=15.0):
    """
    Choose a loop start/end near the end of a sustaining buffer, snapped to
    zero-crossings (rising edge) spaced by an integer number of fundamental
    periods, to keep the loop click-free and pitch-accurate.
    """
    n = len(buf)
    period = sr / max(freq, 1.0)
    periods_needed = max(1, int(round((min_loop_ms / 1000.0 * sr) / period)))
    loop_len = int(round(period * periods_needed))
    loop_len = max(loop_len, 8)
    if loop_len * 2 >= n:
        return 0, max(n - 1, 1)

    end_target = n - int(sr * 0.02)
    start_target = end_target - loop_len
    if start_target < 0:
        start_target = 0
        end_target = min(n - 1, loop_len)

    def snap_rising_zero(idx, window):
        lo = max(1, idx - window)
        hi = min(n - 1, idx + window)
        best = idx
        best_d = None
        for i in range(lo, hi):
            if buf[i - 1] <= 0.0 < buf[i]:
                d = abs(i - idx)
                if best_d is None or d < best_d:
                    best = i
                    best_d = d
        return best

    win = max(4, int(sr * search_ms / 1000.0))
    ls = snap_rising_zero(start_target, win)
    le = ls + loop_len
    le = min(le, n - 1)
    return ls, le


def to_int16(buf):
    buf = np.clip(buf, -1.0, 1.0)
    return (buf * 32767.0).astype("<i2")
