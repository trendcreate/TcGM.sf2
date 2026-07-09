"""
Standard GM percussion map (bank 128 / preset 0), notes 35-81, synthesized
purely from summed sine partials (tonal hits) and sine noise-clusters
(brushed/metallic/scrape textures) -- no samples, no true noise generator.
"""
import numpy as np
from synth_engine import SR, additive_partials, noise_cluster, env_ad, rng_for, TWO_PI, _time_axis

TYPE_OF = {}
for n in (35, 36): TYPE_OF[n] = "kick"
for n in (38, 40): TYPE_OF[n] = "snare"
TYPE_OF[37] = "sidestick"
TYPE_OF[39] = "clap"
for n in (41, 43, 45, 47, 48, 50, 117, 118): TYPE_OF[n] = "tom"
for n in (42, 44): TYPE_OF[n] = "hat_closed"
TYPE_OF[46] = "hat_open"
for n in (49, 51, 55, 57, 59): TYPE_OF[n] = "cymbal"
TYPE_OF[52] = "cymbal_china"
TYPE_OF[53] = "ride_bell"
TYPE_OF[54] = "tambourine"
TYPE_OF[56] = "cowbell"
TYPE_OF[58] = "vibraslap"
for n in (60, 61): TYPE_OF[n] = "bongo"
for n in (62, 63, 64): TYPE_OF[n] = "conga"
for n in (65, 66): TYPE_OF[n] = "timbale"
for n in (67, 68): TYPE_OF[n] = "agogo"
TYPE_OF[69] = "cabasa"
TYPE_OF[70] = "maracas"
for n in (71, 72): TYPE_OF[n] = "whistle"
for n in (73, 74): TYPE_OF[n] = "guiro"
for n in (75, 76, 77): TYPE_OF[n] = "claves"
for n in (78, 79): TYPE_OF[n] = "cuica"
for n in (80, 81): TYPE_OF[n] = "triangle"


def _chirp(f_start, f_end, duration, decay_tau, amp=1.0, n_harm=1, seed=0, sr=SR):
    t = _time_axis(duration, sr)
    k = np.log(max(f_end, 1.0) / max(f_start, 1.0)) / max(duration, 1e-6)
    inst_f = f_start * np.exp(k * t)
    phase = TWO_PI * f_start * (np.exp(k * t) - 1.0) / k if abs(k) > 1e-9 else TWO_PI * f_start * t
    env = env_ad(t, 0.001, 0.0, decay_tau)
    out = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        out += (1.0 / h) * env * np.sin(h * phase)
    return amp * out


def _burst(center, bw, duration, decay_tau, n_partials=40, seed=0, tilt=0.0, attack=0.001):
    return noise_cluster(center, bw, n_partials, duration, decay_tau, seed=seed,
                          spectral_tilt=tilt, attack=attack, decay_to=0.0)


def _tonal_hit(freq, duration, decay_tau, n_harm=4, inharm=0.01, seed=0, sustain=0.0):
    partials = []
    for k in range(1, n_harm + 1):
        ratio = k * np.sqrt(1.0 + inharm * k * k)
        partials.append(dict(ratio=ratio, amp=1.0 / k, attack=0.001,
                              decay_to=sustain, decay_tau=decay_tau / (1 + 0.2 * (k - 1))))
    return additive_partials(freq, duration, partials, seed=seed)


def synthesize_drum(note, seed):
    typ = TYPE_OF[note]
    rng_seed = seed

    if typ == "kick":
        body = _chirp(110, 45, 0.35, 0.14, amp=1.0, n_harm=2, seed=rng_seed)
        click = _burst(1800, 0.9, 0.02, 0.006, n_partials=20, seed=rng_seed + 1)
        n = max(len(body), len(click))
        out = np.zeros(n); out[:len(body)] += body; out[:len(click)] += click * 0.35
        return out, 0.35, False

    if typ == "snare":
        noise = _burst(2200, 0.9, 0.30, 0.10, n_partials=60, seed=rng_seed, tilt=0.2)
        body = _tonal_hit(190, 0.25, 0.07, n_harm=3, inharm=0.02, seed=rng_seed + 1)
        n = max(len(noise), len(body))
        out = np.zeros(n); out[:len(noise)] += noise * 0.8; out[:len(body)] += body * 0.6
        return out, 0.30, False

    if typ == "sidestick":
        click = _burst(1200, 0.7, 0.06, 0.015, n_partials=25, seed=rng_seed)
        body = _tonal_hit(500, 0.06, 0.015, n_harm=2, inharm=0.01, seed=rng_seed + 1)
        n = max(len(click), len(body))
        out = np.zeros(n); out[:len(click)] += click * 0.5; out[:len(body)] += body * 0.5
        return out, 0.06, False

    if typ == "clap":
        n = int(SR * 0.28)
        out = np.zeros(n)
        for i, off in enumerate([0.0, 0.012, 0.026, 0.045]):
            b = _burst(1500, 0.8, 0.22, 0.09, n_partials=35, seed=rng_seed + i)
            s = int(off * SR)
            e = min(n, s + len(b))
            out[s:e] += b[: e - s] * 0.6
        return out, 0.28, False

    if typ == "tom":
        base = {41: 90, 43: 110, 45: 130, 47: 150, 48: 170, 50: 200, 117: 70, 118: 220}[note]
        body = _chirp(base * 1.5, base, 0.45, 0.25, amp=1.0, n_harm=3, seed=rng_seed)
        return body, 0.45, False

    if typ == "hat_closed":
        return _burst(9000, 0.9, 0.10, 0.03, n_partials=70, seed=rng_seed, tilt=0.3), 0.10, False

    if typ == "hat_open":
        return _burst(9000, 0.9, 0.9, 0.35, n_partials=80, seed=rng_seed, tilt=0.3), 0.9, False

    if typ == "cymbal":
        base = {49: 4500, 51: 3500, 55: 6000, 57: 4200, 59: 3200}[note]
        dur = {49: 1.3, 51: 1.1, 55: 0.8, 57: 1.3, 59: 1.1}[note]
        return _burst(base, 1.1, dur, dur * 0.4, n_partials=90, seed=rng_seed, tilt=0.15), dur, False

    if typ == "cymbal_china":
        return _burst(5200, 1.3, 1.0, 0.4, n_partials=100, seed=rng_seed, tilt=0.2), 1.0, False

    if typ == "ride_bell":
        tone = _tonal_hit(1100, 0.9, 0.45, n_harm=5, inharm=0.02, seed=rng_seed)
        shim = _burst(4000, 0.7, 0.9, 0.4, n_partials=35, seed=rng_seed + 1)
        n = max(len(tone), len(shim))
        out = np.zeros(n); out[:len(tone)] += tone * 0.7; out[:len(shim)] += shim * 0.3
        return out, 1.4, False

    if typ == "tambourine":
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        flutter = 0.6 + 0.4 * np.sin(TWO_PI * 22 * t)
        base = _burst(6500, 1.0, 0.5, 0.25, n_partials=60, seed=rng_seed, tilt=0.2)
        m = min(len(base), len(flutter))
        return base[:m] * flutter[:m], 0.5, False

    if typ == "cowbell":
        return _tonal_hit(560, 0.5, 0.3, n_harm=2, inharm=0.03, seed=rng_seed), 0.5, False

    if typ == "vibraslap":
        n = int(SR * 0.7)
        t = np.arange(n) / SR
        flutter = 0.5 + 0.5 * np.sin(TWO_PI * 28 * t) * np.exp(-t / 0.4)
        base = _burst(1800, 0.9, 0.7, 0.4, n_partials=50, seed=rng_seed, tilt=0.1)
        m = min(len(base), len(flutter))
        return base[:m] * flutter[:m], 0.7, False

    if typ == "bongo":
        base = 400 if note == 60 else 300
        return _tonal_hit(base, 0.22, 0.09, n_harm=3, inharm=0.015, seed=rng_seed), 0.22, False

    if typ == "conga":
        base = {62: 220, 63: 260, 64: 180}[note]
        return _tonal_hit(base, 0.28, 0.13, n_harm=3, inharm=0.012, seed=rng_seed), 0.28, False

    if typ == "timbale":
        base = 300 if note == 65 else 220
        return _chirp(base * 1.3, base, 0.35, 0.2, n_harm=3, seed=rng_seed), 0.35, False

    if typ == "agogo":
        base = 900 if note == 67 else 650
        return _tonal_hit(base, 0.4, 0.25, n_harm=3, inharm=0.025, seed=rng_seed), 0.4, False

    if typ == "cabasa":
        return _burst(7000, 1.0, 0.18, 0.06, n_partials=70, seed=rng_seed, tilt=0.25), 0.18, False

    if typ == "maracas":
        return _burst(7500, 1.0, 0.12, 0.04, n_partials=55, seed=rng_seed, tilt=0.25), 0.12, False

    if typ == "whistle":
        dur = 0.35 if note == 71 else 1.1
        t = _time_axis(dur)
        env = env_ad(t, 0.02, 0.9, dur * 0.6)
        tone = env * np.sin(TWO_PI * 2200 * t)
        breath = _burst(2200, 0.3, dur, dur, n_partials=15, seed=rng_seed, tilt=0.1) * 0.15
        n = min(len(tone), len(breath))
        return tone[:n] * 0.85 + breath[:n], dur, note == 72

    if typ == "guiro":
        n_strokes = 3 if note == 73 else 7
        dur = 0.25 if note == 73 else 0.7
        n = int(SR * dur)
        out = np.zeros(n)
        stroke_len = dur / n_strokes
        for i in range(n_strokes):
            b = _burst(3500, 0.8, stroke_len * 0.8, stroke_len * 0.25, n_partials=30, seed=rng_seed + i, tilt=0.2)
            s = int(i * stroke_len * SR)
            e = min(n, s + len(b))
            out[s:e] += b[: e - s]
        return out, dur, False

    if typ == "claves":
        base = {75: 2500, 76: 1600, 77: 1100}[note]
        return _tonal_hit(base, 0.12, 0.04, n_harm=2, inharm=0.01, seed=rng_seed), 0.12, False

    if typ == "cuica":
        dur = 0.3 if note == 78 else 0.6
        body = _chirp(500, 250, dur, dur * 0.5, n_harm=2, seed=rng_seed)
        return body, dur, False

    if typ == "triangle":
        dur = 0.3 if note == 80 else 1.0
        tone = _tonal_hit(2800, dur, dur * 0.5, n_harm=6, inharm=0.04, seed=rng_seed, sustain=0.0)
        return tone, dur, note == 81

    raise KeyError(note)
