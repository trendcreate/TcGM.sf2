"""
Category-level synthesis recipes -> per-note partial lists for the additive
(sine-summation) engine. Expression (vibrato, tremolo, pan, reverb/chorus
send, pitch-bend range) is handled by SF2 generators/modulators, not baked
into the samples, so sample content here is the *static* timbre only.
"""
import numpy as np

CATEGORY_DEFAULTS = {
    # key: n_harm, rolloff, odd_only, inharm(B), attack, decay_tau, sustain, noise_mix, noise_bw,
    #      duration, loopable, pan, reverb, chorus, formants
    "piano":        dict(n_harm=18, rolloff=1.15, odd_only=False, inharm=0.00035, attack=0.004,
                          decay_tau=1.1, sustain=0.0, noise_mix=0.03, noise_bw=0.35,
                          duration=1.1, loopable=False, pan=64, reverb=180, chorus=40, formants=[]),
    "chrom_perc":   dict(n_harm=10, rolloff=1.6, odd_only=False, inharm=0.006, attack=0.002,
                          decay_tau=1.1, sustain=0.0, noise_mix=0.02, noise_bw=0.6,
                          duration=1.2, loopable=False, pan=64, reverb=220, chorus=60, formants=[]),
    "organ":        dict(n_harm=9, rolloff=0.85, odd_only=False, inharm=0.0, attack=0.012,
                          decay_tau=0.05, sustain=1.0, noise_mix=0.0, noise_bw=0.5,
                          duration=0.55, loopable=True, pan=64, reverb=160, chorus=120, formants=[]),
    "guitar":       dict(n_harm=14, rolloff=1.2, odd_only=False, inharm=0.0002, attack=0.003,
                          decay_tau=0.65, sustain=0.0, noise_mix=0.05, noise_bw=0.4,
                          duration=0.8, loopable=False, pan=64, reverb=140, chorus=90, formants=[]),
    "bass":         dict(n_harm=10, rolloff=1.25, odd_only=False, inharm=0.0001, attack=0.004,
                          decay_tau=0.7, sustain=0.15, noise_mix=0.03, noise_bw=0.4,
                          duration=0.8, loopable=False, pan=64, reverb=90, chorus=40, formants=[]),
    "strings":      dict(n_harm=14, rolloff=1.0, odd_only=False, inharm=0.0, attack=0.09,
                          decay_tau=0.4, sustain=1.0, noise_mix=0.02, noise_bw=0.3,
                          duration=0.9, loopable=True, pan=64, reverb=200, chorus=100, formants=[]),
    "ensemble":     dict(n_harm=16, rolloff=1.0, odd_only=False, inharm=0.0, attack=0.12,
                          decay_tau=0.4, sustain=1.0, noise_mix=0.03, noise_bw=0.3,
                          duration=0.9, loopable=True, pan=64, reverb=230, chorus=160, formants=[]),
    "brass":        dict(n_harm=12, rolloff=0.95, odd_only=False, inharm=0.0, attack=0.045,
                          decay_tau=0.3, sustain=1.0, noise_mix=0.03, noise_bw=0.5,
                          duration=0.8, loopable=True, pan=64, reverb=170, chorus=70,
                          formants=[(1200, 800, 1.6)]),
    "reed":         dict(n_harm=13, rolloff=1.05, odd_only=False, inharm=0.0, attack=0.05,
                          decay_tau=0.3, sustain=1.0, noise_mix=0.04, noise_bw=0.6,
                          duration=0.8, loopable=True, pan=64, reverb=160, chorus=70,
                          formants=[(900, 700, 1.5)]),
    "pipe":         dict(n_harm=6, rolloff=1.4, odd_only=False, inharm=0.0, attack=0.06,
                          decay_tau=0.3, sustain=1.0, noise_mix=0.10, noise_bw=1.2,
                          duration=0.7, loopable=True, pan=64, reverb=200, chorus=60, formants=[]),
    "synth_lead":   dict(n_harm=16, rolloff=1.0, odd_only=False, inharm=0.0, attack=0.008,
                          decay_tau=0.2, sustain=1.0, noise_mix=0.0, noise_bw=0.4,
                          duration=0.5, loopable=True, pan=64, reverb=110, chorus=90, formants=[]),
    "synth_pad":    dict(n_harm=12, rolloff=1.1, odd_only=False, inharm=0.0, attack=0.35,
                          decay_tau=0.5, sustain=1.0, noise_mix=0.02, noise_bw=0.4,
                          duration=1.0, loopable=True, pan=64, reverb=230, chorus=150, formants=[]),
    "synth_fx":     dict(n_harm=10, rolloff=1.1, odd_only=False, inharm=0.003, attack=0.25,
                          decay_tau=0.6, sustain=0.6, noise_mix=0.25, noise_bw=1.0,
                          duration=1.0, loopable=True, pan=64, reverb=250, chorus=170, formants=[]),
    "ethnic":       dict(n_harm=13, rolloff=1.2, odd_only=False, inharm=0.0015, attack=0.003,
                          decay_tau=0.55, sustain=0.05, noise_mix=0.06, noise_bw=0.5,
                          duration=0.7, loopable=False, pan=64, reverb=150, chorus=70, formants=[]),
    "percussive":   dict(n_harm=9, rolloff=1.3, odd_only=False, inharm=0.01, attack=0.002,
                          decay_tau=0.4, sustain=0.0, noise_mix=0.15, noise_bw=0.9,
                          duration=0.65, loopable=False, pan=64, reverb=160, chorus=50, formants=[]),
    "sound_fx":     dict(n_harm=4, rolloff=1.0, odd_only=False, inharm=0.0, attack=0.05,
                          decay_tau=0.45, sustain=0.1, noise_mix=0.7, noise_bw=1.4,
                          duration=0.8, loopable=False, pan=64, reverb=140, chorus=40, formants=[]),
}


def _kw(name):
    n = name.lower()
    return n


def apply_keyword_overrides(cat, params, name, program0):
    n = _kw(name)
    p = dict(params)
    p = {**p, "n_harm": p["n_harm"], "formants": list(p["formants"])}

    def bump(key, factor=None, add=None):
        if factor is not None:
            p[key] = p[key] * factor
        if add is not None:
            p[key] = p[key] + add

    if "bright" in n:
        bump("rolloff", 0.8); bump("n_harm", 1.15)
    if "honky" in n:
        p["inharm"] += 0.002
    if "electric" in n and "piano" in n:
        bump("rolloff", 1.05); p["formants"].append((1500, 900, 1.3))
    if "harpsichord" in n or "clavinet" in n:
        bump("rolloff", 0.7); p["attack"] = min(p["attack"], 0.002); p["decay_tau"] *= 0.6
    if "muted" in n or "mute" in n:
        bump("n_harm", 0.5); bump("rolloff", 1.4)
    if "overdriven" in n or "distortion" in n:
        bump("n_harm", 1.6); bump("rolloff", 0.55); p["noise_mix"] += 0.08
    if "jazz" in n:
        bump("rolloff", 1.3)
    if "harmonics" in n:
        p["inharm"] += 0.0; bump("n_harm", 0.4)
    if "slap" in n or "pick" in n:
        p["attack"] = min(p["attack"], 0.002); p["noise_mix"] += 0.06
    if "fretless" in n:
        bump("rolloff", 1.15)
    if "pizzicato" in n:
        p["attack"] = 0.002; p["decay_tau"] = 0.25; p["duration"] = 0.6
        p["sustain"] = 0.0; p["loopable"] = False
    if "harp" in n and "harpsichord" not in n:
        p["attack"] = 0.004; p["decay_tau"] = 0.9; p["duration"] = 1.2
        p["sustain"] = 0.0; p["loopable"] = False; p["noise_mix"] += 0.02
    if "tremolo" in n:
        p["noise_mix"] += 0.02
    if "choir" in n or "voice" in n or "aahs" in n or "oohs" in n:
        p["formants"] += [(700, 150, 2.2), (1200, 250, 1.6)]
        p["noise_mix"] += 0.05
    if "orchestra hit" in n:
        p["attack"] = 0.001; p["decay_tau"] = 0.25; p["duration"] = 0.6; p["loopable"] = False
        bump("n_harm", 1.4)
    if "timpani" in n:
        p["inharm"] += 0.02; p["attack"] = 0.002; p["decay_tau"] = 0.7
        p["duration"] = 1.0; p["sustain"] = 0.0; p["loopable"] = False
    if "french horn" in n:
        p["formants"] = [(500, 400, 1.4)]
    if "sax" in n:
        p["formants"] = [(800 + 100 * ("bari" in n) * -1, 600, 1.5)]
    if "shakuhachi" in n or "whistle" in n or "ocarina" in n or "bottle" in n:
        p["noise_mix"] += 0.08; bump("n_harm", 0.6)
    if "calliope" in n:
        bump("rolloff", 1.3)
    if "chiff" in n:
        p["noise_mix"] += 0.15; p["attack"] = 0.01
    if "charang" in n:
        p["inharm"] += 0.0008; bump("n_harm", 1.2)
    if "fifths" in n:
        p["formants"] = []
    if "metallic" in n or "bell" in n or "steel drum" in n:
        p["inharm"] += 0.02
    if "sitar" in n:
        p["inharm"] += 0.0025; p["decay_tau"] *= 1.4
    if "banjo" in n:
        p["inharm"] += 0.0015; p["decay_tau"] *= 0.6
    if "koto" in n or "shamisen" in n:
        p["inharm"] += 0.001
    if "kalimba" in n:
        p["inharm"] += 0.004; p["decay_tau"] *= 0.7
    if "bagpipe" in n or "shanai" in n:
        p["sustain"] = 1.0; p["loopable"] = True; p["noise_mix"] += 0.06
    if "reverse cymbal" in n:
        p["reverse"] = True
    if "agogo" in n or "woodblock" in n or "wood block" in n or "cowbell" in n:
        p["inharm"] += 0.008
    if "taiko" in n or "tom" in n:
        p["inharm"] += 0.0015; p["decay_tau"] *= 1.2
    if "synth drum" in n:
        p["inharm"] = 0.0
    return p


def key_scaled_rolloff(base_rolloff, note):
    # brighter (lower rolloff exponent -> more high harmonics) for low notes,
    # duller for very high notes, matching natural instrument behaviour.
    return base_rolloff + max(0.0, (note - 72)) * 0.01 - max(0.0, (60 - note)) * 0.004


def build_partials(freq, note, params, seed):
    n_harm = max(1, int(round(params["n_harm"])))
    rolloff = key_scaled_rolloff(params["rolloff"], note)
    inharm = params["inharm"]
    attack = params["attack"]
    decay_tau = params["decay_tau"]
    sustain = params["sustain"]

    partials = []
    for k in range(1, n_harm + 1):
        ratio = k * np.sqrt(1.0 + inharm * k * k)
        amp = 1.0 / (k ** rolloff)
        per_harm_tau = decay_tau / (1.0 + 0.15 * (k - 1))
        partials.append(dict(
            ratio=ratio, amp=amp, attack=attack, decay_to=sustain,
            decay_tau=per_harm_tau,
        ))
    # subtle 2-voice detune for chorus-like richness on sustained pads/strings/ensemble
    if params.get("n_harm", 0) >= 12 and params["loopable"]:
        extra = []
        for p in partials[:8]:
            extra.append(dict(p, detune_cents=+6.0, amp=p["amp"] * 0.55))
            extra.append(dict(p, detune_cents=-6.0, amp=p["amp"] * 0.55))
        partials += extra
    return partials
