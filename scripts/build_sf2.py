import sys
import time
import argparse
import numpy as np

from synth_engine import SR, midi_to_freq, normalize, find_loop_points, to_int16, additive_partials
from timbre_recipes import CATEGORY_DEFAULTS, apply_keyword_overrides, build_partials
from gm_data import GM_NAMES, CATEGORY_RANGES, category_for, DRUM_NOTES
from drum_recipes import synthesize_drum
from sf2_writer import SoundFontBuilder, Sample, Instrument, Preset, GEN, timecents, cb_atten_from_sustain, hz_to_cents

CATEGORY_ATTEN_CB = {
    "piano": 0, "chrom_perc": 30, "organ": 20, "guitar": 30, "bass": 0,
    "strings": 40, "ensemble": 30, "brass": 10, "reed": 30, "pipe": 60,
    "synth_lead": 20, "synth_pad": 60, "synth_fx": 80, "ethnic": 40,
    "percussive": 20, "sound_fx": 60,
}

# Filter (lowpass+resonance) + modulation-LFO movement per category, so
# sustained timbres are not perfectly static (this is what the SF2/EMU8000
# filter+LFO block is for; a flat additive spectrum alone sounds "dead").
# cutoff_hz/q_cb: static filter starting point. env_amt_cents: how much the
# (volume-envelope-synced) mod envelope opens the filter on attack and closes
# it on decay/release. lfo_rate_hz/lfo_to_filter/lfo_to_pitch/lfo_to_vol_cb:
# slow modulation-LFO wobble depth. lfo_delay_s: LFO fade-in delay.
FILTER_LFO = {
    "synth_lead": dict(cutoff=3200, q=180, env_amt=3600, lfo_rate=4.5, lfo_to_filter=500, lfo_to_pitch=4, lfo_to_vol=0, lfo_delay=0.15),
    "synth_pad":  dict(cutoff=1800, q=120, env_amt=4200, lfo_rate=2.8, lfo_to_filter=650, lfo_to_pitch=6, lfo_to_vol=15, lfo_delay=0.4),
    "synth_fx":   dict(cutoff=2200, q=220, env_amt=5000, lfo_rate=3.6, lfo_to_filter=800, lfo_to_pitch=8, lfo_to_vol=20, lfo_delay=0.2),
    "organ":      dict(cutoff=9000, q=60,  env_amt=0,    lfo_rate=5.5, lfo_to_filter=150, lfo_to_pitch=3, lfo_to_vol=8,  lfo_delay=0.3),
    "brass":      dict(cutoff=6000, q=90,  env_amt=1800, lfo_rate=4.8, lfo_to_filter=200, lfo_to_pitch=4, lfo_to_vol=0,  lfo_delay=0.25),
    "reed":       dict(cutoff=5500, q=90,  env_amt=1500, lfo_rate=4.8, lfo_to_filter=180, lfo_to_pitch=5, lfo_to_vol=0,  lfo_delay=0.25),
    "strings":    dict(cutoff=5000, q=60,  env_amt=1200, lfo_rate=4.2, lfo_to_filter=150, lfo_to_pitch=4, lfo_to_vol=0,  lfo_delay=0.3),
    "ensemble":   dict(cutoff=5000, q=60,  env_amt=1200, lfo_rate=3.8, lfo_to_filter=180, lfo_to_pitch=5, lfo_to_vol=6,  lfo_delay=0.3),
    "pipe":       dict(cutoff=7000, q=40,  env_amt=800,  lfo_rate=4.5, lfo_to_filter=100, lfo_to_pitch=5, lfo_to_vol=0,  lfo_delay=0.3),
}
FILTER_LFO_DEFAULT = dict(cutoff=13500, q=0, env_amt=0, lfo_rate=5.0, lfo_to_filter=0, lfo_to_pitch=0, lfo_to_vol=0, lfo_delay=0.3)


def filter_lfo_params(cat, name):
    p = dict(FILTER_LFO.get(cat, FILTER_LFO_DEFAULT))
    if "synth" in name.lower():
        p["cutoff"] = p["cutoff"] * 0.85
        p["q"] = p["q"] + 60
        p["env_amt"] = p["env_amt"] * 1.4 if p["env_amt"] else 2600
        p["lfo_to_filter"] = p["lfo_to_filter"] * 1.6 if p["lfo_to_filter"] else 350
        p["lfo_to_pitch"] = p["lfo_to_pitch"] + 4
        p["lfo_rate"] = max(p["lfo_rate"], 3.5)
    return p

DRUM_PAN = {}
for _n in DRUM_NOTES:
    DRUM_PAN[_n] = 0
for _n in (42, 44, 46, 49, 51, 55, 57, 59, 52, 53): DRUM_PAN[_n] = -20
for _n in (41, 45, 47): DRUM_PAN[_n] = -35
for _n in (43, 48, 50): DRUM_PAN[_n] = 25
for _n in (60, 62): DRUM_PAN[_n] = 30
for _n in (61, 64): DRUM_PAN[_n] = -30


def build_instrument_sample(sfb, program0, note, params, cat):
    freq = midi_to_freq(note)
    partials = build_partials(freq, note, params, seed=program0 * 1000 + note)
    buf = additive_partials(freq, params["duration"], partials, seed=program0 * 1000 + note)
    if params["noise_mix"] > 0:
        from synth_engine import noise_cluster
        noise = noise_cluster(freq * 2.0, 1.5, 24, params["duration"], params["duration"] * 0.4,
                               seed=program0 * 1000 + note + 500, spectral_tilt=-0.3)
        buf = buf * (1.0 - params["noise_mix"]) + noise * params["noise_mix"]
    buf = normalize(buf, peak=0.9)

    loopable = params["loopable"]
    if loopable:
        ls, le = find_loop_points(buf, freq, sr=SR)
    else:
        ls, le = 0, len(buf)

    pcm = to_int16(buf)
    name = f"{GM_NAMES[program0][:14]}_{note}"
    sample = Sample(name, pcm, SR, root_key=note, loop_start=ls, loop_end=le)
    idx = sfb.add_sample(sample)
    return idx


def build_melodic_instruments(sfb, note_limit=None):
    inst_indices = {}
    for program0, name in enumerate(GM_NAMES):
        cat, klo, khi = category_for(program0)
        if note_limit:
            khi = min(khi, klo + note_limit - 1)
        base = CATEGORY_DEFAULTS[cat]
        params = apply_keyword_overrides(cat, base, name, program0)

        inst = Instrument(f"{name[:19]}")
        atten = CATEGORY_ATTEN_CB.get(cat, 0)
        sustain_cb = cb_atten_from_sustain(params["sustain"])
        release_s = 0.05 if not params["loopable"] else 0.3
        fl = filter_lfo_params(cat, name)
        global_gens = [
            (GEN["attackVolEnv"], timecents(params["attack"])),
            (GEN["holdVolEnv"], timecents(0.001)),
            (GEN["decayVolEnv"], timecents(params["decay_tau"])),
            (GEN["sustainVolEnv"], sustain_cb),
            (GEN["releaseVolEnv"], timecents(release_s)),
            (GEN["pan"], 0),
            (GEN["reverbEffectsSend"], int(params["reverb"])),
            (GEN["chorusEffectsSend"], int(params["chorus"])),
            (GEN["initialAttenuation"], atten),
            (GEN["scaleTuning"], 100),
            (GEN["sampleModes"], 1 if params["loopable"] else 0),
            # filter + envelope/LFO movement so timbre isn't static over the sustain
            (GEN["initialFilterFc"], hz_to_cents(fl["cutoff"], lo=1500, hi=13500)),
            (GEN["initialFilterQ"], int(max(0, min(960, fl["q"])))),
            (GEN["modEnvToFilterFc"], int(fl["env_amt"])),
            (GEN["delayModEnv"], timecents(0.001)),
            (GEN["attackModEnv"], timecents(params["attack"])),
            (GEN["holdModEnv"], timecents(0.001)),
            (GEN["decayModEnv"], timecents(params["decay_tau"])),
            (GEN["sustainModEnv"], 700 if fl["env_amt"] else 1000),
            (GEN["releaseModEnv"], timecents(release_s)),
            (GEN["delayModLFO"], timecents(fl["lfo_delay"])),
            (GEN["freqModLFO"], hz_to_cents(fl["lfo_rate"], lo=-12000, hi=8000)),
            (GEN["modLfoToFilterFc"], int(fl["lfo_to_filter"])),
            (GEN["modLfoToPitch"], int(fl["lfo_to_pitch"])),
            (GEN["modLfoToVolume"], int(fl["lfo_to_vol"])),
        ]
        inst.add_global(global_gens)

        sample_by_note = {}
        for note in range(klo, khi + 1):
            idx = build_instrument_sample(sfb, program0, note, params, cat)
            sample_by_note[note] = idx
            inst.add_zone([(GEN["overridingRootKey"], note)], idx, key_lo=note, key_hi=note)

        if klo > 0:
            inst.add_zone([(GEN["overridingRootKey"], klo)], sample_by_note[klo], key_lo=0, key_hi=klo - 1)
        if khi < 127:
            inst.add_zone([(GEN["overridingRootKey"], khi)], sample_by_note[khi], key_lo=khi + 1, key_hi=127)

        inst_idx = sfb.add_instrument(inst)
        inst_indices[program0] = inst_idx

        preset = Preset(name[:19], program0, 0)
        preset.instrument_index = inst_idx
        sfb.add_preset(preset)

        print(f"  [{program0:3d}] {name:28s} cat={cat:11s} notes={khi - klo + 1:3d}", flush=True)
    return inst_indices


def build_drum_kit(sfb, note_limit=None):
    inst = Instrument("Standard Drum Kit")
    inst.add_global([
        (GEN["attackVolEnv"], timecents(0.0005)),
        (GEN["sustainVolEnv"], 0),
        (GEN["releaseVolEnv"], timecents(0.3)),
        (GEN["reverbEffectsSend"], 150),
        (GEN["chorusEffectsSend"], 30),
        (GEN["scaleTuning"], 0),
    ])

    notes = sorted(DRUM_NOTES.keys())
    if note_limit:
        notes = notes[:note_limit]
    for note in notes:
        buf, dur, loop_flag = synthesize_drum(note, seed=note * 17 + 3)
        buf = normalize(buf, peak=0.9)
        approx_freq = 400.0
        if loop_flag:
            ls, le = find_loop_points(buf, approx_freq, sr=SR)
        else:
            ls, le = 0, len(buf)
        pcm = to_int16(buf)
        sample = Sample(f"Drum_{note}_{DRUM_NOTES[note][:10]}", pcm, SR, root_key=note,
                         loop_start=ls, loop_end=le)
        idx = sfb.add_sample(sample)
        pan = DRUM_PAN.get(note, 0)
        gens = [
            (GEN["holdVolEnv"], timecents(max(dur, 0.02))),
            (GEN["pan"], pan),
            (GEN["sampleModes"], 1 if loop_flag else 0),
        ]
        inst.add_zone(gens, idx, key_lo=note, key_hi=note)
        print(f"  drum {note:3d} {DRUM_NOTES[note]:20s} dur={dur:.2f}s", flush=True)

    inst_idx = sfb.add_instrument(inst)
    preset = Preset("Standard", 0, 128)
    preset.instrument_index = inst_idx
    sfb.add_preset(preset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-limit", type=int, default=None,
                     help="cap notes per instrument (quick test builds)")
    ap.add_argument("--out", default="TcGM.sf2")
    args = ap.parse_args()

    t0 = time.time()
    sfb = SoundFontBuilder(bank_name="TcGM")

    print("Building 128 GM melodic instruments...")
    build_melodic_instruments(sfb, note_limit=args.note_limit)

    print("Building standard GM drum kit...")
    build_drum_kit(sfb, note_limit=args.note_limit)

    print(f"Samples: {len(sfb.samples)}  Instruments: {len(sfb.instruments)}  Presets: {len(sfb.presets)}")
    print("Serializing SF2...")
    data = sfb.build()
    with open(args.out, "wb") as f:
        f.write(data)
    dt = time.time() - t0
    print(f"Wrote {args.out}  ({len(data)/1e6:.1f} MB) in {dt:.1f}s")


if __name__ == "__main__":
    main()
