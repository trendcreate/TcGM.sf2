"""
Minimal hand-rolled SoundFont 2.01 (RIFF 'sfbk') writer -- just enough of
the spec to build TcGM.sf2: INFO / sdta / pdta chunks, with phdr/pbag/pgen,
inst/ibag/igen and shdr records. Modulators are left to the SF2 "default
modulator" set (CC1 vibrato depth, CC7/CC11 volume/expression, CC10 pan,
CC91/CC93 reverb/chorus send, pitch-wheel) which every compliant SF2 player
applies automatically, so pmod/imod are written empty (terminal record only).
"""
import struct

SR = 44100

GEN = dict(
    startAddrsOffset=0, endAddrsOffset=1, startloopAddrsOffset=2, endloopAddrsOffset=3,
    modLfoToPitch=5, vibLfoToPitch=6, modEnvToPitch=7,
    initialFilterFc=8, initialFilterQ=9, modLfoToFilterFc=10, modEnvToFilterFc=11,
    chorusEffectsSend=15, reverbEffectsSend=16, pan=17,
    modLfoToVolume=13,
    delayModLFO=21, freqModLFO=22, delayVibLFO=23, freqVibLFO=24,
    delayModEnv=25, attackModEnv=26, holdModEnv=27, decayModEnv=28, sustainModEnv=29, releaseModEnv=30,
    delayVolEnv=33, attackVolEnv=34, holdVolEnv=35, decayVolEnv=36, sustainVolEnv=37, releaseVolEnv=38,
    instrument=41, keyRange=43, velRange=44,
    initialAttenuation=48, coarseTune=51, fineTune=52, sampleID=53, sampleModes=54,
    scaleTuning=56, exclusiveClass=57, overridingRootKey=58,
)


def _pad2(b):
    return b + b"\x00" if len(b) % 2 else b


def _chunk(fourcc, data):
    assert len(fourcc) == 4
    return fourcc.encode("ascii") + struct.pack("<I", len(data)) + _pad2(data)


def _list(fourcc, subchunks):
    body = fourcc.encode("ascii") + b"".join(subchunks)
    return _chunk("LIST", body)


def _cstr(s, size):
    b = s.encode("ascii", errors="replace")[: size - 1]
    return b + b"\x00" * (size - len(b))


def _info_chunk(fourcc, s):
    """INFO string subchunk with an even declared data size (avoids relying on
    RIFF pad-byte handling for odd-length chunks, which some SF2 loaders --
    e.g. BASS/BASSMIDI -- fail to skip correctly)."""
    size = len(s) + 1
    if size % 2:
        size += 1
    return _chunk(fourcc, _cstr(s, size))


def gen_record(oper, amount):
    return struct.pack("<Hh", oper, amount)


def gen_record_range(oper, lo, hi):
    return struct.pack("<HBB", oper, lo, hi)


def timecents(seconds):
    import math
    if seconds <= 0:
        return -32768
    v = int(round(1200.0 * math.log2(seconds)))
    return max(-32768, min(32767, v))


def hz_to_cents(hz, lo=None, hi=None):
    import math
    v = int(round(1200.0 * math.log2(max(hz, 1e-6) / 8.176)))
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def cb_atten_from_sustain(level):
    import math
    level = max(0.0, min(1.0, level))
    if level <= 0.0001:
        return 1000
    v = int(round(-200.0 * math.log10(level)))
    return max(0, min(1000, v))


class Sample:
    __slots__ = ("name", "pcm", "sample_rate", "root_key", "pitch_correction",
                 "loop_start", "loop_end", "start", "end", "startloop", "endloop")

    def __init__(self, name, pcm_int16, sample_rate, root_key, loop_start=None, loop_end=None,
                 pitch_correction=0):
        self.name = name
        self.pcm = pcm_int16
        self.sample_rate = sample_rate
        self.root_key = root_key
        self.pitch_correction = pitch_correction
        self.loop_start = loop_start if loop_start is not None else 0
        self.loop_end = loop_end if loop_end is not None else len(pcm_int16)


class InstrumentZone:
    def __init__(self, gens, sample_index=None):
        self.gens = list(gens)  # list of (oper, amount) or ("range", oper, lo, hi)
        self.sample_index = sample_index  # None => global zone


class Instrument:
    def __init__(self, name):
        self.name = name
        self.global_gens = []
        self.zones = []  # list of InstrumentZone

    def add_global(self, gens):
        self.global_gens = list(gens)

    def add_zone(self, gens, sample_index, key_lo=None, key_hi=None):
        z = []
        if key_lo is not None:
            z.append(("range", GEN["keyRange"], key_lo, key_hi))
        z += list(gens)
        self.zones.append(InstrumentZone(z, sample_index))


class Preset:
    def __init__(self, name, program, bank):
        self.name = name
        self.program = program
        self.bank = bank
        self.instrument_index = None
        self.global_gens = []


class SoundFontBuilder:
    def __init__(self, bank_name="TcGM"):
        self.bank_name = bank_name
        self.samples = []
        self.instruments = []
        self.presets = []

    def add_sample(self, sample: Sample) -> int:
        self.samples.append(sample)
        return len(self.samples) - 1

    def add_instrument(self, inst: Instrument) -> int:
        self.instruments.append(inst)
        return len(self.instruments) - 1

    def add_preset(self, preset: Preset):
        self.presets.append(preset)

    # ---- chunk builders ----
    def _build_sdta(self):
        parts = []
        offset = 0
        for s in self.samples:
            s.start = offset
            n = len(s.pcm)
            s.end = offset + n
            s.startloop = offset + s.loop_start
            s.endloop = offset + s.loop_end
            pad = b"\x00\x00" * 46
            parts.append(s.pcm.tobytes())
            parts.append(pad)
            offset += n + 46
        return _list("sdta", [_chunk("smpl", b"".join(parts))])

    def _build_shdr(self):
        out = bytearray()
        for s in self.samples:
            out += _cstr(s.name, 20)
            out += struct.pack("<IIIII", s.start, s.end, s.startloop, s.endloop, s.sample_rate)
            out += struct.pack("<Bb", s.root_key, s.pitch_correction)
            out += struct.pack("<HH", 0, 1)  # sampleLink=0, sfSampleType=1 (mono)
        out += _cstr("EOS", 20) + struct.pack("<IIIII", 0, 0, 0, 0, 0) + struct.pack("<BbHH", 0, 0, 0, 0)
        return bytes(out)

    def _gen_bytes(self, gens):
        out = bytearray()
        for g in gens:
            if g[0] == "range":
                _, oper, lo, hi = g
                out += gen_record_range(oper, lo, hi)
            else:
                oper, amount = g
                out += gen_record(oper, amount)
        return bytes(out)

    def _build_inst_pdta(self):
        inst_hdr = bytearray()
        ibag = bytearray()
        igen = bytearray()
        gen_ndx = 0
        bag_ndx = 0
        for inst in self.instruments:
            inst_hdr += _cstr(inst.name, 20) + struct.pack("<H", bag_ndx)
            # global zone (only if it has generators)
            if inst.global_gens:
                ibag += struct.pack("<HH", gen_ndx, 0)
                bag_ndx += 1
                gb = self._gen_bytes(inst.global_gens)
                igen += gb
                gen_ndx += len(gb) // 4
            for z in inst.zones:
                ibag += struct.pack("<HH", gen_ndx, 0)
                bag_ndx += 1
                gens = list(z.gens) + [(GEN["sampleID"], z.sample_index)]
                gb = self._gen_bytes(gens)
                igen += gb
                gen_ndx += len(gb) // 4
        inst_hdr += _cstr("EOI", 20) + struct.pack("<H", bag_ndx)
        ibag += struct.pack("<HH", gen_ndx, 0)
        igen += struct.pack("<HH", 0, 0)  # terminal generator record (oper=0,val=0)
        return bytes(inst_hdr), bytes(ibag), bytes(igen)

    def _build_preset_pdta(self):
        phdr = bytearray()
        pbag = bytearray()
        pgen = bytearray()
        gen_ndx = 0
        bag_ndx = 0
        for p in self.presets:
            phdr += _cstr(p.name, 20)
            phdr += struct.pack("<HH", p.program, p.bank)
            phdr += struct.pack("<H", bag_ndx)
            phdr += struct.pack("<III", 0, 0, 0)
            if p.global_gens:
                pbag += struct.pack("<HH", gen_ndx, 0)
                bag_ndx += 1
                gb = self._gen_bytes(p.global_gens)
                pgen += gb
                gen_ndx += len(gb) // 4
            pbag += struct.pack("<HH", gen_ndx, 0)
            bag_ndx += 1
            gens = [(GEN["instrument"], p.instrument_index)]
            gb = self._gen_bytes(gens)
            pgen += gb
            gen_ndx += len(gb) // 4
        phdr += _cstr("EOP", 20) + struct.pack("<HH", 0, 0) + struct.pack("<H", bag_ndx) + struct.pack("<III", 0, 0, 0)
        pbag += struct.pack("<HH", gen_ndx, 0)
        pgen += struct.pack("<HH", 0, 0)
        return bytes(phdr), bytes(pbag), bytes(pgen)

    def _build_info(self):
        subs = []
        subs.append(_chunk("ifil", struct.pack("<HH", 2, 1)))
        subs.append(_chunk("isng", _cstr("EMU8000", 8)))
        subs.append(_info_chunk("INAM", self.bank_name))
        cmt = ("TcGM.sf2 - GS(MSGS)-compatible GM1 bank synthesized entirely by additive "
               "(sine-summation) synthesis. No recorded samples were used.")
        subs.append(_info_chunk("ICMT", cmt))
        subs.append(_info_chunk("ISFT", "TcGM additive synth builder"))
        return _list("INFO", subs)

    def build(self) -> bytes:
        info = self._build_info()
        sdta = self._build_sdta()  # must run before shdr (fills start/end offsets)
        shdr = self._build_shdr()
        inst_hdr, ibag, igen = self._build_inst_pdta()
        phdr, pbag, pgen = self._build_preset_pdta()
        empty_mod = struct.pack("<HHhHH", 0, 0, 0, 0, 0)
        pdta_subs = [
            _chunk("phdr", phdr), _chunk("pbag", pbag), _chunk("pmod", empty_mod), _chunk("pgen", pgen),
            _chunk("inst", inst_hdr), _chunk("ibag", ibag), _chunk("imod", empty_mod), _chunk("igen", igen),
            _chunk("shdr", shdr),
        ]
        pdta = _list("pdta", pdta_subs)
        body = b"sfbk" + info + sdta + pdta
        return self._riff(body)

    @staticmethod
    def _riff(body):
        return b"RIFF" + struct.pack("<I", len(body)) + body
