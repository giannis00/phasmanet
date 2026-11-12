#!/usr/bin/env python3
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
import pycbc.psd
import pycbc.coordinates as co
from pycbc.filter import sigmasq

import numpy as np
import pandas as pd
from scipy.stats import qmc
from joblib import Parallel, delayed
from argparse import ArgumentParser
import os, math

# ---------------- CLI ----------------
parser = ArgumentParser()
parser.add_argument('--output', type=str, default='out.pkl', help="Output pickle path")
parser.add_argument('--nsamples', type=int, default=1000, help="Number of samples to generate")
parser.add_argument('--ncores', type=int, default=4, help="CPU cores for parallel jobs")
parser.add_argument('--force', action='store_true', help="Overwrite existing file")
parser.add_argument('--m12uniform', action='store_true', help="Use uniform m1, m2 sampling instead of Sobol for Mc, q")

args = parser.parse_args()

if os.path.isfile(args.output) and not args.force:
    raise IOError(f"File {args.output} exists. Use --force to overwrite.")

# ---------------- Global constants ----------------
np.random.seed(44)

sampling_rate   = 2048
signal_duration = 20
delta_f         = 1.0 / signal_duration
f_Nyquist       = sampling_rate / 2.0
f_higher        = f_Nyquist
flen            = int(f_Nyquist / delta_f) + 1
f_lower         = 10.0

ds = 1000.0   # Mpc
tc = 0.0
apx = 'IMRPhenomXPHM'

# Mass limits (ordered)
m_min, m_max = 3.0, 170.0

# Derived (for Sobol box)
Mc_min = (m_min*m_min)**(3/5) / (2*m_min)**(1/5)
Mc_max = (m_max*m_max)**(3/5) / (2*m_max)**(1/5)
q_min, q_max = 1.0, m_max / m_min         # q = m1/m2 >= 1

logMc_min, logMc_max = np.log10(Mc_min), np.log10(Mc_max)
logq_min,  logq_max  = np.log10(q_min),  np.log10(q_max)

# Spin & extrinsic ranges (unchanged)
s1_mag_min, s1_mag_max = 0.0, 0.99
s2_mag_min, s2_mag_max = 0.0, 0.99
s1_polar_min, s1_polar_max = 0.0, np.pi
s2_polar_min, s2_polar_max = 0.0, np.pi
s1_azimuthal_min, s1_azimuthal_max = 0.0, 2*np.pi
s2_azimuthal_min, s2_azimuthal_max = 0.0, 2*np.pi

cos_inc_min, cos_inc_max = -1.0, 1.0
sin_dec_min, sin_dec_max = -1.0, 1.0
ra_min, ra_max = 0.0, 2*np.pi
pol_min, pol_max = 0.0, np.pi

# PSD (H1)
psd_fs = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_lower)
#psd_np = np.asarray(psd_fs)

# ---------- Helpers ----------
def masses_from_Mc_q(Mc, q):
    # q = m1/m2 >= 1
    m1 = Mc * (1+q)**(1/5) / (q**(3/5))
    m2 = m1 / q
    return m1, m2

def snr_opt_fd(h_fd, psd_fs, f_low):
    # h_fd: PyCBC FrequencySeries (detector strain)
    # psd_fs: PyCBC FrequencySeries (same delta_f, length)
    try:
        rho2 = sigmasq(h_fd, psd=psd_fs, low_frequency_cutoff=f_low)
    except Exception:
        # Fallback: mask non-finite and zeros explicitly (rare)
        h = np.asarray(h_fd.data)
        Sn = np.asarray(psd_fs.data)
        df = float(h_fd.delta_f)
        f = h_fd.sample_frequencies.numpy()
        m = np.isfinite(h) & np.isfinite(Sn) & (Sn > 0) & (f >= f_low)
        rho2 = 4.0 * np.sum((np.abs(h[m])**2) / Sn[m]) * df
    return float(np.sqrt(max(rho2, 0.0)))


def fourier_mean_snr(vals):
    # mean over K uniformly spaced phases equals a0 of Fourier series
    return float(np.mean(vals))

if args.m12uniform: 
    print("Using uniform sampling in (m1, m2) instead of Sobol in (logMc, logq).")

    # ---------- Uniform mass sampling in (m1, m2) with m1 >= m2 ----------
    Ns = args.nsamples
    rng = np.random.default_rng(44)

    # Uniform draws in the box, then enforce ordering
    m1_draw = rng.uniform(m_min, m_max, size=Ns)
    m2_draw = rng.uniform(m_min, m_max, size=Ns)

    # Enforce m1 >= m2 by swapping where needed
    swap = m2_draw > m1_draw
    m1_samples = np.where(swap, m2_draw, m1_draw)
    m2_samples = np.where(swap, m1_draw, m2_draw)

    # Convert to logMc, logq for the worker / features (keep the worker unchanged)
    Mc_samples = (m1_samples * m2_samples)**(3/5) / (m1_samples + m2_samples)**(1/5)
    q_samples  = m1_samples / m2_samples
    logMc_samples = np.log10(Mc_samples)
    logq_samples  = np.log10(q_samples)

    print(f"Uniform (m1,m2) sampling: generated {Ns} ordered mass pairs in [{m_min}, {m_max}] Msun.")

else:
    # ---------- Sobol sampling in (logMc, logq) with rejection ----------
    Ns = args.nsamples
    # oversample so we likely accept Ns without resampling Sobol
    overshoot = 2
    target_draws = 1
    while True:
        # try enough Sobol points; use power-of-two for balance
        m_pow2 = int(np.ceil(np.log2(Ns * overshoot * target_draws)))
        sob = qmc.Sobol(d=2, scramble=True, seed=44 + target_draws)
        U = sob.random_base2(m=m_pow2)  # shape (2^m, 2)
        # map to log-box
        logMc_all = logMc_min + U[:,0] * (logMc_max - logMc_min)
        logq_all  = logq_min  + U[:,1] * (logq_max  - logq_min)

        accepted = []
        for lMc, lq in zip(logMc_all, logq_all):
            Mc = 10**lMc
            q  = 10**lq
            m1, m2 = masses_from_Mc_q(Mc, q)
            # enforce physical & ordered masses
            if (m_min <= m2 <= m1 <= m_max):
                accepted.append((lMc, lq))
                if len(accepted) == Ns:
                    break
        if len(accepted) >= Ns:
            break
        # not enough accepted; increase target_draws and try again
        target_draws += 1

    print(f"Accepted {len(accepted)} / {len(U)} Sobol points "
        f"(needed {Ns}; drew {len(U)}; tries={target_draws})")

    logMc_samples = np.array([a[0] for a in accepted[:Ns]])
    logq_samples  = np.array([a[1] for a in accepted[:Ns]])

# ---------- Randomize the rest (unchanged) ----------
rand_extras = np.random.uniform(0, 1, size=(Ns, 10))

# ---------- Worker ----------
def do_one(i, lMc, lq, rands):
    # create detector per-process (avoid pickling issues)
    det = Detector('H1')

    # masses
    Mc = 10**lMc
    q  = 10**lq
    m1, m2 = masses_from_Mc_q(Mc, q)

    # spins (spherical -> Cartesian)
    s1_mag = s1_mag_min + rands[0]*(s1_mag_max - s1_mag_min)
    s1_pol = s1_polar_min + rands[1]*(s1_polar_max - s1_polar_min)
    s1_azi = s1_azimuthal_min + rands[2]*(s1_azimuthal_max - s1_azimuthal_min)
    s2_mag = s2_mag_min + rands[3]*(s2_mag_max - s2_mag_min)
    s2_pol = s2_polar_min + rands[4]*(s2_polar_max - s2_polar_min)
    s2_azi = s2_azimuthal_min + rands[5]*(s2_azimuthal_max - s2_azimuthal_min)

    s1x, s1y, s1z = co.spherical_to_cartesian(s1_mag, s1_pol, s1_azi)
    s2x, s2y, s2z = co.spherical_to_cartesian(s2_mag, s2_pol, s2_azi)

    # extrinsics
    cos_inc = cos_inc_min + rands[6]*(cos_inc_max - cos_inc_min)
    inc = np.arccos(cos_inc)
    dec = np.arcsin(sin_dec_min + rands[7]*(sin_dec_max - sin_dec_min))
    ra  = ra_min + rands[8]*(ra_max - ra_min)
    pol = pol_min + rands[9]*(pol_max - pol_min)

    Fp, Fx = det.antenna_pattern(ra, dec, pol, tc)
    start_bin = int(f_lower / delta_f)

    # phase averaging with K=7 points
    K = 7
    phis = np.linspace(0, 2*np.pi, K, endpoint=False)
    snrs = []
    for phic in phis:
        hp, hc = get_fd_waveform(approximant=apx,
                                 f_lower=f_lower, f_final=f_higher, delta_f=delta_f,
                                 mass1=m1, mass2=m2,
                                 spin1x=s1x, spin1y=s1y, spin1z=s1z,
                                 spin2x=s2x, spin2y=s2y, spin2z=s2z,
                                 inclination=inc, distance=ds,
                                 coa_phase=phic, f_ref=f_lower)
        # pad/crop to PSD length
        if len(hp) < flen:
            hp.resize(flen); hc.resize(flen)
        elif len(hp) > flen:
            hp = hp[:flen]; hc = hc[:flen]

        h_sig = Fp * hp + Fx * hc
        #snrs.append(snr_opt_fd(h_sig, psd_np, start_bin, delta_f))
        snrs.append(snr_opt_fd(h_sig, psd_fs, f_lower))

    SNR_mean = fourier_mean_snr(snrs)
    logSNR   = float(np.log10(SNR_mean + 1e-12))

    return dict(ds=ds, logMc=lMc, logq=lq, cos_inc=cos_inc,
                s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z,
                Fp=Fp, Fx=Fx, tc=tc, logSNR=logSNR)

# ---------- Parallel run ----------
results = Parallel(n_jobs=args.ncores, verbose=5)(
    delayed(do_one)(i, logMc_samples[i], logq_samples[i], rand_extras[i])
    for i in range(args.nsamples)
)

# ---------- Save ----------
df = pd.DataFrame(results)
df = df[['ds','logMc','logq','cos_inc',
         's1x','s1y','s1z','s2x','s2y','s2z',
         'Fp','Fx','tc','logSNR']]
df.to_pickle(args.output)

print(f"Saved {len(df)} samples to {args.output}")
