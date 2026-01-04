"""
create the frames to be used in the animation
takes each time step as an individual frame, samples a set number of frames for each 
"stage" which is a unique combination of stellar stypes
interpolate frames where there are larger jumps in any values
also lots of mass trasfer stuff that should prob be checked later

to do:
add preprocessing from a csv (startrack)
play around with current hardcoded values
ADD COMMENTS AND DOUBLE CHECK ALL CODE
"""

import math
import argparse
from pathlib import Path
import numpy as np
import h5py as h5

BASE_DIR = Path(__file__).parent
HDF5_PATH = BASE_DIR / "BSE_Detailed_Output_0.h5"
OUTPUT_FRAMES_FILE = BASE_DIR / "frames_data.npz"

FRAMES_PER_PHASE = 100
INTERPOLATED_FRAMES_FOR_JUMP = 50
JUMP_THRESHOLD_RATIO = 1.2
JUMP_THRESHOLD_ABS = 50.0
MT_PADDING_FRAMES = 60



def getStellarTypes():
    stellarTypes = [
        'MS','MS','HG','FGB','CHeB','EAGB','TPAGB','HeMS','HeHG',
        'HeGB','HeWD','COWD','ONeWD','NS','BH','MR','CHE'
    ]
    def typeMap(idx):
        return stellarTypes[int(idx)] if int(idx) < len(stellarTypes) else "unknown"
    return typeMap


def load_hdf5_and_mask(path):
    f = h5.File(str(path), "r")
    mask = f["Record_Type"][()] == 4
    Data = {k: f[k][()][mask] for k in f.keys()}
    f.close()
    return Data

#helper functions


def detect_phases_indices(Data):
    n = len(Data["Time"])
    phases = []
    start = 0
    for i in range(1, n):
        if (Data["Stellar_Type(1)"][i] != Data["Stellar_Type(1)"][i-1] or
            Data["Stellar_Type(2)"][i] != Data["Stellar_Type(2)"][i-1]):
            phases.append((start, i-1))
            start = i
    phases.append((start, n-1))
    return phases

def detect_mt_starts(Data):
    mt = Data["MT_History"].astype(int)
    starts = set()
    for i in range(1, len(mt)):
        if mt[i-1] == 0 and mt[i] > 0:
            starts.add(i)
    return starts


def sample_indices(start, end, n):
    return np.linspace(start, end, n).tolist()


def interp(Data, idx, key):
    if abs(idx - round(idx)) < 1e-9:
        return float(Data[key][int(idx)])
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return (1-frac)*float(Data[key][lo]) + frac*float(Data[key][hi])


def detect_large_jump(v1, v2):
    if v1 <= 0 or v2 <= 0:
        return False
    if max(v1, v2)/min(v1, v2) >= JUMP_THRESHOLD_RATIO:
        return True
    return abs(v1 - v2) >= JUMP_THRESHOLD_ABS


def make_event_string(idx, Data, typeMap):
    if idx == 0:
        return f"Zero-age main-sequence, Z = {float(Data['Metallicity@ZAMS(1)'][0]):.4f}"
    t1 = typeMap(Data["Stellar_Type(1)"][idx])
    t2 = typeMap(Data["Stellar_Type(2)"][idx])
    return f"Phase: {t1} + {t2}"

#actual code
def preprocess_to_frames(hdf5_path, out_path):
    print("Loading HDF5...")
    Data = load_hdf5_and_mask(hdf5_path)
    typeMap = getStellarTypes()
    phases = detect_phases_indices(Data)
    mt_starts = detect_mt_starts(Data)

    frames = []

    for pidx, (start, end) in enumerate(phases):
        positions = sample_indices(start, end, FRAMES_PER_PHASE)
        sampled = []

        # sampling
        for pos in positions:
            f = {}
            for k in [
                "Time","SemiMajorAxis","Eccentricity",
                "Radius(1)","Radius(2)",
                "Mass(1)","Mass(2)"
            ]:
                f[k] = interp(Data, pos, k)

            f["stypeName1"] = typeMap(int(round(interp(Data, pos, "Stellar_Type(1)"))))
            f["stypeName2"] = typeMap(int(round(interp(Data, pos, "Stellar_Type(2)"))))
            f["eventString"] = make_event_string(int(round(pos)), Data, typeMap)
            sampled.append(f)

        # interpolation
        enhanced = []
        for i in range(len(sampled)-1):
            a, b = sampled[i], sampled[i+1]
            enhanced.append(a)

            sep_a = a["SemiMajorAxis"]*(1+a["Eccentricity"])
            sep_b = b["SemiMajorAxis"]*(1+b["Eccentricity"])

            if (detect_large_jump(a["Radius(1)"], b["Radius(1)"]) or
                detect_large_jump(a["Radius(2)"], b["Radius(2)"]) or
                detect_large_jump(sep_a, sep_b)):

                for j in range(1, INTERPOLATED_FRAMES_FOR_JUMP+1):
                    alpha = j/(INTERPOLATED_FRAMES_FOR_JUMP+1)
                    inter = {}
                    for k in [
                        "Time","SemiMajorAxis","Eccentricity",
                        "Radius(1)","Radius(2)",
                        "Mass(1)","Mass(2)"
                    ]:
                        inter[k] = (1-alpha)*a[k] + alpha*b[k]
                    inter["stypeName1"] = a["stypeName1"]
                    inter["stypeName2"] = a["stypeName2"]
                    inter["eventString"] = a["eventString"]
                    enhanced.append(inter)

        enhanced.append(sampled[-1])

      

        # mass transfer
        mt_frames = []
        last_mt_frame = -999  # track the last MT frame added

        for i, f in enumerate(enhanced):
            mt_frames.append(f)

            data_idx = int(round(start + (end - start) * (i / max(1, len(enhanced)-1))))

            if data_idx in mt_starts:
                # Skip if previous MT was too close
                if i - last_mt_frame < 25:
                    continue

                for _ in range(MT_PADDING_FRAMES):
                    g = f.copy()
                    g["eventString"] = "Mass Transfer"
                    mt_frames.append(g)
                
                last_mt_frame = i

        # inter-phase interpolation
        if frames:
            prev = frames[-1]
            nxt = mt_frames[0]
            sep_prev = prev["SemiMajorAxis"]*(1+prev["Eccentricity"])
            sep_next = nxt["SemiMajorAxis"]*(1+nxt["Eccentricity"])

            if (detect_large_jump(prev["Radius(1)"], nxt["Radius(1)"]) or
                detect_large_jump(prev["Radius(2)"], nxt["Radius(2)"]) or
                detect_large_jump(sep_prev, sep_next)):

                for j in range(1, INTERPOLATED_FRAMES_FOR_JUMP+1):
                    alpha = j/(INTERPOLATED_FRAMES_FOR_JUMP+1)
                    inter = {}
                    for k in [
                        "Time","SemiMajorAxis","Eccentricity",
                        "Radius(1)","Radius(2)",
                        "Mass(1)","Mass(2)"
                    ]:
                        inter[k] = (1-alpha)*prev[k] + alpha*nxt[k]
                    inter["stypeName1"] = prev["stypeName1"]
                    inter["stypeName2"] = prev["stypeName2"]
                    inter["eventString"] = prev["eventString"]
                    frames.append(inter)

        frames.extend(mt_frames)
        print(f"Phase {pidx+1}/{len(phases)} → {len(mt_frames)} frames")

    np.savez_compressed(out_path, frames=frames)
    print(f"Saved {len(frames)} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=str(HDF5_PATH))
    parser.add_argument("--out", default=str(OUTPUT_FRAMES_FILE))
    args = parser.parse_args()

    preprocess_to_frames(args.hdf5, args.out)


if __name__ == "__main__":
    main()
