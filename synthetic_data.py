import numpy as np
import torch
from pathlib import Path
import pickle
DATA_PATH = Path(__file__).resolve().parent / "training_data.npz"


def synthetic_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    X = np.zeros((n, 7, 48, 12), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.int64)

    for i in range(n):
        # 0 = depression, 1 = euthymia, 2 = mania
        label = rng.integers(0, 3)
        y[i, 0] = label

        stress_base = rng.uniform(0.2, 0.6)
        temp_base = rng.normal(0.0, 0.15)
        mood_base = rng.normal(0.0, 0.3)

        if label == 0:
            trend = rng.normal(0, 0.02)
            stress_shift = rng.uniform(-0.05, 0.10)
            sleep_disruption = rng.uniform(0.15, 0.35)
            mood_shift = -0.8
            activity_shift = -0.5

        elif label == 1:
            trend = rng.normal(0, 0.01)
            stress_shift = rng.normal(0, 0.04)
            sleep_disruption = rng.uniform(0, 0.08)
            mood_shift = 0.0
            activity_shift = 0.0

        else:
            trend = rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
            stress_shift = rng.uniform(0.10, 0.30)
            sleep_disruption = rng.uniform(0.20, 0.40)
            mood_shift = 0.8
            activity_shift = 0.5

        disturbance_week = rng.normal(0, 1)

        stability_week = np.clip(
            0.70 - 0.15 * abs(trend) * 3
            - 0.04 * abs(disturbance_week), 0, 1
        )

        variability_week = np.clip(
            0.80 + 0.5 * abs(trend) * 3
            + 0.05 * abs(disturbance_week), 0, 3
        )

        amplitude_week = np.clip(
            0.60 - 0.3 * abs(trend) * 3, 0, 1
        )

        for day in range(7):
            disturbance_day = rng.normal(0, 1)

            stress_day = (
                stress_base + stress_shift
                + trend * day + 0.05 * disturbance_day
            )

            temp_day = temp_base + 0.08 * disturbance_day

            mood_day = (
                mood_base + mood_shift
                + 0.3 * disturbance_day
            )

            for segment in range(48):
                minutes = segment * 30 + 15
                angle = 2 * np.pi * minutes / 1440

                sin_time = np.sin(angle)
                cos_time = np.cos(angle)

                hour = minutes / 60
                daytime = np.exp(-((hour - 14) / 5) ** 2)
                nighttime = hour >= 23 or hour < 7

                stress = (
                    stress_day + 0.06 * daytime
                    + rng.normal(0, 0.06)
                )

                if nighttime:
                    sleep = (
                        0.95 - sleep_disruption
                        + 0.10 * rng.normal()
                    )
                else:
                    sleep = (
                        0.03 + 0.08 * sleep_disruption
                        + 0.04 * rng.normal()
                    )

                temp_deviation = temp_day + rng.normal(0, 0.08)
                mood = mood_day + rng.normal(0, 0.3)

                activity = (
                    0.5 + activity_shift
                    + 0.1 * daytime + rng.normal(0, 0.1)
                )

                excess_light = max(
                    0,
                    rng.normal(
                        0.3 + 0.5 * abs(disturbance_day)
                        + 0.8 * nighttime, 0.4
                    )
                )

                max_lux = max(
                    1,
                    rng.lognormal(
                        mean=3.0 + 0.2 * disturbance_day
                        + 0.5 * daytime,
                        sigma=0.4
                    )
                )

                light_variance = max(
                    0,
                    rng.lognormal(
                        mean=1.5 + 0.2 * disturbance_day,
                        sigma=0.5
                    )
                )

                X[i, day, segment] = [
                    np.clip(stress, 0, 1),
                    np.clip(sleep, 0, 1),
                    stability_week,
                    variability_week,
                    amplitude_week,
                    np.clip(temp_deviation, -2, 2),
                    np.clip(mood, -2, 2),
                    np.clip(excess_light, 0, 8),
                    max_lux,
                    light_variance,
                    sin_time,
                    cos_time,
                ]

    X = X.reshape(n, 336, 12).transpose(0, 2, 1).copy()

    return X, y.flatten()


def save_training_data(X, y, append=False):
    """Save examples, optionally appending to existing examples."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if X.shape != (len(y), 12, 336):
        raise ValueError(
            f"Expected X shape (N, 12, 336); got {X.shape}"
        )

    if np.any((y < 0) | (y > 2)):
        raise ValueError("Labels must be 0, 1, or 2.")

    if append and DATA_PATH.exists():
        with np.load(DATA_PATH) as saved:
            old_X = saved["X"]
            old_y = saved["y"]

        X = np.concatenate([old_X, X], axis=0)
        y = np.concatenate([old_y, y], axis=0)

    np.savez_compressed(DATA_PATH, X=X, y=y)

    print(f"Saved {len(y)} training examples to {DATA_PATH}")


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No training data found at {DATA_PATH}. "
            "Run synthetic_data.py first."
        )

    with np.load(DATA_PATH) as saved:
        X = saved["X"].copy()
        y = saved["y"].copy()

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


if __name__ == "__main__":
    if not DATA_PATH.exists():
        X, y = synthetic_data(n=2000, seed=42)
        save_training_data(X, y)
    else:
        print(f"Training file already exists: {DATA_PATH}")
        print("Keeping existing data. No examples were regenerated.")

    X, y = load_training_data()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Labels:", torch.unique(y))
    print("Class counts:", torch.bincount(y))




# Same label coding as synthetic_data() above: 0 = depression, 1 = euthymia, 2 = mania
STATE_NAMES = ["depression", "euthymia", "mania"]


_STATE_PARAMS = {
    0: dict(bed=(23.5, 1.5), length=(9.5, 1.2), mood=-1.3, activity=0.6, night_light=0.15, wakes=1.5),
    1: dict(bed=(23.0, 0.5), length=(7.7, 0.5), mood=0.0,  activity=1.0, night_light=0.03, wakes=0.4),
    2: dict(bed=(25.0, 1.5), length=(4.8, 1.0), mood=1.3,  activity=1.5, night_light=0.40, wakes=1.0),
}


def _sleep_schedule(rng, p, n_days=7):

    asleep = np.zeros((n_days + 2) * 1440, dtype=bool)
    # nights -1 .. n_days-1; the extra first day lets night -1 cover the early morning of day 0
    for night in range(-1, n_days):
        offset = (night + 1) * 1440
        bed = rng.normal(*p["bed"]) * 60
        length = max(2.0, rng.normal(*p["length"])) * 60
        start = int(offset + bed)
        end = int(start + length)
        asleep[start:end] = True
        for _ in range(rng.poisson(p["wakes"])):
            a = int(rng.integers(start, max(start + 1, end)))
            asleep[a:a + int(rng.integers(5, 25))] = False
    return asleep[1440:1440 + n_days * 1440]          # drop the helper day


def make_week(seed=None):

    rng = np.random.default_rng(seed)
    state = int(rng.integers(0, 3))
    p = _STATE_PARAMS[state]
    asleep = _sleep_schedule(rng, p)
    make_week.info = {"seed": seed, "state": state, "state_name": STATE_NAMES[state]}
    print(f"Synthetic week: seed={seed}, generated state={STATE_NAMES[state]}")

    t = np.arange(6000) / 100.0                       # 60 s sampled at 100 Hz
    week = []

    for day_num in range(7):
        mood = float(np.clip(np.round(rng.normal(p["mood"], 0.6)), -2, 2)) + 0.0
        day_light = rng.lognormal(5.6, 0.5)           # how sunny the day is (peak lux)
        day_temp = rng.normal(0.0, 0.15)
        minutes = []

        for minute in range(1440):
            hour = minute / 60.0
            sleeping = bool(asleep[day_num * 1440 + minute])
            night = hour < 7 or hour >= 23

            if sleeping:
                sigma = 0.004
                if rng.random() < 0.03:               # occasional turning over
                    sigma = 0.06
            else:
                sigma = 0.05 + 0.25 * p["activity"] * rng.lognormal(-0.3, 0.6) * (0.4 if night else 1.0)
            acc = rng.normal(0, sigma, (6000, 3)).astype(np.float32)
            acc[:, 2] += 1.0                          # gravity

            bpm = (rng.normal(56, 3) if sleeping else rng.normal(75, 6)) + 6 * (state == 2) - 3 * (state == 0)
            bvp = (np.sin(2 * np.pi * (bpm / 60.0) * t) + rng.normal(0, 0.1, 6000)).astype(np.float32)

            eda_raw = rng.normal(0.3 if sleeping else 0.5, 0.1, 600).astype(np.float32)
            eda = np.repeat(eda_raw, 10)

            temp_raw = rng.normal(32 + 0.6 * sleeping + day_temp, 0.2, 1200).astype(np.float32)
            temp = np.repeat(temp_raw, 5)

            if sleeping:
                lux = max(0.0, rng.normal(0.5, 0.5))
                if rng.random() < p["night_light"] / 10:      # a lamp / phone left on
                    lux = rng.uniform(15, 80)
            elif night:
                lux = rng.uniform(20, 90) if rng.random() < 0.5 + p["night_light"] else 2.0
            else:
                bell = np.exp(-((hour - 13) / 3.5) ** 2)
                lux = (60 + day_light * bell) * rng.lognormal(0, 0.25)   # clouds / indoors-outdoors
            light = np.maximum(0, lux * (1 + 0.15 * rng.normal(size=6)))

            minutes.append([acc, bvp, eda, temp, light])

        week.append([minutes, mood])

    return week
