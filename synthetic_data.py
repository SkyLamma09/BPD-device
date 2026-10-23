import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def synthetic_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 7, 24, 10), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)

    for i in range(n):
        label = rng.integers(0, 2)
        y[i, 0] = label

        stress_base = rng.uniform(0.2, 0.6)
        sleep_base = rng.normal(7.5, 0.7)
        temp_base = rng.normal(0.0, 0.15)


        if label == 1:
            trend = rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
            stress_shift = rng.uniform(0.20, 0.35)
            sleep_shift = rng.uniform(-3.0, -2.0)
            mood_shift = rng.choice([-1, 1])
        else:
            trend = rng.normal(0, 0.01)
            stress_shift = rng.normal(0, 0.04)
            sleep_shift = rng.normal(0, 0.25)
            mood_shift = 0

        for day in range(7):
            disturbance = rng.normal(0, 1)

            stress = (
                stress_base + stress_shift
                + trend * day + 0.05 * disturbance
                + rng.normal(0, 0.06)
            )

            sleep = (
                sleep_base + sleep_shift
                - trend * day * 4
                - 0.25 * disturbance
                + rng.normal(0, 0.4)
            )

            stability = (
                0.70 - 0.15 * abs(trend) * day
                - 0.04 * abs(disturbance)
                + rng.normal(0, 0.04)
            )

            variability = (
                0.80 + 0.5 * abs(trend) * day
                + 0.05 * abs(disturbance)
                + rng.normal(0, 0.06)
            )

            amplitude = (
                0.60 - 0.3 * abs(trend) * day
                + rng.normal(0, 0.05)
            )

            temp_deviation = (
                temp_base + 0.08 * disturbance
                + rng.normal(0, 0.08)
            )

            mood = mood_shift + 0.3 * disturbance + rng.normal(0, 0.3)

            excess_light = max(
                0, rng.normal(0.3 + 0.5 * abs(disturbance), 0.4)
            )

            max_lux = max(
                1, rng.lognormal(mean=3.0 + 0.2 * disturbance, sigma=0.4)
            )

            light_variance = max(
                0, rng.lognormal(mean=1.5 + 0.2 * disturbance, sigma=0.5)
            )

            X[i, day] = [
                np.clip(stress, 0, 1),
                np.clip(sleep, 0, 14),
                np.clip(stability, 0, 1),
                np.clip(variability, 0, 3),
                np.clip(amplitude, 0, 1),
                np.clip(temp_deviation, -2, 2),
                np.clip(mood, -2, 2),
                np.clip(excess_light, 0, 8),
                max_lux,
                light_variance
            ]

    X = X.reshape(n, 168, 10).transpose(0, 2, 1)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )


X, y = synthetic_data(n=2000)
