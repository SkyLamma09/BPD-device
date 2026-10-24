
import numpy as np
import torch


def synthetic_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    # 7 days x 48 half-hour segments x 12 features
    X = np.zeros((n, 7, 48, 12), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)

    for i in range(n):
        label = rng.integers(0, 2)
        y[i, 0] = label

        stress_base = rng.uniform(0.2, 0.6)
        temp_base = rng.normal(0.0, 0.15)
        mood_base = rng.normal(0.0, 0.3)

        if label == 1:
            trend = rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
            stress_shift = rng.uniform(0.20, 0.35)
            sleep_disruption = rng.uniform(0.20, 0.40)
            mood_shift = rng.choice([-1, 1])
        else:
            trend = rng.normal(0, 0.01)
            stress_shift = rng.normal(0, 0.04)
            sleep_disruption = rng.uniform(0, 0.08)
            mood_shift = 0.0

        # One weekly value per metric, repeated across the week.
        # This matches the current real-data pipeline's approach.
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
                mood_base + mood_shift + 0.3 * disturbance_day
            )

            for segment in range(48):
                # Half-hour segment's midpoint, in minutes after midnight.
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

                # Weekly circadian metrics, repeated per segment.
                stability = stability_week
                variability = variability_week
                amplitude = amplitude_week

                temp_deviation = temp_day + rng.normal(0, 0.08)
                mood = mood_day + rng.normal(0, 0.3)

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
                    stability,
                    variability,
                    amplitude,
                    np.clip(temp_deviation, -2, 2),
                    np.clip(mood, -2, 2),
                    np.clip(excess_light, 0, 8),
                    max_lux,
                    light_variance,
                    sin_time,
                    cos_time,
                ]

    # (n, 7, 48, 12) -> (n, 12, 336)
    X = X.reshape(n, 336, 12).transpose(0, 2, 1).copy()

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


if __name__ == "__main__":
    X, y = synthetic_data(n=2000)
    print("X shape:", X.shape)  # (2000, 12, 336)
    print("y shape:", y.shape)  # (2000, 1)
