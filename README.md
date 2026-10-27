# Prodrome

A multimodal approach to detecting the onset of bipolar disorder episodes.

Prodrome is a prototype for a wearable device (BPD-device). It takes a week of sensor data, runs it through several submodels and a temporal convolutional network (TCN), and returns the percent chance of **euthymia** (stable baseline), **mania**, or **depression**. The dashboard also shows the numbers behind the prediction, since patients often struggle to remember medical details between visits.

**Disclaimer:** Research prototype using synthetic data. Not a medical device and not for diagnosis or treatment.

## Features

- **Homepage:** minimal landing page with the logo.
- **Predictions:** one button generates a synthetic week and sends it to a Flask API, which returns the three probabilities.
- **Weekly insights:** sleep, stress, skin temperature, light, and circadian rhythm metrics shown as numbers with trends.
- **Self report:** mood scale from 0 (neutral) and 1 (low energy) up to 4 (high energy), designed as a model input.
- **Notes:** journaling and coping strategies.

## How it works

Synthetic week → submodels (sleep, stress, skin temperature, light, circadian rhythm) → 12 features × 336 half-hour steps → TCN → probabilities.

## Run it

```bash
pip install flask flask-cors torch numpy scipy scikit-learn joblib agcounts
python api.py
```

Then open `website/index.html` and press **Run with a synthetic week**. The first run can take several minutes.
## AI usage
- The synthetic data algorithm was written with GPT because we could not find public open source bipolar disorder datasets.
- GPT helped debug the GUI.
## Roadmap

- Use real wearable data
- Send self-report mood to the API
- Save notes between sessions
- Speed up the pipeline
