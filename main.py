from datetime import datetime
current_time = str(datetime.now().time())[:5]
from total_sleep import predictions, activity_counts
from skintemp import skintemp
import torch
from circadianrhythm import intravar, interstab, relatamp
from ambientlight import lights
from stress_score import stress_score
from finalmodel import TCN, trainer
import numpy as np
from pathlib import Path
import json
from synthetic_data import make_week
def finalpredict(seed=42):
    week = make_week(seed)
    def chunk_average(values):
        return [np.mean(values[i:i + 30]) for i in range(0, len(values), 30)]
    def chunk_max(values):
        return [max(values[i:i + 30]) for i in range(0, len(values), 30)]
    minutes = np.arange(336) * 30 + 15
    angles = 2 * np.pi * (minutes % 1440) / 1440
    sin_time = np.sin(angles)
    cos_time = np.cos(angles)

    def chunk_var(values):
        return [float(np.var(values[i:i + 30])) for i in range(0, len(values), 30)]

    #week = [[day[min[features]],mood]]
    #min = [acc, bvp, eda, temp, light]
    previousactivity = [0]*11

    weekhighlight = []
    weekmalux = []
    weeklightv = []
    weekstdtemp = []
    weekstress = []
    weeksleep = []
    weekmood = []
    for days in week:
        weekmood.append(days[1])

        highlight = []
        malux = []
        luxlist = []
        lightv = []
        stdtemp = []
        stress = []
        sleeplist = []
        for min in days[0]:
            acc, bvp, eda, temp, light = min
            luxlist.append(float(np.mean(light)))
            sleepexposure,mlux,lvariance = lights(np.asarray(light))

            highlight.append(sleepexposure)
            malux.append(mlux)
            lightv.append(lvariance)

            stdtemp.append(skintemp(np.asarray(temp))[1])
            stress.append(stress_score(acc, bvp, eda, temp))
            acc1, acc2 = acc[:len(acc)//2], acc[len(acc)//2:]
            bvp1, bvp2 = bvp[:len(bvp) // 2], bvp[len(bvp) // 2:]

            pred1 = predictions(acc1, bvp1, previousactivity)

            previousactivity.append(activity_counts(acc1))
            previousactivity.pop(0)

            pred2 = predictions(acc2, bvp2, previousactivity)

            previousactivity.append(activity_counts(acc2))
            previousactivity.pop(0)

            sleeplist.append(np.mean([pred1, pred2]))

        sleep30 = chunk_average(sleeplist)
        weeksleep.append(sleep30)

        weekhighlight.append(chunk_average(highlight))
        weekmalux.append(chunk_max(malux))
        weeklightv.append(chunk_var(luxlist))
        weekstdtemp.append(chunk_average(stdtemp))
        weekstress.append(chunk_average(stress))

    fsleepweek = [value for day in weeksleep for value in day]
    weekhighlight = [value for day in weekhighlight for value in day]
    weekmalux = [value for day in weekmalux for value in day]
    weeklightv = [value for day in weeklightv for value in day]
    weekstdtemp = [value for day in weekstdtemp for value in day]
    weekmood = [mood for mood in weekmood for i in range(48)]
    interdailys = interstab(weeksleep)
    intradailys = intravar(fsleepweek)
    relamps = relatamp(weeksleep)

    def noNan(v):
        return 0 if not np.isfinite(v) else float(v)

    interdaily_graph = np.full(336, interdailys)
    intradaily_graph = np.full(336, intradailys)
    relamp_graph = np.full(336, relamps)
    interdailys = np.full(336, noNan(interdailys))
    intradailys = np.full(336, noNan(intradailys))
    relamps = np.full(336, noNan(relamps))
    stresss = [value for day in weekstress for value in day]

    features1 = [stresss, fsleepweek, interdailys, intradailys, relamps, weekstdtemp, weekmood, weekhighlight, weekmalux, weeklightv, sin_time,cos_time]
    X_real = np.asarray(features1, dtype=np.float32)
    X_real = torch.tensor(X_real, dtype=torch.float32).unsqueeze(0)
    model = TCN()
    model_path = Path(__file__).resolve().parent / "model" / "tcn.pth"

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    model.eval()

    with torch.no_grad():
        prediction = model(X_real)

    probabilities = torch.softmax(prediction / 4, dim=1)[0]
    predicted_class = int(torch.argmax(probabilities).item())

    classes = ["mania", "euthymia", "depression"]
    return {
        "predicted_class": classes[predicted_class],
        "class_index": predicted_class,
        "probabilities": {
            classes[i]: float(probabilities[i].item())
            for i in range(3)
        },
        "graphs": {
            "stress": [float(x) for x in stresss],
            "sleep": [float(x) for x in fsleepweek],
            "skin_temperature": [float(x) for x in weekstdtemp],
            "mood": [float(x) for x in weekmood],
            "light_exposure": [float(x) for x in weekhighlight],
            "ambient_light": [float(x) for x in weekmalux],
            "light_variability": [float(x) for x in weeklightv],
            "intradaily_variability": [float(x) for x in intradaily_graph],
            "interdaily_stability": [float(x) for x in interdaily_graph],
            "relative_amplitude": [float(x) for x in relamp_graph],
            "time_sin": [float(x) for x in sin_time],
            "time_cos": [float(x) for x in cos_time]
        }
    }


if __name__ == "__main__":
    trainer()
    result = finalpredict()

    output_path = (Path(__file__).resolve().parent/"public"/"prediction_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
