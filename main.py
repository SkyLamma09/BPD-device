from datetime import datetime
current_time = str(datetime.now().time())[:5]
import numpy as np
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
def finalpredict():



    week = make_week()


    def chunk_average(values):
        return [np.mean(values[i:i + 30]) for i in range(0, len(values), 30)]
    def chunk_max(values):
        return [max(values[i:i + 30]) for i in range(0, len(values), 30)]
    minutes = np.arange(336) * 30 + 15
    angles = 2 * np.pi * (minutes % 1440) / 1440
    sin_time = np.sin(angles)
    cos_time = np.cos(angles)#not temporary

    #weeklist daylist[[30 min block[30 one minute blocks]]
    #assume that the data coming in is in one minute chunks
    #use synthetic data generator(check how much it has in one minute chunk for each datapoint)
    #mins are taken out then added to each list after processing
    #week = [[day[min],mood],day[min],day[min]]
    #min = [acc, bvp, eda, temp, light]
    previousactivity = [0]*11
    #previousactivity.pop(0)
    #previousactivity.append(activity_counts(week[0][1][0][:len(week[0][1][0])//2]))
    weekhighlight = []
    weekmalux = []
    weeklightv = []
    weekstdtemp = []
    weekstress = []
    #weekrelamp = []
    weeksleep = []
    weekmood = []
    #all_stress = []
    for days in week:
        weekmood.append(days[1])

        highlight = []
        malux = []
        lightv = []
        stdtemp = []
        stress = []
        sleeplist = []
        for min in days[0]:
            acc, bvp, eda, temp, light = min
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
            if len(sleeplist) < 5:
                print("Sleep predictions:", pred1, pred2)
        sleep30 = chunk_average(sleeplist)  # 48 values per day
        weeksleep.append(sleep30)

        weekhighlight.append(chunk_average(highlight))
        weekmalux.append(chunk_max(malux))
        weeklightv.append(chunk_average(lightv))
        weekstdtemp.append(chunk_average(stdtemp))
        weekstress.append(chunk_average(stress))




    fsleepweek = [value for day in weeksleep for value in day]
    print("First 20 sleep values:", fsleepweek[:20])
    print("Unique sleep values:", np.unique(fsleepweek))
    print("Sleep value range:", np.min(fsleepweek), np.max(fsleepweek))
    weekhighlight = [value for day in weekhighlight for value in day]
    weekmalux = [value for day in weekmalux for value in day]
    weeklightv = [value for day in weeklightv for value in day]
    weekstdtemp = [value for day in weekstdtemp for value in day]
    weekmood = [mood for mood in weekmood for i in range(48)]
    print("Sleep data shape:", np.asarray(weeksleep).shape)
    print("Unique sleep values:", np.unique(weeksleep))
    print("First day:", weeksleep[0])
    interdailys = interstab(weeksleep)

    intradailys = intravar(fsleepweek)

    relamps = relatamp(weeksleep)
    interdailys = np.full(336, interdailys)
    intradailys = np.full(336, intradailys)
    relamps = np.full(336, relamps)
    stresss = [value for day in weekstress for value in day]
    #does the tcn take a 21 circ rhtym values 3 per day or just 3 for a week
    # I need sleep for in terms of day, for that I need to cycle through 30 secs of each minute and label it
    features1 = [stresss, fsleepweek, interdailys, intradailys, relamps, weekstdtemp, weekmood, weekhighlight, weekmalux, weeklightv, sin_time,cos_time]
    X_real = np.asarray(features1, dtype=np.float32)

    assert X_real.shape == (12, 336), (
        f"Expected (12, 336), got {X_real.shape}"
    )

    X_real = torch.tensor(X_real, dtype=torch.float32).unsqueeze(0)

    model = TCN()

    model_path = Path(__file__).resolve().parent / "models" / "tcn.pth"

    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    model.eval()

    with torch.no_grad():
        prediction = model(X_real)

    probabilities = torch.softmax(prediction, dim=1)[0]

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
            "intradaily_variability": [float(x) for x in intradailys],
            "interdaily_stability": [float(x) for x in interdailys],
            "relative_amplitude": [float(x) for x in relamps],
            "time_sin": [float(x) for x in sin_time],
            "time_cos": [float(x) for x in cos_time]
        }
    }


if __name__ == "__main__":
    result = finalpredict()
    output_path = (
            Path(__file__).resolve().parent
            / "public"
            / "prediction_results.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Prediction:", result)
    print("Results saved to:", output_path)
    trainer()
