from datetime import datetime
current_time = str(datetime.now().time())[:5]
import numpy as np
from totalsleep import predictions, activity_counts
from skintemp import skintemp
import torch
import sounddevice as sd
from circ import intravar, interstab, relatamp
from ambient import lights
from stressscore import stress_score
from finalmodel import TCN
import numpy as np
def finalpredict():
    def make_week(seed=42):
        rng = np.random.default_rng(seed)
        week = []

        for day_num in range(1):
            minutes = []

            for minute in range(1440):
                hour = minute // 60

                # Example sensor data for testing only
                bvp = rng.normal(0, 1, 6000).astype(np.float32)
                acc = rng.normal(0, 1, (6000, 3)).astype(np.float32)

                eda_raw = rng.normal(0.5, 0.1, 600).astype(np.float32)
                eda = np.repeat(eda_raw, 10)

                temp_raw = rng.normal(32, 0.2, 1200).astype(np.float32)
                temp = np.repeat(temp_raw, 5)

                # BH1750: ambient light, one measurement per hour
                # Approximate the current hour's light level for this test minute.
                hour = minute // 60
                light_level = 3 if hour < 7 or hour >= 22 else 200
                light = np.array([max(0, rng.normal(light_level, 10))])

                minutes.append([acc, bvp, eda, temp, light])

            # Replace this example mood value with your actual daily mood label
            mood = 0

            week.append([minutes, mood])

        return week


    week = make_week()

    print("Days:", len(week))
    print("Minutes in first day:", len(week[0][0]))
    print("Sensor arrays for first minute:", [x.shape for x in week[0][0][0]])
    print("Mood for first day:", week[0][1])
    def chunk_average(values):
        return [np.mean(values[i:i + 30]) for i in range(0, len(values), 30)]
    def chunk_max(values):
        return [max(values[i:i + 30]) for i in range(0, len(values), 30)]
    def encode_time(time_str):
        hour, minute = map(int, time_str.split(":"))

        minutes = hour * 60 + minute
        angle = 2 * np.pi * minutes / 1440

        return np.sin(angle), np.cos(angle)
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
    weekrelamp = []
    weeksleep = []
    weekmood = []
    all_stress = []
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

    # Expected shape: (1, 10, 336)
    model = TCN()
    model.load_state_dict(
        torch.load("tcn.pth", map_location="cpu", weights_only=True)
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
        }
    }


if __name__ == "__main__":
    result = finalpredict()
    print("Prediction:", result)