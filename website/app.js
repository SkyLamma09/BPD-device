const classInfo = {
    mania: "Mania",
    euthymia: "Euthymia/Stable",
    depression: "Depression"
};

const metrics = [
    { label: "Sleep per night", unit: " h", values: g =>
        Array.from({ length: 7 }, (_, d) =>
            g.sleep.slice(d * 48, d * 48 + 48).reduce((s, v) => s + (Number(v) || 0), 0) / 2) },
    { label: "Skin temp deviation", unit: "", values: g => g.skin_temperature },
    { label: "Ambient light", unit: " lux", values: g => g.ambient_light },
    { label: "Stress estimate", unit: "", values: g => g.stress },
    { label: "Night light exposure", unit: "", values: g => g.light_exposure },
    { label: "Light variability", unit: "", values: g => g.light_variability },
    { label: "Mood rating", unit: "", values: g => g.mood },
    { label: "Interdaily stability", unit: "", values: g => g.interdaily_stability },
    { label: "Intradaily variability", unit: "", values: g => g.intradaily_variability },
    { label: "Relative amplitude", unit: "", values: g => g.relative_amplitude }
];
const clean = arr => (arr || []).map(Number).filter(Number.isFinite);
const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;

function trend(v) {
    if (v.length < 2) return "";
    const half = Math.floor(v.length / 2);
    const diff = mean(v.slice(half)) - mean(v.slice(0, half));
    if (Math.abs(diff) < 0.05 * Math.abs(mean(v))) return "steady";
    return diff > 0 ? "rising" : "falling";
}

function showRisk({ probabilities: probs, predicted_class: top }) {
    const pct = name => ((probs[name] || 0) * 100).toFixed(0) + "%";

    document.getElementById("topResult").textContent = pct(top);
    document.getElementById("topLabel").textContent = `chance of ${classInfo[top] || top}`;
    document.getElementById("bars").innerHTML = Object.entries(classInfo)
        .map(([name, label]) => `<div>${label}: <strong>${pct(name)}</strong></div>`)
        .join("");
}

function showInsights(graphs) {
    document.getElementById("insights").innerHTML = metrics.map(m => {
        const v = clean(m.values(graphs));
        const value = v.length ? mean(v).toFixed(2) + m.unit : "--";
        return `<div><small>${m.label}</small><strong>${value}</strong><small>${v.length ? trend(v) : ""}</small></div>`;
    }).join("");
}

async function runPrediction() {
    const btn = document.getElementById("runBtn");
    const status = document.getElementById("risk");

    btn.disabled = true;
    status.textContent = "Running prediction. This will take around 15 minutes.";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict");
        const data = await response.json();

        showRisk(data);
        showInsights(data.graphs);
        status.textContent = "";
    } catch (error) {
        status.textContent = "Prediction failed. Check the backend.";
        console.error(error);
    }

    btn.disabled = false;
}
function submitMood(event) {
    event.preventDefault();
    event.target.reset();
}

function saveNote() {
    const box = document.querySelector("#notes textarea");
    const text = box.value.trim();
    if (!text) return;

    const li = document.createElement("li");
    li.textContent = text;
    document.getElementById("notesList").prepend(li);
    box.value = "";
}
