const API_URL = "http://localhost:8000";


async function predict() {

   const data = {
    hour: 14,
    day: 15,
    month: 3,
    day_of_week: 2,
    is_weekend: 0,
    lag_1: parseFloat(document.getElementById("lag1").value),
    lag_24: parseFloat(document.getElementById("lag24").value),
    block_Block_B: 1
};


    const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById("predictionResult").innerText =
        "Predicted Units: " + result.predicted_units;
}

async function checkAnomaly() {

    const data = {
        units: parseFloat(document.getElementById("units").value)
    };

    const response = await fetch(`${API_URL}/anomaly`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById("anomalyResult").innerText =
        result.anomaly === 1 ? "⚠️ Anomaly Detected!" : "Normal Usage";
}
