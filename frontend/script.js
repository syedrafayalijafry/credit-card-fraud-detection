const predictBtn = document.getElementById("predictBtn");

const waiting = document.querySelector(".waiting");
const resultBox = document.querySelector(".result-box");

predictBtn.addEventListener("click", predictTransaction);

async function predictTransaction() {

    const amount = parseFloat(document.getElementById("amount").value);

    if (isNaN(amount) || amount <= 0) {

        alert("Please enter a valid transaction amount.");

        return;

    }

    const data = {

        TransactionAmt: amount,

        ProductCD: document.getElementById("product").value,

        card4: document.getElementById("card4").value,

        card6: document.getElementById("card6").value,

        M4: document.getElementById("m4").value

    };

    waiting.innerHTML = `
        <i class="fa-solid fa-spinner fa-spin"></i>
        <p>Analyzing transaction...</p>
    `;

    resultBox.classList.add("hidden");

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify(data)

        });

        if (!response.ok) {

            throw new Error("Server error.");

        }

        const result = await response.json();

        displayResult(result);

    }

    catch (error) {

        waiting.innerHTML = `
            <i class="fa-solid fa-circle-xmark"></i>
            <p>Unable to connect to backend.</p>
        `;

        console.error(error);

    }

}


function displayResult(result) {

    waiting.style.display = "none";

    resultBox.classList.remove("hidden");

    const probability = result.fraud_probability * 100;

    document.getElementById("probability").innerText =
        probability.toFixed(2) + "%";

    document.getElementById("status").innerText =
        result.status;

    document.getElementById("confidence").innerText =
        result.confidence;

    const recommendation = document.getElementById("recommendation");

    if (result.prediction === 1) {

        recommendation.innerText =
            "High fraud risk detected. Review this transaction before approval.";

    }

    else {

        recommendation.innerText =
            "Transaction appears legitimate.";

    }

    const riskBar = document.getElementById("riskBar");

    riskBar.style.width = probability + "%";

    const status = document.getElementById("status");

    status.classList.remove("safe", "warning", "danger");

    if (probability < 30) {

        riskBar.style.background = "#22C55E";

        status.classList.add("safe");

    }

    else if (probability < 70) {

        riskBar.style.background = "#F59E0B";

        status.classList.add("warning");

    }

    else {

        riskBar.style.background = "#EF4444";

        status.classList.add("danger");

    }

}