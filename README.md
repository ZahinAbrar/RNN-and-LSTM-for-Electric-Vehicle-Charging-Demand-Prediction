
# ⚡ Electric Vehicle Charging Demand Prediction using RNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![DeepLearning](https://img.shields.io/badge/Deep%20Learning-RNN-yellow?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)

Predict the future charging demand of electric vehicles (EVs) using a Recurrent Neural Network (RNN). This project helps smart grid operators optimize energy distribution and plan charging infrastructure more effectively.

---

## 🚗 Problem Overview

With the rapid adoption of EVs, it’s crucial to forecast energy needs at charging stations to:
- Prevent overloading the grid
- Optimize energy distribution
- Ensure availability of charging infrastructure

---

## 📊 Dataset

Each data point includes:
- **Timestamp** – the time of observation
- **Charging Demand** – number of EVs charging during that time window

---

## 🧠 Model Architecture

### 🔁 Why RNN?

RNNs are well-suited for time-series forecasting due to their memory of previous states, allowing the model to understand sequential patterns.

### 📐 Architecture Summary

- **Input Layer** – sequences of past demand values
- **RNN or LSTM Layer** – 50 units
- **Dense Layer** – 1 unit for the predicted demand

```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, input_shape=(timesteps, features)),
    tf.keras.layers.Dense(1)
])
```

---

## 📈 Results & Insights

- The RNN effectively learned temporal demand patterns.
- Enabled predictive energy management strategies.
- Can be scaled to include weather, time-of-day, and pricing features.

---

## 🛠️ Future Work

- Replace RNN with **LSTM** or **GRU** for longer sequences
- Add external features like **weather** or **traffic conditions**
- Integrate into **real-time grid management system**

---

## 📂 Folder Structure

```
.
├── data/
├── model/
├── notebook/
├── utils/
└── README.md
```

---

## 🤝 Contributions

Feel free to fork and contribute to improve the prediction model or adapt it for other time-series forecasting tasks!

---

## 📄 License

This project is licensed under the MIT License.
