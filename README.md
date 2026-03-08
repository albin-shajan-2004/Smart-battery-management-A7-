# ⚡ Battery Intelligence System — Deep Learning & RL Based BMS Dashboard

<div align="center">

![Battery Management System](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20GRU-00d4ff?style=for-the-badge&logo=tensorflow)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-00ff9d?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-ffcc00?style=for-the-badge&logo=python)
![HTML](https://img.shields.io/badge/Dashboard-HTML%20%7C%20Chart.js-ff6b35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-white?style=for-the-badge)

**A final year engineering project combining Deep Learning and Reinforcement Learning for real-time Battery State Estimation with an interactive EV Drive Cycle Dashboard.**

[🚀 Live Demo](#live-demo) · [📊 Features](#features) · [🧠 Models](#models) · [📁 Dataset](#dataset) · [⚙️ Installation](#installation) · [📸 Screenshots](#screenshots)

</div>

---

## 📌 Project Overview

This project presents an intelligent **Battery Management System (BMS)** that estimates three critical battery health parameters in real time:

| Parameter | Full Name | Description |
|---|---|---|
| **SoC** | State of Charge | Remaining battery charge (%) |
| **SoH** | State of Health | Battery capacity retention over cycles (%) |
| **RUL** | Remaining Useful Life | Predicted remaining charge-discharge cycles |

The system uses **LSTM** and **GRU** deep learning architectures trained on real LG 18650 battery data, with a **Reinforcement Learning** agent optimizing the charge/discharge policy. An interactive web dashboard provides real-time simulation of an EV drive cycle with live predictions.

---

## ✨ Features

### 🚗 EV Drive Cycle Simulation
- Animated car simulation on a road with real-time battery depletion
- Battery bar on the car changes color: **green → yellow → red** as SoC drops
- Live SOC chart updating every tick during simulation
- Playback controls: Play / Pause / Reset / Speed (1x–20x)
- Toggle between **LSTM**, **GRU**, and **Actual** predictions on the fly
- Real UDDS (Urban Dynamometer Driving Schedule) drive cycle data

### 📊 BMS Dashboard
- Four animated arc **gauges** for SoC, SoH, RUL, and Temperature
- Full drive cycle **SOC comparison** — LSTM vs GRU vs Actual
- **SOH degradation** curve over the full drive cycle
- **RUL countdown** curve showing remaining useful life
- LSTM **training history** (40 epochs, logarithmic loss scale)

### 🧠 Model Analysis
- **Bar + line combo chart**: RMSE and R² for all 12 LSTM/GRU configurations
- **Bubble chart**: Training time vs accuracy vs batch size
- Color-coded **epoch × batch size heatmap** with best config highlighted
- Full sortable **results table** with model badges

### 🔧 Hyperparameter Optimization
- **Optimizer × dropout** performance heatmap (Adam, Adagrad, RMSprop, SGD)
- **Sequence length × learning rate** comparison bar chart
- Animated progress bars for best configuration summary
- Interactive hover tooltips on all charts

---

## 🧠 Models

### Deep Learning Architectures

#### LSTM (Long Short-Term Memory)
```
Input Layer  →  LSTM (units=128)  →  Dropout  →  LSTM (units=64)  →  Dense(1)
```
- Trained on sequences of Voltage, Current, Temperature, Time
- Best configuration: **Epochs=30, Batch=256, R²=0.99941**
- Optimizer: **Adam**, Learning Rate: **0.001**

#### GRU (Gated Recurrent Unit)
```
Input Layer  →  GRU (units=128)  →  Dropout  →  GRU (units=64)  →  Dense(1)
```
- Computationally lighter than LSTM with comparable accuracy
- Best configuration: **Epochs=15, Batch=256, R²=0.99942**
- Best RMSE: **0.00672** — slightly outperforms LSTM

### Model Performance Summary

| Model | Config | RMSE | MAE | R² | Train Time |
|---|---|---|---|---|---|
| **GRU** ⭐ | E15 B256 | **0.00672** | 0.00545 | **0.99942** | 405s |
| LSTM | E30 B64 | 0.00679 | 0.00543 | 0.99941 | 2993s |
| LSTM | E30 B256 | 0.00682 | 0.00529 | 0.99940 | 780s |
| LSTM | E15 B64 | 0.00701 | 0.00533 | 0.99937 | 1493s |

### Reinforcement Learning Agent
- **Environment**: Custom battery charge/discharge cycle environment
- **Algorithm**: Deep Q-Network (DQN) with experience replay
- **State space**: SoC, SoH, Voltage, Current, Temperature
- **Action space**: Charge rate selection (discrete)
- **Reward**: Maximizing battery life while maintaining SoC bounds

---

## 📊 Hyperparameter Tuning Results

### Optimizer Comparison (GRU, Epochs=15, Batch=256)

| Optimizer | Dropout | RMSE | R² | Best? |
|---|---|---|---|---|
| **Adam** | 0.1 | **0.01152** | **0.99831** | ⭐ |
| Adam | 0.2 | 0.01191 | 0.99820 | |
| RMSprop | 0.1 | 0.01890 | 0.99546 | |
| RMSprop | 0.2 | 0.01895 | 0.99543 | |
| SGD | 0.1 | 0.04209 | 0.97747 | |
| Adagrad | 0.1 | 0.04374 | 0.97566 | |

### Sequence Length × Learning Rate

| SeqLen | LR | RMSE | R² |
|---|---|---|---|
| 200 | 0.001 | **0.04738** | **0.97093** |
| 50 | 0.001 | 0.05439 | 0.96238 |
| 100 | 0.0001 | 0.05766 | 0.95746 |
| 50 | 0.0001 | 0.06827 | 0.94072 |

---

## 📁 Dataset

**Source**: LG 18650HG2 Li-ion Battery Dataset  
**Conditions**: Tested at multiple temperatures (10°C and 25°C)  
**Drive Cycles Used**:

| File | Description | Rows |
|---|---|---|
| `551_UDDS_processed.csv` | Urban Dynamometer Driving Schedule | 15,967 |
| `549_Charge_processed.csv` | Standard charge cycle | ~1,800 |
| Various raw files | Mixed, HWFET, US06, Charge cycles | varies |

**Features used for model input**:
```
Voltage [V]  |  Current [A]  |  Temperature [°C]  |  Time [s]
```

**Target outputs**:
```
SOC [-]  |  SOH [-]  |  RUL [cycles]
```

---

## 📂 Project Structure

```
Main_Project/
│
├── Code/
│   ├── Epoch_Batch.ipynb          # Epoch & batch size tuning
│   ├── opt_drp_lstm.ipynb         # Optimizer & dropout tuning (LSTM)
│   ├── opt_drp_gru.ipynb          # Optimizer & dropout tuning (GRU)
│   ├── seqlen_LR_gru.ipynb        # Sequence length & LR tuning
│   ├── RL_pred_soc_soh.ipynb      # Reinforcement Learning agent
│   └── gru_lstm_pred.ipynb        # Final LSTM vs GRU comparison
│
├── Dataset/LG_1/
│   ├── raw/                       # Raw battery cycle CSV files
│   ├── eval/                      # Evaluation datasets (UDDS, Charge)
│   ├── models/                    # Trained .keras and .h5 model files
│   ├── scalers/                   # MinMaxScaler pickle files
│   └── history/                   # Training history .npy files
│
├── Results/
│   ├── Optimizer(adam,adagrad,rms,sdg)_droput(0.1,0.2).csv
│   ├── Epoch_Batch_LSTM_GRU(15,30)(64,128,256).csv
│   └── seq_len(50,100,200)_lr(0.1,0.001.0.0001).csv
│
├── Graph/
│   ├── DL/                        # Deep learning prediction plots
│   └── RL/                        # Reinforcement learning plots
│
└── BMS_Dashboard.html             # ⭐ Interactive dashboard (open in browser)
```

---

## ⚙️ Installation

### Option 1 — Run the HTML Dashboard (No Installation)
Just download `BMS_Dashboard.html` and open it in any browser. No setup required.

```bash
# Simply double-click BMS_Dashboard.html
# Or open in browser: File → Open → BMS_Dashboard.html
```

### Option 2 — Run Jupyter Notebooks

**Prerequisites**
```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/battery-bms-dashboard.git
cd battery-bms-dashboard
```

**Launch Jupyter**
```bash
jupyter notebook
```
Then open any notebook inside the `Code/` folder.

### Option 3 — Plotly Dash App (Full Real-Time Inference)

```bash
pip install dash plotly tensorflow pandas scikit-learn numpy
python app.py
```
Open browser → `http://127.0.0.1:8050`

---

## 🖥️ Usage

### HTML Dashboard
1. Open `BMS_Dashboard.html` in Chrome / Firefox / Edge
2. Navigate using the 4 tabs at the top:
   - **🚗 EV Simulation** — Watch the car drive and battery deplete live
   - **📊 BMS Dashboard** — Full gauges and prediction charts
   - **🧠 Model Analysis** — Compare all LSTM/GRU configurations
   - **🔧 Hyperparameters** — Explore optimizer and tuning results
3. On the Simulation tab:
   - Press **▶ PLAY** to start the drive cycle
   - Adjust **SPEED** slider to control simulation pace
   - Toggle **LSTM / GRU / ACTUAL** to switch prediction models

### Jupyter Notebooks
Run notebooks in this order for full reproducibility:
```
1. Epoch_Batch.ipynb          → Find best epoch/batch combo
2. opt_drp_lstm.ipynb         → Tune optimizer and dropout for LSTM
3. opt_drp_gru.ipynb          → Tune optimizer and dropout for GRU
4. seqlen_LR_gru.ipynb        → Tune sequence length and learning rate
5. gru_lstm_pred.ipynb        → Final model predictions and comparison
6. RL_pred_soc_soh.ipynb      → Train and evaluate RL agent
```

---

## 📈 Results

### Key Findings

- **GRU outperforms LSTM** slightly in accuracy (R²=0.99942 vs 0.99941) while being faster to train
- **Adam optimizer** is clearly superior to SGD and Adagrad for this task
- **Batch size 256** offers the best speed-accuracy tradeoff
- **Sequence length 200 with LR=0.001** gives highest R² for SOC prediction
- The **RL agent** successfully learns a charge policy that extends battery life compared to baseline

### LSTM Training Convergence
- Training loss drops from **0.0122 → 0.00128** over 40 epochs
- Validation loss: **0.0199 → 0.00380** — no significant overfitting
- Learning rate reduced at epoch 33: **0.001 → 0.0005**

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Reinforcement Learning | Custom DQN (TensorFlow) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | HTML5, CSS3, Chart.js 4.4 |
| Development | Jupyter Notebook, VS Code |
| Dataset | LG 18650HG2 Battery (McMaster University) |

---

## 🔮 Future Work

- [ ] Deploy Plotly Dash app to Hugging Face Spaces or Render
- [ ] Integrate real-time serial port data from physical battery test bench
- [ ] Add Transformer-based model (attention mechanism) for comparison
- [ ] Implement online learning — model updates as new data arrives
- [ ] Add SOH estimation with capacity fade modelling
- [ ] Export predictions as PDF report from dashboard

---

## 👤 Author

**Albin**  
Final Year B.Tech / B.E. — Electrical / Electronics Engineering  
📧 your.email@college.edu  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **LG 18650HG2 Battery Dataset** — McMaster University Battery Research Group
- **TensorFlow / Keras** — Open source deep learning framework
- **Plotly / Dash** — Interactive visualization library
- **Chart.js** — HTML5 charting library used in the dashboard

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ as a Final Year Project

</div>
