# GridSense AI

> **Unified Machine Learning for Sustainable Electricity**  
> *"One AI Core. Two Operational Arms. One Greener Grid."*

---

### 👥 Team & Hackathon Details
- **Team Name:** CodeCatalyst
- **Team Members:**
  - Dipika Vijay Bhadane
  - Pritesh Kalpesh Dhulla
  - Dhanashree Prakash Desale
- **College Name:** A. P. Shah Institute Of Technology
- **Topic:** Smart Cities, Energy & Circular Economy
- **Hackathon:** PCCOE Indradhanu Hackathon – Better Tomorrow
- **Regional Grid Context:** Maharashtra / Western India Grid (MSEDCL & CEA Standards)

---

## ⚡ The Problem & The Challenge

### The Problem
In India, renewable energy (rooftop solar and captive wind) is intermittent and often mismatched with commercial/campus demand. During midday (11:00 AM – 3:00 PM), intense solar output often exceeds local facility demand, causing clean energy wastage (**curtailment**). Conversely, between 6:00 PM and 10:00 PM, clean generation drops while peak demand surges, forcing utilities to dispatch coal-fired peaker plants under high Time-of-Day (ToD) tariff penalties.

Traditional systems treat **forecasting**, **battery storage**, and **demand scheduling** as disconnected, siloed modules, leading to two critical systemic failures in India:
1. **The Indian Battery Paradox:** Fixed timers charge commercial batteries during night off-peak tariff hours (01:00 AM – 05:00 AM) to capture the ₹6.30/kWh off-peak ToD rate. However, India's overnight grid baseload is over 80% coal (~760 gCO₂/kWh). When accounting for round-trip battery conversion losses (~15%), cycling on night coal actually **increases net carbon emissions**!
2. **Blind Plug-in Loads:** Commercial EV fleet delivery vehicles, agricultural/municipal pumps, and campus batch computing run immediately upon plug-in, coinciding with high-carbon evening hours and triggering severe MSEDCL demand surcharges (₹10.30+/kWh).

### Our Challenge
> *How can AI intelligently predict, store, and schedule electricity to maximize clean-energy use while reducing electricity bills in Rupees (₹) and carbon emissions?*

---

## 💡 The Proposed Solution: GridSense AI

GridSense AI unifies forecasting, supply-side battery storage, and demand-side load shifting into a single closed-loop decision platform:

```
        Predict (Layer 1: ML Core)
                   ↓
        Optimize (Layer 2: Two Arms)
        ↙                       ↘
  Supply-Side Storage       Demand-Side Schedulers
  (Battery Arbitrage LP)     (Carbon-Aware Load MILP)
        ↘                       ↙
           Act (Grid Execution)
                   ↓
       Measure (Layer 3: Dashboard)
```

### System Architecture: 3-Layer Design

```
+-----------------------------------------------------------------------------------+
|  DATA INGESTION: Weather APIs (Open-Meteo, NISE) & Grid APIs (MSEDCL, CEA India)  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|  LAYER 1: SHARED FORECASTING CORE                                                 |
|  - Multi-target ML: Solar (kW), Wind (kW), Tariffs (₹/kWh), Carbon (gCO₂/kWh)     |
|  - Models: Random Forest, SVR, Neural MLP Regressors                              |
+-----------------------------------------------------------------------------------+
                     │                                         │
                     ▼                                         ▼
+---------------------------------------+   +---------------------------------------+
|  LAYER 2A: SUPPLY-SIDE OPTIMIZATION   |   |  LAYER 2B: DEMAND-SIDE SCHEDULING     |
|  (Module A: Battery Storage)          |   |  (Module B: Load Shifter)             |
|  - HiGHS Linear Programming (LP)      |   |  - Mixed-Integer LP (MILP)            |
|  - Arbitrage: Charge clean solar,     |   |  - Shifts EV Fleets, Compute Clusters,|
|    discharge during evening peak ToD  |   |    APMC Cold Storage & Water Pumps    |
|  - Eliminates 'Battery Paradox'       |   |  - Aligns loads with solar troughs    |
+---------------------------------------+   +---------------------------------------+
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|  LAYER 3: FRONTEND & IMPACT DASHBOARD                                             |
|  - Minimalist, Editorial Monochrome (White & Black) User Interface                |
|  - Humanized, uncluttered engineering telemetry with responsive Plotly charts     |
|  - Empirical comparison metrics vs. Naive Baseline in Indian Rupees (₹)           |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Results & Impact vs. Naive Baseline (24-Hour Horizon)

| Performance Metric | Naive Baseline (Unmanaged) | GridSense AI (Unified System) | Net Benefit |
| :--- | :--- | :--- | :--- |
| **Electricity Utility Bill** | ₹15,812.97 | **₹13,904.81** | **₹1,908.16 (12.1% Saved)** |
| **Grid Carbon Footprint** | 1,593.40 kg CO₂ | **1,308.14 kg CO₂** | **285.25 kg CO₂ (17.9% Reduction)** |
| **Renewable Curtailment** | 514.05 kWh wasted | **290.06 kWh** | **224.0 kWh Clean Energy Saved** |
| **Peak Grid Demand** | 289.26 kW | **166.68 kW** | **122.6 kW (42.4% Peak Shaving)** |
| **The Battery Paradox** | **Failed** (Charges on night coal) | **Solved** (100% clean solar charge) | **368.9 kWh Dirty Charging Blocked** |

---

## 🛠️ Project Structure

```
GridsenseAI/
├── README.md                      # Complete project documentation & hackathon submission guide
├── requirements.txt               # Python package dependencies
├── run_demo.py                    # Standalone CLI runner with full Indian grid simulation
├── data/
│   ├── __init__.py
│   ├── generator.py               # Western India solar/wind, MSEDCL ToD tariffs & CEA carbon data synthesizer
│   └── weather_grid_client.py     # Live Open-Meteo API connector (Pune/PCCOE coordinates) with offline fallback
├── core/
│   ├── __init__.py
│   ├── forecasting.py             # Layer 1: Multi-target ML models (Random Forest, SVR, MLP)
│   ├── battery_optimizer.py       # Layer 2A: Battery dispatch LP (HiGHS solver)
│   ├── load_scheduler.py          # Layer 2B: Carbon-aware load scheduling MILP
│   └── impact_engine.py           # System-level KPI engine vs. Naive Baseline in Rupees (₹)
├── dashboard/
│   ├── __init__.py
│   └── app.py                     # Layer 3: Minimalist White & Black Streamlit & Plotly Dashboard
└── tests/
    ├── __init__.py
    └── test_pipeline.py           # Automated unit and integration test suite
```

---

## 💻 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Standalone CLI Pipeline
```bash
python run_demo.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Verification Tests
```bash
pytest -v tests/test_pipeline.py
```

---

## 📚 Indian Power Sector Standards & References

1. **Central Electricity Authority (CEA)** — *CO₂ Baseline Database for the Indian Power Sector, Version 19*, Ministry of Power, Government of India.
2. **Maharashtra Electricity Regulatory Commission (MERC)** — *Time of Day (ToD) Tariff Regulations & Commercial Tariffs for MSEDCL*.
3. **National Institute of Solar Energy (NISE)** — *Solar Radiation Resource Assessment (SRRA) Data for Western India*.
4. **COIN-OR & SciPy HiGHS** — *High Performance Simplex and Interior Point Solvers for Linear Energy Optimization*.
