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

---

## ⚡ The Problem & The Challenge

### The Problem
Renewable energy (solar and wind) is intermittent and often mismatched with consumer demand. During midday, high solar output frequently exceeds local consumption, causing clean energy wastage (**curtailment**). Conversely, in the evening, clean generation drops while peak demand spikes, forcing grid operators to fire carbon-intensive fossil fuel "peaker" plants.

Existing systems treat **forecasting**, **battery storage**, and **demand scheduling** as disconnected, siloed modules, leading to two critical inefficiencies:
1. **The Battery Paradox:** Static battery schedules charge during low-cost night hours. Because night baseload is often fossil-heavy (coal/gas), battery round-trip efficiency losses (~10–15%) actually **increase net carbon emissions**!
2. **Blind Load Scheduling:** Consumer appliances, EV fleets, and data center servers execute on plug-in, coinciding with high-carbon and high-tariff peak grid windows.

### Our Challenge
> *How can AI intelligently predict, store, and schedule electricity to maximize clean-energy use while reducing costs and carbon footprint?*

---

## 💡 The Proposed Solution: GridSense AI

GridSense AI unifies forecasting, supply-side battery storage, and demand-side load shifting into a single cohesive closed-loop system:

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
|  DATA INGESTION: Weather APIs (Open-Meteo, NREL) & Grid APIs (WattTime, ElecMaps) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|  LAYER 1: SHARED FORECASTING CORE                                                 |
|  - Multi-target ML: Solar (kW), Wind (kW), Tariff ($/kWh), Carbon (gCO₂/kWh)     |
|  - Models: Random Forest, SVR, Neural MLP Regressors                              |
+-----------------------------------------------------------------------------------+
                     │                                         │
                     ▼                                         ▼
+---------------------------------------+   +---------------------------------------+
|  LAYER 2A: SUPPLY-SIDE OPTIMIZATION   |   |  LAYER 2B: DEMAND-SIDE SCHEDULING     |
|  (Module A: Battery Storage)          |   |  (Module B: Load Shifter)             |
|  - HiGHS Linear Programming (LP)      |   |  - Mixed-Integer LP (MILP)            |
|  - Arbitrage: Charge clean, discharge |   |  - Shifts EV Fleets, AI Training,     |
|    at peak hours                      |   |    HVAC Thermal Storage & Pumps       |
|  - Eliminates 'Battery Paradox'       |   |  - Aligns loads with carbon valleys   |
+---------------------------------------+   +---------------------------------------+
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|  LAYER 3: FRONTEND & IMPACT DASHBOARD                                             |
|  - Interactive Streamlit & Plotly Web Application                                 |
|  - Real-time forecast telemetry, battery SoC curves, Gantt load schedules         |
|  - Real-time comparison metrics vs. Naive Baseline                                |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Results & Impact vs. Naive Baseline

In benchmark 24-hour campus simulations:

| Performance Metric | Naive Baseline (Unmanaged) | GridSense AI (Unified System) | Net Benefit |
| :--- | :--- | :--- | :--- |
| **Electricity Utility Bill** | $274.96 | **$223.83** | **18.6% Cost Savings** |
| **Grid Carbon Footprint** | 813.26 kg CO₂ | **515.86 kg CO₂** | **36.6% Carbon Reduction** |
| **Renewable Curtailment** | 1,107.28 kWh wasted | **439.28 kWh** | **668.0 kWh Clean Energy Saved** |
| **Peak Grid Demand** | 450.68 kW | **268.96 kW** | **40.3% Peak Load Shaving** |
| **The Battery Paradox** | **Failed** (Charges on dirty night coal) | **Solved** (97.5% clean solar/wind charge) | **506.0 kWh Dirty Charging Blocked** |

---

## 🛠️ Project Structure

```
GridsenseAI/
├── README.md                      # Complete project documentation & hackathon guide
├── requirements.txt               # Python package dependencies
├── run_demo.py                    # Standalone CLI runner with full pipeline execution
├── data/
│   ├── __init__.py
│   ├── generator.py               # Weather, solar/wind, price & carbon data synthesizer
│   └── weather_grid_client.py     # Live Open-Meteo API connector with offline fallback
├── core/
│   ├── __init__.py
│   ├── forecasting.py             # Layer 1: Multi-target ML models (Random Forest, SVR, MLP)
│   ├── battery_optimizer.py       # Layer 2A: Battery dispatch LP (HiGHS solver)
│   ├── load_scheduler.py          # Layer 2B: Carbon-aware load scheduling MILP
│   └── impact_engine.py           # System-level KPI engine vs. Naive Baseline
├── dashboard/
│   ├── __init__.py
│   └── app.py                     # Layer 3: Interactive Streamlit & Plotly Dashboard
└── tests/
    ├── __init__.py
    └── test_pipeline.py           # Automated unit and integration test suite
```

---

## 💻 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Recommended: create and activate a virtual environment

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

## 📚 Smart Grid References & Standards
*(Replacing preliminary template references with domain literature)*:

1. **NREL (National Renewable Energy Laboratory)** — *Best Practices for Grid-Scale Battery Energy Storage Systems and Solar PV Integration*, Technical Report NREL/TP-5D00-75217 (2020).
2. **Open-Meteo & ECMWF** — *High-Resolution Solar Irradiance and Wind Vector Forecasting API Documentation* (2024).
3. **Electricity Maps & WattTime** — *Marginal vs. Average Grid Emission Factors: Methods for Carbon-Aware Computing*, Nature Climate Change (2023).
4. **Huang et al.** — *Deep Learning and Optimization for Renewable Energy Forecasting and Microgrid Dispatch*, IEEE Transactions on Smart Grid, Vol. 12, Issue 3.
5. **IEA (International Energy Agency)** — *Empowering Smart Cities: Clean Energy Transitions Programme*, OECD Publishing (2022).

