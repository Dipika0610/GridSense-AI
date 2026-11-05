# GridSense AI — Hackathon Presentation & Pitch Guide
**PCCOE Indradhanu Hackathon – Better Tomorrow**  
**Team CodeCatalyst | A. P. Shah Institute of Technology**

---

## 🎙️ 3-Minute Pitch Script

### **1. Hook & Problem (0:00 - 0:45)**
> *"Good morning respected judges. Did you know that in many solar-rich cities, up to 25% of clean electricity is simply dumped or curtailed every single day, while at 7 PM, utilities are forced to switch on dirty fossil-fuel peaker plants?*
> 
> *The root cause isn't a lack of green energy; it's a lack of intelligent synchronization. Current systems manage forecasting, battery storage, and demand in isolated silos. This leads to the infamous **Battery Paradox**—where static batteries blindly charge during cheap midnight hours powered by dirty coal, actually increasing net carbon emissions!"*

### **2. The Solution (0:45 - 1:30)**
> *"Meet **GridSense AI**: Unified Machine Learning for Sustainable Electricity. Our motto is: **One AI Core. Two Operational Arms. One Greener Grid.***
> 
> *Instead of separate fragmented systems, GridSense AI employs a single shared ML intelligence core that forecasts 24 hours ahead across three critical signals: renewable generation, time-of-use tariffs, and dynamic grid carbon intensity.*
> 
> *These shared predictions simultaneously drive two coordinated arms:
> - **Module A (Supply Arm):** A Linear Programming optimizer that dispatches our battery strictly on clean power and peak-shaves high-tariff hours.
> - **Module B (Demand Arm):** A Mixed-Integer scheduler that automatically shifts flexible loads—like EV fleet charging and data center compute—straight into ultra-clean mid-day valleys."*

### **3. Live Demo & Impact (1:30 - 2:30)**
> *(Switch to the Streamlit Dashboard)*
> *"Here on our dashboard, you can see our 3-layer architecture live:*
> - *Layer 1 shows our Random Forest and Gradient Boosted models predicting generation, pricing, and carbon curves with an R² exceeding 0.98.*
> - *Layer 2A visualizes our battery SoC, completely eliminating dirty night charging.*
> - *Layer 2B shows our EV fleets and data center AI training jobs seamlessly re-allocated into the solar peak.*
> - *The bottom line? Compared to the traditional naive baseline, GridSense AI achieves an **18.6% utility bill savings**, a **36.6% carbon emissions reduction**, and avoids **over 660 kWh of renewable curtailment** every single day."*

### **4. Conclusion (2:30 - 3:00)**
> *"GridSense AI turns intermittent renewables into reliable, dispatchable, cost-effective power for smart cities and campuses. Thank you, and we welcome your questions!"*

---

## ❓ Frequently Asked Judge Questions & Answers

#### Q1: "What is the Battery Paradox and how does your model mathematically prevent it?"
> **Answer:** "The Battery Paradox occurs when an energy storage system charges based purely on low tariff rates without accounting for the marginal grid emissions. For instance, at 2 AM, electricity might be cheap, but 80% of it comes from coal baseload. Accounting for a 15% roundtrip efficiency loss, cycling the battery actually adds emissions.  
> We mathematically solve this by adding a dynamic carbon-intensity constraint in our LP formulation ($A_{ub} x \le b_{ub}$). If the grid carbon intensity exceeds the clean threshold, battery charging is restricted strictly to local renewable generation surplus ($P_{\text{charge}, t} \le \max(0, G_t - D_{\text{base}, t})$), guaranteeing 0% dirty charging."

#### Q2: "Why do you use a shared forecasting core instead of separate models?"
> **Answer:** "Energy supply and demand interact simultaneously. If the battery controller and the EV charging scheduler use different weather and tariff forecasts, they can make conflicting decisions—such as the battery discharging while EV chargers draw high power, tripping peak demand limits. A shared core guarantees mathematical consistency across both operational arms."

#### Q3: "What happens if weather conditions deviate suddenly from the forecast?"
> **Answer:** "Our pipeline is formulated for receding-horizon Model Predictive Control (MPC). While we forecast 24 hours ahead for planning, the schedule can re-solve every 15 to 60 minutes as fresh live readings arrive from Open-Meteo or local sensors, correcting for any intra-hour variance."

