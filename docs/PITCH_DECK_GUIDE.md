# GridSense AI — Hackathon Presentation & Pitch Guide (India Edition)
**PCCOE Indradhanu Hackathon – Better Tomorrow**  
**Team CodeCatalyst | A. P. Shah Institute of Technology**

---

## 🎙️ 3-Minute Pitch Script

### **1. Hook & Problem (0:00 - 0:45)**
> *"Good morning respected judges. In solar-rich Indian states like Maharashtra and Gujarat, commercial and academic campuses waste up to 25% of their daytime rooftop solar generation through curtailment. Yet, between 6:00 PM and 10:00 PM, when clean generation drops, facilities are hit with severe MSEDCL Time-of-Day peak surcharges of over ₹10.30 per kilowatt-hour, powered by dirty thermal peaker plants.*
> 
> *Current building management tools treat forecasting, battery storage, and load scheduling in isolated silos. This causes what we call the **Indian Battery Paradox**: fixed timers charge commercial batteries at 2:00 AM because tariffs are low. But in India, nighttime grid baseload is over 80% coal (~760 gCO₂/kWh). Factoring in 15% roundtrip conversion loss, charging the battery on night coal actually **increases net CO₂ emissions**!"*

### **2. The Solution (0:45 - 1:30)**
> *"To solve this, our team, **CodeCatalyst**, developed **GridSense AI**: Unified Machine Learning for Sustainable Electricity. Our operating motto is: **One AI Core. Two Operational Arms. One Greener Grid.***
> 
> *Instead of fragmented systems, GridSense AI uses a single unified ML forecasting core that looks 24 hours ahead across three simultaneous signals: captive solar & wind generation, MSEDCL Time-of-Day tariffs, and CEA National Grid carbon intensity.*
> 
> *These shared predictions simultaneously drive two coordinated arms:
> - **Module A (Supply Arm):** A Linear Programming optimizer that dispatches our battery strictly on captive solar surplus and discharges during the 6:00 PM to 10:00 PM peak ToD window, completely eliminating dirty night coal charging.
> - **Module B (Demand Arm):** A Mixed-Integer scheduler that automatically shifts flexible loads—like commercial EV fleet charging depots, campus compute servers, APMC cold storage, and MIDC water pumps—straight into midday solar troughs."*

### **3. Live Demo & Impact (1:30 - 2:30)**
> *(Switch to the Streamlit Dashboard)*
> *"Here on our dashboard, built with an editorial monochrome aesthetic for clean utility operation:*
> - *Layer 1 demonstrates our Random Forest and Gradient Boosted models predicting generation, pricing, and carbon curves with R² exceeding 0.98.*
> - *Layer 2A visualizes our battery SoC, showing 100% clean solar charging and peak discharge.*
> - *Layer 2B shows commercial EV fleet charging and water pumps neatly shifted into solar hours.*
> - *The financial and environmental return? Over a 24-hour cycle on our simulated campus, GridSense AI saves **₹1,908 per day (a 12.1% electricity bill reduction)**, cuts **285 kg of CO₂ emissions (a 17.9% reduction)**, and avoids **over 220 kWh of clean solar curtailment**."*

### **4. Conclusion (2:30 - 3:00)**
> *"GridSense AI turns variable renewables into reliable, dispatchable, cost-effective power tailored for Indian smart cities and campus microgrids. Thank you, and we welcome your questions!"*

---

## ❓ Frequently Asked Judge Questions & Answers

#### Q1: "What is the Indian Battery Paradox and how does your model mathematically prevent it?"
> **Answer:** "The Battery Paradox occurs when an energy storage system charges based purely on low tariff rates without accounting for the marginal grid emission factor. Under MSEDCL ToD tariffs, night power (22:00 to 06:00) has an off-peak rebate of -₹1.50/kWh. But in India, overnight power is 80%+ coal baseload with an emission factor of ~760 gCO₂/kWh. Accounting for a 15% roundtrip efficiency loss, cycling the battery on midnight power actually adds net carbon emissions.  
> We mathematically solve this by adding a clean-charging inequality constraint in our LP formulation ($A_{\text{ub}} x \le b_{\text{ub}}$). Whenever grid carbon intensity exceeds the clean threshold, battery charging is restricted strictly to local rooftop solar surplus ($P_{\text{charge}, t} \le \max(0, G_t - D_{\text{base}, t})$), guaranteeing 0% dirty coal charging."

#### Q2: "How is GridSense AI tailored specifically for the Indian power landscape?"
> **Answer:** "We built our models around real Indian regulatory data:
> 1. Tariffs follow MERC / MSEDCL Commercial & Industrial Time-of-Day (ToD) slots with morning and evening peak surcharges.
> 2. Carbon intensity is calibrated using the Central Electricity Authority (CEA) Version 19 National Grid database.
> 3. Flexible loads are modeled on real Indian institutional demands: commercial 3W/4W EV fleet depots, APMC agricultural cold storage, tech park compute clusters, and MIDC water pumping stations."

#### Q3: "Why did you choose a White & Black minimalist design for the UI?"
> **Answer:** "Most hackathon dashboards rely on loud rainbow gradients and neon colors that look artificial. Utility operators, grid engineers, and facility directors demand high-contrast, uncluttered, legible interfaces with fast scanning speeds, crisp typography, and high data density. Our monochrome editorial design reflects industrial-grade engineering software."
