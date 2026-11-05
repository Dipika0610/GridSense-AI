"""
GridSense AI - End-to-End Simulation Runner
Runs the full pipeline via CLI:
1. Historical Data Synthesis
2. Layer 1: Multi-Model Forecasting (Training & Evaluation)
3. Layer 2A: Battery Storage LP Optimization (Anti-Battery-Paradox)
4. Layer 2B: Carbon-Aware Load Scheduling (MILP)
5. Layer 3: System Impact vs. Naive Baseline Evaluation
"""

import sys
import os
import pandas as pd
import numpy as np

from data.generator import EnergyDataGenerator
from core.forecasting import ForecastingCore
from core.battery_optimizer import BatteryOptimizer
from core.load_scheduler import CarbonAwareLoadScheduler
from core.impact_engine import ImpactEngine


def main():
    print("=" * 75)
    print("  ⚡ GRIDSENSE AI - UNIFIED ML FOR SUSTAINABLE ELECTRICITY ⚡")
    print("  One AI Core. Two Operational Arms. One Greener Grid.")
    print("  Team: CodeCatalyst | PCCOE Indradhanu Hackathon")
    print("=" * 75)
    print()

    # Step 1: Data Generation
    print("[1/5] Synthesizing 60-day historical training dataset...")
    gen = EnergyDataGenerator(seed=42)
    hist_df = gen.generate_historical_dataset(days=60)
    print(f"      -> Generated {len(hist_df)} hourly records across weather, solar, wind, price, and carbon intensity.")

    # Step 2: Layer 1 Forecasting Core
    print("\n[2/5] Training Layer 1: Shared Forecasting Core (Random Forest & Ensemble)...")
    forecaster = ForecastingCore(model_type="random_forest")
    metrics = forecaster.train(hist_df)
    
    print("      Model Performance Metrics:")
    for target, m in metrics.items():
        print(f"      - {target:25s} | R²: {m['r2']:0.3f} | MAE: {m['mae']:0.3f} | RMSE: {m['rmse']:0.3f}")

    # Generate 24-hour decision horizon
    print("\n      Generating 24-hour operational decision horizon...")
    horizon_df = gen.generate_forecast_horizon(hours=24, scenario="normal")
    forecast_df = forecaster.predict(horizon_df)
    print("      -> Forecast predictions generated for all grid and generation vectors.")

    # Step 3: Layer 2A Battery Optimization
    print("\n[3/5] Solving Layer 2A: Battery Storage Optimization (PuLP LP)...")
    battery_opt = BatteryOptimizer(
        capacity_kwh=600.0,
        max_power_kw=200.0,
        soc_initial_pct=0.25,
        roundtrip_efficiency=0.90
    )
    dispatch_df = battery_opt.optimize_dispatch(forecast_df, alpha_cost=0.55, enforce_clean_charging=True)
    naive_dispatch_df = battery_opt.simulate_naive_baseline(forecast_df)
    print(f"      -> Optimal battery schedule computed.")
    print(f"      -> Total Stored: {dispatch_df['battery_charge_kw'].sum():.1f} kWh | Clean Energy Share: {dispatch_df['battery_clean_charge_kw'].sum() / max(1, dispatch_df['battery_charge_kw'].sum()) * 100:.1f}%")
    print(f"      -> Total Discharged: {dispatch_df['battery_discharge_kw'].sum():.1f} kWh during peak tariff & dirty peaker hours.")

    # Step 4: Layer 2B Demand Load Scheduling
    print("\n[4/5] Solving Layer 2B: Demand Load Scheduling (PuLP MILP)...")
    scheduler = CarbonAwareLoadScheduler(facility_max_flexible_kw=200.0)
    schedule_df = scheduler.schedule_loads(forecast_df, alpha_cost=0.55)
    naive_schedule_df = scheduler.simulate_naive_schedule(forecast_df)
    print(f"      -> Flexible loads (EV Fleet, AI Training, HVAC Cold Storage, Pumps) scheduled.")
    print(f"      -> Total Shifted Demand: {schedule_df['total_flexible_demand_kw'].sum():.1f} kWh moved into carbon valleys.")

    # Step 5: Impact Evaluation
    print("\n[5/5] Evaluating System Impact vs. Naive Baseline...")
    impact_engine = ImpactEngine()
    impact = impact_engine.evaluate_system(
        dispatch_df, schedule_df, naive_dispatch_df, naive_schedule_df
    )

    savings = impact["savings"]
    gridsense = impact["gridsense"]
    naive = impact["naive"]

    print("\n" + "=" * 75)
    print("                     🏆 SYSTEM IMPACT COMPARISON 🏆")
    print("=" * 75)
    print(f"{'Performance Metric':<32} | {'Naive Baseline':<16} | {'GridSense AI':<16} | {'Net Savings / Gain'}")
    print("-" * 75)
    print(f"{'Total Electricity Bill ($)':<32} | ${naive['total_cost']:<15.2f} | ${gridsense['total_cost']:<15.2f} | -${savings['cost_saved']:0.2f} ({savings['cost_saved_pct']:0.1f}%)")
    print(f"{'Carbon Emissions (kg CO2)':<32} | {naive['total_carbon_kg']:<15.2f} kg | {gridsense['total_carbon_kg']:<15.2f} kg | -{savings['carbon_saved_kg']:0.2f} kg ({savings['carbon_saved_pct']:0.1f}%)")
    print(f"{'Renewable Curtailment (kWh)':<32} | {naive['curtailment_kwh']:<15.2f} kWh| {gridsense['curtailment_kwh']:<15.2f} kWh| Avoided {savings['curtailment_avoided_kwh']:0.1f} kWh")
    print(f"{'Peak Grid Import (kW)':<32} | {naive['peak_grid_kw']:<15.2f} kW | {gridsense['peak_grid_kw']:<15.2f} kW | Shaved {savings['peak_reduction_kw']:0.1f} kW ({savings['peak_reduction_pct']:0.1f}%)")
    print(f"{'Battery Paradox Avoided':<32} | {'FAILED (Dirty)':<16} | {'SOLVED (Clean)':<16} | {savings['battery_paradox_prevented_kwh']:0.1f} kWh dirty charge blocked")
    print("=" * 75)

    print("\n✨ GridSense AI simulation completed successfully!")
    print("👉 To launch the interactive dashboard, run: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()

