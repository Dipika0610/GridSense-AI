"""
GridSense AI - End-to-End Simulation Runner (India Edition)
Calibrated for:
- MSEDCL / MERC Time-of-Day (ToD) Electricity Tariffs (₹/kWh)
- CEA National Grid Baseline Carbon Emission Factors (gCO₂/kWh)
- C&I Solar-Wind Hybrid Facility (Pune / Maharashtra Region)
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
    print("=" * 78)
    print("  GRIDSENSE AI — UNIFIED ML FOR SUSTAINABLE ELECTRICITY (INDIA)")
    print("  One AI Core. Two Operational Arms. One Greener Grid.")
    print("  Team: CodeCatalyst | PCCOE Indradhanu Hackathon — Better Tomorrow")
    print("=" * 78)
    print()

    # Step 1: Data Generation
    print("[1/5] Synthesizing 60-day Maharashtra regional solar/wind & grid dataset...")
    gen = EnergyDataGenerator(seed=42)
    hist_df = gen.generate_historical_dataset(days=60)
    print(f"      -> Processed {len(hist_df)} hourly records (ToD Tariffs in ₹/kWh, CEA Carbon in gCO₂/kWh).")

    # Step 2: Layer 1 Forecasting Core
    print("\n[2/5] Training Layer 1: Shared Multi-Target Forecasting Core...")
    forecaster = ForecastingCore(model_type="random_forest")
    metrics = forecaster.train(hist_df)
    
    print("      Model Performance Metrics:")
    for target, m in metrics.items():
        unit = "₹/kWh" if "price" in target else ("gCO₂/kWh" if "carbon" in target else "kW")
        print(f"      - {target:25s} | R²: {m['r2']:0.3f} | MAE: {m['mae']:0.3f} {unit} | RMSE: {m['rmse']:0.3f}")

    print("\n      Forecasting upcoming 24-hour operational decision horizon...")
    horizon_df = gen.generate_forecast_horizon(hours=24, scenario="normal")
    forecast_df = forecaster.predict(horizon_df)
    print("      -> 24-hour predictions generated for Solar, Wind, Tariffs (₹), and Carbon Intensity.")

    # Step 3: Layer 2A Battery Optimization
    print("\n[3/5] Solving Layer 2A: Battery Storage Optimization (Supply Arm)...")
    battery_opt = BatteryOptimizer(
        capacity_kwh=500.0,
        max_power_kw=150.0,
        soc_initial_pct=0.25,
        roundtrip_efficiency=0.90
    )
    dispatch_df = battery_opt.optimize_dispatch(forecast_df, alpha_cost=0.55, enforce_clean_charging=True)
    naive_dispatch_df = battery_opt.simulate_naive_baseline(forecast_df)
    total_stored = dispatch_df['battery_charge_kw'].sum()
    clean_stored = dispatch_df['battery_clean_charge_kw'].sum()
    clean_pct = (clean_stored / total_stored * 100.0) if total_stored > 0 else 100.0
    print(f"      -> Battery dispatch optimized: {total_stored:.1f} kWh stored ({clean_pct:.1f}% clean energy).")
    print(f"      -> Peak discharge: {dispatch_df['battery_discharge_kw'].sum():.1f} kWh shaving evening peaker tariff (18:00 - 22:00).")

    # Step 4: Layer 2B Demand Load Scheduling
    print("\n[4/5] Solving Layer 2B: Demand Load Scheduling (Demand Arm)...")
    scheduler = CarbonAwareLoadScheduler(facility_max_flexible_kw=200.0)
    schedule_df = scheduler.schedule_loads(forecast_df, alpha_cost=0.55)
    naive_schedule_df = scheduler.simulate_naive_schedule(forecast_df)
    print(f"      -> Scheduled Indian C&I loads (Commercial EV fleet, Compute, APMC Cold Storage, Water Pumps).")
    print(f"      -> Shifted {schedule_df['total_flexible_demand_kw'].sum():.1f} kWh of demand into solar troughs.")

    # Step 5: Impact Evaluation
    print("\n[5/5] Evaluating System Impact vs. Traditional Naive Baseline...")
    impact_engine = ImpactEngine()
    impact = impact_engine.evaluate_system(
        dispatch_df, schedule_df, naive_dispatch_df, naive_schedule_df
    )

    savings = impact["savings"]
    gridsense = impact["gridsense"]
    naive = impact["naive"]

    print("\n" + "=" * 78)
    print("                  SYSTEM IMPACT BENCHMARK (24-HOUR HORIZON)")
    print("=" * 78)
    print(f"{'Performance Metric':<32} | {'Naive Baseline':<16} | {'GridSense AI':<16} | {'Net Savings / Gain'}")
    print("-" * 78)
    print(f"{'Total Electricity Bill (₹)':<32} | ₹{naive['total_cost']:<15.2f} | ₹{gridsense['total_cost']:<15.2f} | -₹{savings['cost_saved']:0.2f} ({savings['cost_saved_pct']:0.1f}%)")
    print(f"{'Carbon Emissions (kg CO₂)':<32} | {naive['total_carbon_kg']:<15.2f} kg | {gridsense['total_carbon_kg']:<15.2f} kg | -{savings['carbon_saved_kg']:0.2f} kg ({savings['carbon_saved_pct']:0.1f}%)")
    print(f"{'Renewable Curtailment (kWh)':<32} | {naive['curtailment_kwh']:<15.2f} kWh| {gridsense['curtailment_kwh']:<15.2f} kWh| Avoided {savings['curtailment_avoided_kwh']:0.1f} kWh")
    print(f"{'Peak Grid Import (kW)':<32} | {naive['peak_grid_kw']:<15.2f} kW | {gridsense['peak_grid_kw']:<15.2f} kW | Shaved {savings['peak_reduction_kw']:0.1f} kW ({savings['peak_reduction_pct']:0.1f}%)")
    print(f"{'Battery Paradox Avoidance':<32} | {'Failed (Coal)':<16} | {'Solved (Clean)':<16} | {savings['battery_paradox_prevented_kwh']:0.1f} kWh dirty charge blocked")
    print("=" * 78)

    print("\nGridSense AI simulation completed successfully.")
    print("Launch the dashboard via: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
