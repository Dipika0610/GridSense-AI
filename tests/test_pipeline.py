"""
GridSense AI - Automated Pipeline Verification Tests
Verifies:
1. Data generation integrity
2. Layer 1 Forecasting training & prediction
3. Layer 2A Battery optimization physical constraint satisfaction
4. Layer 2B Load scheduler job completion and deadline compliance
5. System-level impact calculation
"""

import pytest
import numpy as np
import pandas as pd

from data.generator import EnergyDataGenerator
from core.forecasting import ForecastingCore
from core.battery_optimizer import BatteryOptimizer
from core.load_scheduler import CarbonAwareLoadScheduler, FlexibleLoadJob
from core.impact_engine import ImpactEngine


def test_data_generator():
    gen = EnergyDataGenerator(seed=123)
    df = gen.generate_historical_dataset(days=10)
    
    assert len(df) == 240
    assert "solar_generation_kw" in df.columns
    assert "wind_generation_kw" in df.columns
    assert "electricity_price_kwh" in df.columns
    assert "grid_carbon_intensity" in df.columns
    assert (df["solar_generation_kw"] >= 0).all()
    assert (df["wind_generation_kw"] >= 0).all()
    assert (df["electricity_price_kwh"] > 0).all()
    assert (df["grid_carbon_intensity"] > 0).all()


def test_forecasting_core():
    gen = EnergyDataGenerator(seed=123)
    hist_df = gen.generate_historical_dataset(days=15)
    horizon_df = gen.generate_forecast_horizon(hours=24)

    forecaster = ForecastingCore(model_type="random_forest")
    metrics = forecaster.train(hist_df)

    assert "total_renewable_kw" in metrics
    assert metrics["total_renewable_kw"]["r2"] > 0.60

    preds = forecaster.predict(horizon_df)
    assert len(preds) == 24
    assert "forecast_total_renewable_kw" in preds.columns
    assert "forecast_electricity_price_kwh" in preds.columns
    assert "forecast_grid_carbon_intensity" in preds.columns
    assert (preds["forecast_total_renewable_kw"] >= 0).all()


def test_battery_optimizer_constraints():
    gen = EnergyDataGenerator(seed=123)
    horizon_df = gen.generate_forecast_horizon(hours=24)

    cap = 500.0
    p_max = 150.0
    battery_opt = BatteryOptimizer(
        capacity_kwh=cap,
        max_power_kw=p_max,
        soc_min_pct=0.10,
        soc_max_pct=0.90,
        soc_initial_pct=0.20
    )

    res = battery_opt.optimize_dispatch(horizon_df, alpha_cost=0.5, enforce_clean_charging=True)

    # Check bounds
    assert (res["battery_charge_kw"] >= -1e-5).all()
    assert (res["battery_charge_kw"] <= p_max + 1e-4).all()
    assert (res["battery_discharge_kw"] >= -1e-5).all()
    assert (res["battery_discharge_kw"] <= p_max + 1e-4).all()
    assert (res["battery_soc_kwh"] >= (cap * 0.10) - 1e-3).all()
    assert (res["battery_soc_kwh"] <= (cap * 0.90) + 1e-3).all()


def test_load_scheduler_completion():
    gen = EnergyDataGenerator(seed=123)
    horizon_df = gen.generate_forecast_horizon(hours=24)

    scheduler = CarbonAwareLoadScheduler()
    test_job = FlexibleLoadJob(
        job_id="test_ev",
        name="Test EV",
        category="EV Charging",
        energy_kwh=50.0,
        max_power_kw=20.0,
        earliest_start=5,
        deadline=15
    )

    res = scheduler.schedule_loads(horizon_df, jobs=[test_job])

    # Check that total energy scheduled matches requirement
    delivered_energy = res["scheduled_test_ev_kw"].sum()
    assert abs(delivered_energy - 50.0) < 1e-2

    # Check that power is 0 outside the window [5, 15]
    assert (res["scheduled_test_ev_kw"].iloc[:5] == 0).all()
    assert (res["scheduled_test_ev_kw"].iloc[16:] == 0).all()


def test_impact_engine_evaluation():
    gen = EnergyDataGenerator(seed=123)
    hist_df = gen.generate_historical_dataset(days=15)
    horizon_df = gen.generate_forecast_horizon(hours=24)

    forecaster = ForecastingCore(model_type="random_forest")
    forecaster.train(hist_df)
    forecast_df = forecaster.predict(horizon_df)

    battery_opt = BatteryOptimizer()
    dispatch_df = battery_opt.optimize_dispatch(forecast_df)
    naive_dispatch_df = battery_opt.simulate_naive_baseline(forecast_df)

    scheduler = CarbonAwareLoadScheduler()
    schedule_df = scheduler.schedule_loads(forecast_df)
    naive_schedule_df = scheduler.simulate_naive_schedule(forecast_df)

    engine = ImpactEngine()
    impact = engine.evaluate_system(dispatch_df, schedule_df, naive_dispatch_df, naive_schedule_df)

    assert "savings" in impact
    assert impact["savings"]["cost_saved"] > 0
    assert impact["savings"]["carbon_saved_kg"] > 0
    assert impact["gridsense"]["clean_charge_pct"] >= 80.0
