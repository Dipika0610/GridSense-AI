"""
GridSense AI - Impact & Value Proposition Engine
Calculates side-by-side performance indicators between GridSense AI and Naive Baseline:
1. Carbon Emissions Abatement (kg CO2 & % reduction)
2. Energy Bill Cost Savings ($ or ₹ & % reduction)
3. Renewable Curtailment Reduction (kWh avoided wastage)
4. Peak Grid Demand Relief (kW peak reduction)
5. Battery Paradox Prevention (Dirty grid charging avoided)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class ImpactEngine:
    """
    Computes rigorous metrics comparing GridSense AI's unified approach
    against the traditional, unmanaged baseline.
    """

    def evaluate_system(
        self,
        dispatch_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        naive_dispatch_df: pd.DataFrame,
        naive_schedule_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Integrates supply-side dispatch and demand-side scheduling to produce
        comprehensive system-level impact analytics.
        """
        T = len(dispatch_df)
        hours = list(range(T))

        price = dispatch_df["electricity_price_kwh"].values
        carbon = dispatch_df["grid_carbon_intensity"].values  # gCO2/kWh
        gen = dispatch_df.get("forecast_total_renewable_kw", dispatch_df["total_renewable_kw"]).values
        base_demand = dispatch_df.get("baseline_demand_kw", np.zeros(T)).values

        # ----------------- 1. GridSense AI Unified System -----------------
        # Total load = Baseline Demand + Battery Charge + Scheduled Flexible Loads
        # Total local supply = Renewable Generation + Battery Discharge
        gridsense_flex = schedule_df["total_flexible_demand_kw"].values
        gridsense_ch = dispatch_df["battery_charge_kw"].values
        gridsense_dis = dispatch_df["battery_discharge_kw"].values

        gridsense_net_demand = (base_demand + gridsense_flex + gridsense_ch) - (gen + gridsense_dis)
        gridsense_grid_draw = np.maximum(0.0, gridsense_net_demand)
        gridsense_curtailment = np.maximum(0.0, -gridsense_net_demand)

        # Financial cost
        gridsense_total_cost = np.sum(gridsense_grid_draw * price)
        # Carbon footprint (convert g to kg: / 1000)
        gridsense_total_carbon_kg = np.sum((gridsense_grid_draw * carbon) / 1000.0)
        # Total curtailment
        gridsense_total_curtail_kwh = np.sum(gridsense_curtailment)
        # Peak grid demand
        gridsense_peak_grid_kw = np.max(gridsense_grid_draw)
        # Clean charging share
        clean_ch_kwh = np.sum(dispatch_df.get("battery_clean_charge_kw", np.zeros(T)).values)
        total_ch_kwh = np.sum(gridsense_ch)
        clean_charge_pct = (clean_ch_kwh / total_ch_kwh * 100.0) if total_ch_kwh > 0 else 100.0

        # ----------------- 2. Naive Baseline System -----------------
        naive_flex = naive_schedule_df["naive_flexible_demand_kw"].values
        naive_ch = naive_dispatch_df["naive_battery_charge_kw"].values
        naive_dis = naive_dispatch_df["naive_battery_discharge_kw"].values

        naive_net_demand = (base_demand + naive_flex + naive_ch) - (gen + naive_dis)
        naive_grid_draw = np.maximum(0.0, naive_net_demand)
        naive_curtailment = np.maximum(0.0, -naive_net_demand)

        naive_total_cost = np.sum(naive_grid_draw * price)
        naive_total_carbon_kg = np.sum((naive_grid_draw * carbon) / 1000.0)
        naive_total_curtail_kwh = np.sum(naive_curtailment)
        naive_peak_grid_kw = np.max(naive_grid_draw)

        # ----------------- 3. Comparative Delta & Improvements -----------------
        cost_saved = naive_total_cost - gridsense_total_cost
        cost_saved_pct = (cost_saved / naive_total_cost * 100.0) if naive_total_cost > 0 else 0.0

        carbon_saved_kg = naive_total_carbon_kg - gridsense_total_carbon_kg
        carbon_saved_pct = (carbon_saved_kg / naive_total_carbon_kg * 100.0) if naive_total_carbon_kg > 0 else 0.0

        curtail_avoided_kwh = naive_total_curtail_kwh - gridsense_total_curtail_kwh
        curtail_avoided_pct = (
            (curtail_avoided_kwh / naive_total_curtail_kwh * 100.0)
            if naive_total_curtail_kwh > 0 else 0.0
        )

        peak_reduced_kw = naive_peak_grid_kw - gridsense_peak_grid_kw
        peak_reduced_pct = (peak_reduced_kw / naive_peak_grid_kw * 100.0) if naive_peak_grid_kw > 0 else 0.0

        # Battery Paradox metric:
        # Detect how many kWh naive battery charged when carbon was above the median
        carbon_median = np.median(carbon)
        dirty_charging_hours = np.where(carbon > carbon_median)[0]
        naive_dirty_charge_kwh = np.sum(naive_ch[dirty_charging_hours])
        gridsense_dirty_charge_kwh = np.sum(
            dispatch_df.get("battery_grid_charge_kw", np.zeros(T)).values[dirty_charging_hours]
        )
        battery_paradox_avoided_kwh = max(0.0, naive_dirty_charge_kwh - gridsense_dirty_charge_kwh)

        return {
            "gridsense": {
                "total_cost": round(gridsense_total_cost, 2),
                "total_carbon_kg": round(gridsense_total_carbon_kg, 2),
                "curtailment_kwh": round(gridsense_total_curtail_kwh, 2),
                "peak_grid_kw": round(gridsense_peak_grid_kw, 2),
                "clean_charge_pct": round(clean_charge_pct, 1),
                "grid_draw_series": np.round(gridsense_grid_draw, 2).tolist(),
                "curtailment_series": np.round(gridsense_curtailment, 2).tolist(),
            },
            "naive": {
                "total_cost": round(naive_total_cost, 2),
                "total_carbon_kg": round(naive_total_carbon_kg, 2),
                "curtailment_kwh": round(naive_total_curtail_kwh, 2),
                "peak_grid_kw": round(naive_peak_grid_kw, 2),
                "grid_draw_series": np.round(naive_grid_draw, 2).tolist(),
                "curtailment_series": np.round(naive_curtailment, 2).tolist(),
            },
            "savings": {
                "cost_saved": round(cost_saved, 2),
                "cost_saved_pct": round(cost_saved_pct, 1),
                "carbon_saved_kg": round(carbon_saved_kg, 2),
                "carbon_saved_pct": round(carbon_saved_pct, 1),
                "curtailment_avoided_kwh": round(curtail_avoided_kwh, 2),
                "curtailment_avoided_pct": round(curtail_avoided_pct, 1),
                "peak_reduction_kw": round(peak_reduced_kw, 2),
                "peak_reduction_pct": round(peak_reduced_pct, 1),
                "battery_paradox_prevented_kwh": round(battery_paradox_avoided_kwh, 2),
            }
        }

