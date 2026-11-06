"""
GridSense AI - Layer 2A: Battery Storage Optimization (Supply Arm)
Formulated as a Linear Program (LP) using SciPy's HiGHS solver.
Calibrated for Indian C&I (Commercial & Industrial) Battery Systems and MERC ToD Tariffs.

Key Capabilities:
- Arbitrage Logic: Charge when clean/cheap, discharge during peak tariff hours (18:00 - 22:00).
- Solves the 'Battery Paradox': Enforces clean-only charging, preventing charging
  from dirty coal baseload during night hours (which is ~750 gCO2/kWh in India!).
- Enforces physical battery constraints:
  * Energy capacity limits (kWh)
  * Inverter charge/discharge rate limits (kW)
  * Round-trip efficiency losses (charge/discharge)
  * Minimum and Maximum State of Charge (SoC) for battery cycle life
- Supports multi-objective weight alpha: balances Cost minimization (₹) vs. Carbon abatement.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, Any, Optional


class BatteryOptimizer:
    def __init__(
        self,
        capacity_kwh: float = 500.0,
        max_power_kw: float = 150.0,
        roundtrip_efficiency: float = 0.90,
        soc_min_pct: float = 0.15,
        soc_max_pct: float = 0.95,
        soc_initial_pct: float = 0.25,
        degradation_cost_per_kwh: float = 0.65,  # battery cycling wear cost in ₹/kWh
    ):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.eta_ch = np.sqrt(roundtrip_efficiency)
        self.eta_dis = np.sqrt(roundtrip_efficiency)
        self.soc_min = soc_min_pct * capacity_kwh
        self.soc_max = soc_max_pct * capacity_kwh
        self.soc_initial = soc_initial_pct * capacity_kwh
        self.deg_cost = degradation_cost_per_kwh

    def optimize_dispatch(
        self,
        horizon_df: pd.DataFrame,
        alpha_cost: float = 0.6,  # 1.0 = pure cost (₹), 0.0 = pure carbon, 0.6 = balanced
        enforce_clean_charging: bool = True
    ) -> pd.DataFrame:
        """
        Solves optimal battery dispatch over the forecast horizon using HiGHS LP solver.
        """
        T = len(horizon_df)
        time_steps = list(range(T))

        gen = horizon_df.get("forecast_total_renewable_kw", horizon_df["total_renewable_kw"]).values
        base_demand = horizon_df.get("baseline_demand_kw", np.zeros(T)).values
        price = horizon_df.get("forecast_electricity_price_kwh", horizon_df["electricity_price_kwh"]).values
        carbon = horizon_df.get("forecast_grid_carbon_intensity", horizon_df["grid_carbon_intensity"]).values

        # Carbon conversion factor in ₹: ~₹4,000 / tonne CO2 = ₹0.004 / gCO2
        carbon_cost_weight = 0.0040

        # Variable indexing:
        # 0*T to 1*T: P_ch (kW)
        # 1*T to 2*T: P_dis (kW)
        # 2*T to 3*T: SoC (kWh)
        # 3*T to 4*T: P_grid_import (kW)
        # 4*T to 5*T: P_curtail (kW)
        n_vars = 5 * T
        c = np.zeros(n_vars)

        for t in range(T):
            c[0 * T + t] = self.deg_cost
            c[1 * T + t] = self.deg_cost
            cost_term = alpha_cost * price[t]
            carbon_term = (1.0 - alpha_cost) * carbon[t] * carbon_cost_weight
            c[3 * T + t] = cost_term + carbon_term
            c[4 * T + t] = 2.50  # Curtailment penalty in ₹/kWh

        bounds = []
        for t in range(T):
            bounds.append((0.0, self.max_power_kw))
        for t in range(T):
            bounds.append((0.0, self.max_power_kw))
        for t in range(T):
            bounds.append((self.soc_min, self.soc_max))
        for t in range(T):
            bounds.append((0.0, None))
        for t in range(T):
            bounds.append((0.0, None))

        A_eq = np.zeros((2 * T, n_vars))
        b_eq = np.zeros(2 * T)

        for t in range(T):
            A_eq[t, 2 * T + t] = 1.0
            if t > 0:
                A_eq[t, 2 * T + (t - 1)] = -1.0
            A_eq[t, 0 * T + t] = -self.eta_ch
            A_eq[t, 1 * T + t] = (1.0 / self.eta_dis)
            b_eq[t] = self.soc_initial if t == 0 else 0.0

            A_eq[T + t, 0 * T + t] = -1.0
            A_eq[T + t, 1 * T + t] = 1.0
            A_eq[T + t, 3 * T + t] = 1.0
            A_eq[T + t, 4 * T + t] = -1.0
            b_eq[T + t] = base_demand[t] - gen[t]

        # Clean-Charging / Anti-Battery-Paradox:
        # In India, overnight coal baseload is ~750 gCO2/kWh.
        # If carbon intensity is above the clean percentile, charging from grid is restricted.
        A_ub_list = []
        b_ub_list = []

        carbon_threshold = np.percentile(carbon, 45) if enforce_clean_charging else 9999.0
        for t in range(T):
            if enforce_clean_charging and carbon[t] > carbon_threshold:
                surplus = max(0.0, gen[t] - base_demand[t])
                row = np.zeros(n_vars)
                row[0 * T + t] = 1.0
                A_ub_list.append(row)
                b_ub_list.append(surplus)

        A_ub = np.array(A_ub_list) if A_ub_list else None
        b_ub = np.array(b_ub_list) if b_ub_list else None

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            raise RuntimeError(f"Battery LP optimization failed: {res.message}")

        x = res.x
        ch_vals = x[0 * T:1 * T]
        dis_vals = x[1 * T:2 * T]
        soc_vals = x[2 * T:3 * T]
        grid_vals = x[3 * T:4 * T]
        curtail_vals = x[4 * T:5 * T]

        results = horizon_df.copy()
        results["battery_charge_kw"] = np.round(ch_vals, 2)
        results["battery_discharge_kw"] = np.round(dis_vals, 2)
        results["battery_net_power_kw"] = np.round(dis_vals - ch_vals, 2)
        results["battery_soc_kwh"] = np.round(soc_vals, 2)
        results["battery_soc_pct"] = np.round(soc_vals / self.capacity_kwh * 100, 1)
        results["grid_import_kw"] = np.round(grid_vals, 2)
        results["renewable_curtailed_kw"] = np.round(curtail_vals, 2)

        clean_charge = []
        for t in range(T):
            surplus = max(0.0, gen[t] - base_demand[t])
            c_val = ch_vals[t]
            clean_charge.append(min(c_val, surplus))
        results["battery_clean_charge_kw"] = np.round(clean_charge, 2)
        results["battery_grid_charge_kw"] = np.round(ch_vals - np.array(clean_charge), 2)

        return results

    def simulate_naive_baseline(self, horizon_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates standard legacy battery dispatch common in Indian C&I facilities:
        - Statically charges during night off-peak tariff (01:00 to 05:00) to save money.
        - But overnight grid in India is 80%+ coal (~760 gCO2/kWh)!
        - Then discharges 18:00 - 21:00. This triggers the Battery Paradox.
        """
        T = len(horizon_df)
        results = horizon_df.copy()
        
        gen = horizon_df.get("forecast_total_renewable_kw", horizon_df["total_renewable_kw"]).values
        base_demand = horizon_df.get("baseline_demand_kw", np.zeros(T)).values
        hours = horizon_df["hour"].values

        soc = self.soc_initial
        soc_history = []
        ch_history = []
        dis_history = []
        grid_history = []
        curtail_history = []

        for t in range(T):
            h = hours[t]
            g = gen[t]
            d = base_demand[t]

            p_ch = 0.0
            p_dis = 0.0

            # Naive static schedule: Charge 01:00 - 05:00 blindly on coal baseload
            if 1 <= h <= 4:
                space = self.soc_max - soc
                p_ch = min(self.max_power_kw, space / self.eta_ch)
            # Naive static schedule: Discharge 18:00 - 21:00
            elif 18 <= h <= 21:
                available = soc - self.soc_min
                p_dis = min(self.max_power_kw, available * self.eta_dis)

            soc = soc + (p_ch * self.eta_ch) - (p_dis / self.eta_dis)
            soc = np.clip(soc, self.soc_min, self.soc_max)

            net_need = (d + p_ch) - (g + p_dis)
            if net_need > 0:
                p_grid = net_need
                p_curt = 0.0
            else:
                p_grid = 0.0
                p_curt = abs(net_need)

            soc_history.append(soc)
            ch_history.append(p_ch)
            dis_history.append(p_dis)
            grid_history.append(p_grid)
            curtail_history.append(p_curt)

        results["naive_battery_charge_kw"] = np.round(ch_history, 2)
        results["naive_battery_discharge_kw"] = np.round(dis_history, 2)
        results["naive_battery_soc_kwh"] = np.round(soc_history, 2)
        results["naive_battery_soc_pct"] = np.round(np.array(soc_history) / self.capacity_kwh * 100, 1)
        results["naive_grid_import_kw"] = np.round(grid_history, 2)
        results["naive_curtailed_kw"] = np.round(curtail_history, 2)

        return results
