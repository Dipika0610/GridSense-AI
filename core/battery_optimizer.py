"""
GridSense AI - Layer 2A: Battery Storage Optimization (Supply Arm)
Formulated as a high-performance Linear Program (LP) using SciPy's HiGHS solver.

Key Capabilities:
- Arbitrage Logic: Charge when clean/cheap, discharge when costly/peaking.
- Solves the 'Battery Paradox': Enforces clean-only charging, preventing charging
  from dirty fossil peakers during low-price night hours.
- Enforces physical battery constraints:
  * Energy capacity limits (kWh)
  * Inverter charge/discharge rate limits (kW)
  * Round-trip efficiency losses (charge/discharge)
  * Minimum and Maximum State of Charge (SoC) for battery cycle life
- Supports multi-objective weight alpha: balances Cost minimization vs. Carbon abatement.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, Any, Optional


class BatteryOptimizer:
    def __init__(
        self,
        capacity_kwh: float = 600.0,
        max_power_kw: float = 200.0,
        roundtrip_efficiency: float = 0.90,
        soc_min_pct: float = 0.15,
        soc_max_pct: float = 0.95,
        soc_initial_pct: float = 0.25,
        degradation_cost_per_kwh: float = 0.008,  # battery cycling wear cost ($/kWh)
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
        alpha_cost: float = 0.6,  # 1.0 = pure cost, 0.0 = pure carbon, 0.6 = balanced
        enforce_clean_charging: bool = True
    ) -> pd.DataFrame:
        """
        Solves the optimal battery dispatch over the forecast horizon using HiGHS LP solver.
        """
        T = len(horizon_df)
        time_steps = list(range(T))

        # Extract forecasted signals
        gen = horizon_df.get("forecast_total_renewable_kw", horizon_df["total_renewable_kw"]).values
        base_demand = horizon_df.get("baseline_demand_kw", np.zeros(T)).values
        price = horizon_df.get("forecast_electricity_price_kwh", horizon_df["electricity_price_kwh"]).values
        carbon = horizon_df.get("forecast_grid_carbon_intensity", horizon_df["grid_carbon_intensity"]).values

        # Carbon conversion factor ($ per gCO2 based on $100/tonne Social Cost of Carbon = $0.0001/g)
        carbon_cost_weight = 0.00012

        # Variable indexing:
        # 0*T to 1*T: P_ch (kW)
        # 1*T to 2*T: P_dis (kW)
        # 2*T to 3*T: SoC (kWh)
        # 3*T to 4*T: P_grid_import (kW)
        # 4*T to 5*T: P_curtail (kW)
        n_vars = 5 * T
        c = np.zeros(n_vars)

        # Objective coefficients:
        for t in range(T):
            # Degradation cost on charge/discharge
            c[0 * T + t] = self.deg_cost
            c[1 * T + t] = self.deg_cost
            # Grid import cost + carbon footprint
            cost_term = alpha_cost * price[t]
            carbon_term = (1.0 - alpha_cost) * carbon[t] * carbon_cost_weight
            c[3 * T + t] = cost_term + carbon_term
            # Curtailment penalty
            c[4 * T + t] = 0.05

        # Variable Bounds
        bounds = []
        for t in range(T):
            bounds.append((0.0, self.max_power_kw))  # P_ch
        for t in range(T):
            bounds.append((0.0, self.max_power_kw))  # P_dis
        for t in range(T):
            bounds.append((self.soc_min, self.soc_max))  # SoC
        for t in range(T):
            bounds.append((0.0, None))  # P_grid_import
        for t in range(T):
            bounds.append((0.0, None))  # P_curtail

        # Equality Constraints: A_eq @ x == b_eq
        # 1. SoC recursion:
        #    For t=0: SoC_0 - eta_ch*P_ch_0 + (1/eta_dis)*P_dis_0 = soc_initial
        #    For t>0: SoC_t - SoC_{t-1} - eta_ch*P_ch_t + (1/eta_dis)*P_dis_t = 0
        # 2. Site power balance:
        #    -P_ch_t + P_dis_t + P_grid_t - P_curt_t = base_demand_t - gen_t
        A_eq = np.zeros((2 * T, n_vars))
        b_eq = np.zeros(2 * T)

        for t in range(T):
            # SoC balance (row t)
            A_eq[t, 2 * T + t] = 1.0  # SoC_t
            if t > 0:
                A_eq[t, 2 * T + (t - 1)] = -1.0  # -SoC_{t-1}
            A_eq[t, 0 * T + t] = -self.eta_ch  # -eta_ch * P_ch_t
            A_eq[t, 1 * T + t] = (1.0 / self.eta_dis)  # +(1/eta_dis) * P_dis_t
            b_eq[t] = self.soc_initial if t == 0 else 0.0

            # Power balance (row T + t)
            A_eq[T + t, 0 * T + t] = -1.0  # -P_ch_t
            A_eq[T + t, 1 * T + t] = 1.0   # +P_dis_t
            A_eq[T + t, 3 * T + t] = 1.0   # +P_grid_t
            A_eq[T + t, 4 * T + t] = -1.0  # -P_curt_t
            b_eq[T + t] = base_demand[t] - gen[t]

        # Inequality Constraints: A_ub @ x <= b_ub
        # Clean-Charging / Anti-Battery-Paradox:
        # If carbon intensity is above the clean threshold, limit charging strictly to local surplus
        A_ub_list = []
        b_ub_list = []

        carbon_threshold = np.percentile(carbon, 45) if enforce_clean_charging else 9999.0
        for t in range(T):
            if enforce_clean_charging and carbon[t] > carbon_threshold:
                surplus = max(0.0, gen[t] - base_demand[t])
                row = np.zeros(n_vars)
                row[0 * T + t] = 1.0  # P_ch_t <= surplus
                A_ub_list.append(row)
                b_ub_list.append(surplus)

        if A_ub_list:
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            A_ub = None
            b_ub = None

        # Solve via HiGHS
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

        # Distinguish clean renewable charging vs. clean grid charging
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
        Simulates standard legacy battery dispatch:
        - Statically charges during fixed night hours (01:00 to 05:00) blindly.
        - Discharges during fixed evening hours (18:00 to 21:00).
        - Ignores carbon intensity and solar generation timing, triggering the Battery Paradox.
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

            # Static naive schedule: Charge blindly 01:00 - 05:00
            if 1 <= h <= 4:
                space = self.soc_max - soc
                p_ch = min(self.max_power_kw, space / self.eta_ch)
            # Static naive schedule: Discharge 18:00 - 21:00
            elif 18 <= h <= 21:
                available = soc - self.soc_min
                p_dis = min(self.max_power_kw, available * self.eta_dis)

            soc = soc + (p_ch * self.eta_ch) - (p_dis / self.eta_dis)
            soc = np.clip(soc, self.soc_min, self.soc_max)

            # Grid import and curtailment under naive dispatch
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
