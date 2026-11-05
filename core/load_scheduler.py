"""
GridSense AI - Layer 2B: Carbon-Aware Load Scheduling (Demand Arm)
Implements intelligent demand-side management as specified in PPT Slide 2, 3, & 4:
- Shifts flexible loads (EVs, Data Center AI training, HVAC/Cold storage, Water pumps)
  into ultra-clean & low-cost forecasted windows.
- Prevents the 'Blind Plug-in' problem where devices consume high-carbon electricity.
- Formulated as a Linear Program (LP) using SciPy's HiGHS solver.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import List, Dict, Any, Optional


class FlexibleLoadJob:
    def __init__(
        self,
        job_id: str,
        name: str,
        category: str,  # 'EV Charging', 'Data Center', 'Smart Appliance', 'Industrial'
        energy_kwh: float,
        max_power_kw: float,
        earliest_start: int,
        deadline: int,
        preemptible: bool = True
    ):
        self.job_id = job_id
        self.name = name
        self.category = category
        self.energy_kwh = energy_kwh
        self.max_power_kw = max_power_kw
        self.earliest_start = earliest_start
        self.deadline = deadline
        self.preemptible = preemptible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "category": self.category,
            "energy_kwh": self.energy_kwh,
            "max_power_kw": self.max_power_kw,
            "earliest_start": self.earliest_start,
            "deadline": self.deadline,
            "preemptible": self.preemptible
        }


class CarbonAwareLoadScheduler:
    def __init__(self, facility_max_flexible_kw: float = 200.0):
        self.facility_max_flexible_kw = facility_max_flexible_kw
        self.default_jobs = self._create_standard_benchmark_jobs()

    def _create_standard_benchmark_jobs(self) -> List[FlexibleLoadJob]:
        """
        Creates standard representative demand-response workloads.
        """
        return [
            FlexibleLoadJob(
                job_id="job_ev_fleet",
                name="Corporate EV Fleet Charging",
                category="EV Charging",
                energy_kwh=160.0,
                max_power_kw=50.0,
                earliest_start=8,
                deadline=18,
                preemptible=True
            ),
            FlexibleLoadJob(
                job_id="job_ai_training",
                name="Data Center AI Model Training",
                category="Data Center",
                energy_kwh=220.0,
                max_power_kw=60.0,
                earliest_start=6,
                deadline=22,
                preemptible=True
            ),
            FlexibleLoadJob(
                job_id="job_hvac_thermal",
                name="Thermal Cold-Storage & HVAC Pre-Cool",
                category="Smart Appliance",
                energy_kwh=90.0,
                max_power_kw=30.0,
                earliest_start=9,
                deadline=17,
                preemptible=True
            ),
            FlexibleLoadJob(
                job_id="job_water_pumps",
                name="Municipal Water Treatment & Pumping",
                category="Industrial",
                energy_kwh=75.0,
                max_power_kw=25.0,
                earliest_start=7,
                deadline=20,
                preemptible=True
            ),
        ]

    def schedule_loads(
        self,
        horizon_df: pd.DataFrame,
        jobs: Optional[List[FlexibleLoadJob]] = None,
        alpha_cost: float = 0.5  # Weight between Cost and Carbon
    ) -> pd.DataFrame:
        """
        Solves optimal schedule placing loads in cleanest/cheapest time slots using HiGHS.
        """
        if jobs is None:
            jobs = self.default_jobs

        T = len(horizon_df)
        J = len(jobs)
        n_vars = J * T

        price = horizon_df.get("forecast_electricity_price_kwh", horizon_df["electricity_price_kwh"]).values
        carbon = horizon_df.get("forecast_grid_carbon_intensity", horizon_df["grid_carbon_intensity"]).values
        carbon_cost_weight = 0.00012

        # Objective vector c:
        c = np.zeros(n_vars)
        for j_idx, job in enumerate(jobs):
            for t in range(T):
                idx = j_idx * T + t
                cost_term = alpha_cost * price[t]
                carbon_term = (1.0 - alpha_cost) * carbon[t] * carbon_cost_weight
                c[idx] = cost_term + carbon_term

        # Variable Bounds:
        bounds = []
        for j_idx, job in enumerate(jobs):
            for t in range(T):
                if job.earliest_start <= t <= job.deadline:
                    bounds.append((0.0, job.max_power_kw))
                else:
                    bounds.append((0.0, 0.0))

        # Equality Constraints: each job must receive exactly job.energy_kwh
        A_eq = np.zeros((J, n_vars))
        b_eq = np.zeros(J)
        for j_idx, job in enumerate(jobs):
            for t in range(T):
                A_eq[j_idx, j_idx * T + t] = 1.0
            b_eq[j_idx] = job.energy_kwh

        # Inequality Constraints: facility max power limit per hour
        A_ub = np.zeros((T, n_vars))
        b_ub = np.full(T, self.facility_max_flexible_kw)
        for t in range(T):
            for j_idx in range(J):
                A_ub[t, j_idx * T + t] = 1.0

        # Solve
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
            raise RuntimeError(f"Load scheduling optimization failed: {res.message}")

        x = res.x
        results = horizon_df.copy()
        total_scheduled_kw = np.zeros(T)

        for j_idx, job in enumerate(jobs):
            job_power = np.round(x[j_idx * T:(j_idx + 1) * T], 2)
            results[f"scheduled_{job.job_id}_kw"] = job_power
            total_scheduled_kw += job_power

        results["total_flexible_demand_kw"] = np.round(total_scheduled_kw, 2)
        return results

    def simulate_naive_schedule(
        self,
        horizon_df: pd.DataFrame,
        jobs: Optional[List[FlexibleLoadJob]] = None
    ) -> pd.DataFrame:
        """
        Simulates the naive 'plug-and-run' baseline:
        Loads start immediately upon earliest availability at maximum power until completed,
        completely unaware of grid carbon peaks or high price periods.
        """
        if jobs is None:
            jobs = self.default_jobs

        T = len(horizon_df)
        results = horizon_df.copy()
        total_naive_kw = np.zeros(T)

        for job in jobs:
            rem_energy = job.energy_kwh
            naive_power = np.zeros(T)

            for t in range(T):
                if t >= job.earliest_start and rem_energy > 0:
                    power = min(job.max_power_kw, rem_energy)
                    naive_power[t] = power
                    rem_energy -= power

            results[f"naive_{job.job_id}_kw"] = np.round(naive_power, 2)
            total_naive_kw += naive_power

        results["naive_flexible_demand_kw"] = np.round(total_naive_kw, 2)
        return results
