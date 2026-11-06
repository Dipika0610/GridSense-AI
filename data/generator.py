"""
GridSense AI - Indian Power Grid Data Generator & Synthesizer
Calibrated to Indian power sector regulations and empirical benchmarks:
- Central Electricity Authority (CEA) CO2 Baseline Database (v19)
- Maharashtra Electricity Regulatory Commission (MERC) / MSEDCL Time-of-Day (ToD) Tariff Structure
- National Institute of Solar Energy (NISE) solar irradiance patterns for Western India (Pune/Mumbai)
- Indian Commercial & Industrial (C&I) facility loads
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class EnergyDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_historical_dataset(self, days: int = 60, start_date: str = "2026-09-01") -> pd.DataFrame:
        """
        Generates hourly historical records calibrated for Indian grid conditions.
        """
        start = pd.to_datetime(start_date)
        hours = days * 24
        timestamps = [start + timedelta(hours=i) for i in range(hours)]
        
        df = pd.DataFrame({"timestamp": timestamps})
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear

        # 1. Weather features (Maharashtra / Deccan plateau weather)
        # Average ambient temperatures 24C to 34C
        base_temp = 27.0 + 3.0 * np.sin(2 * np.pi * (df["day_of_year"] - 100) / 365)
        diurnal_temp = 6.5 * np.sin(2 * np.pi * (df["hour"] - 8) / 24)
        noise_temp = self.rng.normal(0, 1.2, hours)
        df["temperature_c"] = np.round(base_temp + diurnal_temp + noise_temp, 2)

        # Cloud cover percentage
        raw_clouds = np.zeros(hours)
        val = 0.20
        for t in range(hours):
            val = 0.90 * val + 0.10 * self.rng.uniform(0.0, 0.8) + self.rng.normal(0, 0.04)
            val = np.clip(val, 0.0, 1.0)
            raw_clouds[t] = val
        df["cloud_cover_pct"] = np.round(raw_clouds * 100, 1)

        # Global Horizontal Irradiance (GHI: W/m²)
        solar_elevation = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        solar_elevation = np.where((df["hour"] >= 6) & (df["hour"] <= 18), solar_elevation, 0.0)
        clear_sky_ghi = 920.0 * (solar_elevation ** 1.05)
        cloud_factor = 1.0 - 0.70 * (df["cloud_cover_pct"] / 100.0)
        df["solar_irradiance_w_m2"] = np.maximum(0, clear_sky_ghi * cloud_factor + self.rng.normal(0, 10, hours))
        df["solar_irradiance_w_m2"] = np.round(df["solar_irradiance_w_m2"], 1)

        # Wind speed (m/s)
        base_wind = 4.8 + 1.5 * np.cos(2 * np.pi * (df["hour"] - 3) / 24)
        wind_noise = self.rng.weibull(1.9, hours) * 1.8
        df["wind_speed_m_s"] = np.maximum(0.5, np.round(base_wind + wind_noise, 2))

        # 2. Generation Models
        # Rooftop Solar PV plant: 450 kW rated
        solar_capacity_kw = 450.0
        temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
        df["solar_generation_kw"] = np.round(
            (df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2
        )

        # Captive Wind Turbine: 250 kW rated
        wind_capacity_kw = 250.0
        ws = df["wind_speed_m_s"]
        wind_factor = np.where(
            ws < 3.0, 0.0,
            np.where(ws < 11.5, ((ws - 3.0) / 8.5) ** 3,
                     np.where(ws <= 25.0, 1.0, 0.0))
        )
        df["wind_generation_kw"] = np.round(
            wind_factor * wind_capacity_kw * (1.0 + self.rng.normal(0, 0.03, hours)), 2
        )
        df["wind_generation_kw"] = np.maximum(0, df["wind_generation_kw"])
        df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

        # 3. Indian Electricity Tariff: MERC / MSEDCL Commercial ToD Tariff (₹/kWh)
        # Base HT commercial rate: ₹7.80/kWh
        # 18:00 - 22:00: Peak surcharge (+₹1.80 to +₹2.50) -> ₹9.60 - ₹10.30
        # 06:00 - 09:00: Morning peak (+₹0.80) -> ₹8.60
        # 12:00 - 16:00: Solar hour incentive (-₹0.80) -> ₹7.00
        # 22:00 - 06:00: Night off-peak (-₹1.50) -> ₹6.30
        base_tariff = 7.80
        tariff_series = np.zeros(hours)
        for i, h in enumerate(df["hour"]):
            if 18 <= h <= 22:
                tariff_series[i] = base_tariff + 2.20 + self.rng.uniform(-0.2, 0.3)
            elif 6 <= h <= 9:
                tariff_series[i] = base_tariff + 0.80 + self.rng.uniform(-0.1, 0.1)
            elif 12 <= h <= 16:
                tariff_series[i] = base_tariff - 0.80 + self.rng.uniform(-0.1, 0.1)
            elif 22 <= h or h <= 5:
                tariff_series[i] = base_tariff - 1.50 + self.rng.uniform(-0.1, 0.1)
            else:
                tariff_series[i] = base_tariff + self.rng.uniform(-0.05, 0.05)

        # Occasional grid demand charges / Open Access DSM price spikes
        spikes = (self.rng.random(hours) < 0.02).astype(float) * self.rng.uniform(1.5, 3.5, hours)
        df["electricity_price_kwh"] = np.round(tariff_series + spikes, 2)

        # 4. Indian Grid Marginal Carbon Intensity (gCO₂/kWh)
        # Based on CEA National Grid Baseline (Average ~ 710 gCO2/kWh)
        # Midday solar absorption pushes it down to ~480 - 580 gCO2
        # Evening coal peaker ramp drives it up to ~840 - 920 gCO2
        # Night coal baseload: ~720 - 790 gCO2 (Highlights the Battery Paradox!)
        base_carbon = 710.0
        carbon_diurnal = (
            160.0 * np.sin(2 * np.pi * (df["hour"] - 16) / 24)
            - 140.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        )
        carbon_noise = self.rng.normal(0, 15, hours)
        df["grid_carbon_intensity"] = np.maximum(
            420.0, np.round(base_carbon + carbon_diurnal + carbon_noise, 1)
        )

        # 5. Facility Baseline Inflexible Load (kW)
        # Academic campus / tech park non-shiftable baseline (lighting, ventilation, labs)
        facility_base = 160.0 + 70.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 7) / 13)
        weekend_factor = np.where(df["is_weekend"] == 1, 0.50, 1.0)
        df["baseline_demand_kw"] = np.maximum(
            45.0, np.round((facility_base * weekend_factor) + self.rng.normal(0, 8, hours), 1)
        )

        return df

    def generate_forecast_horizon(self, hours: int = 24, scenario: str = "normal") -> pd.DataFrame:
        """
        Generates upcoming 24-hour operational decision horizon.
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        timestamps = [now + timedelta(hours=i) for i in range(hours)]
        
        df = pd.DataFrame({"timestamp": timestamps})
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear

        if scenario == "high_solar":
            cloud_cover = np.clip(self.rng.normal(12, 4, hours), 0, 20)
            wind_base = 4.5
            price_spike_prob = 0.0
        elif scenario == "cloudy_storm":
            cloud_cover = np.clip(self.rng.normal(80, 10, hours), 55, 95)
            wind_base = 8.5
            price_spike_prob = 0.04
        elif scenario == "extreme_peak_pricing":
            cloud_cover = np.clip(self.rng.normal(25, 8, hours), 5, 50)
            wind_base = 4.0
            price_spike_prob = 0.20
        else:
            cloud_cover = np.clip(self.rng.normal(28, 12, hours), 5, 75)
            wind_base = 5.2
            price_spike_prob = 0.02

        df["cloud_cover_pct"] = np.round(cloud_cover, 1)
        df["temperature_c"] = np.round(27.5 + 5.5 * np.sin(2 * np.pi * (df["hour"] - 8) / 24) + self.rng.normal(0, 0.7, hours), 1)
        
        # Solar
        solar_elevation = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        solar_elevation = np.where((df["hour"] >= 6) & (df["hour"] <= 18), solar_elevation, 0.0)
        clear_sky_ghi = 930.0 * (solar_elevation ** 1.05)
        cloud_factor = 1.0 - 0.70 * (df["cloud_cover_pct"] / 100.0)
        df["solar_irradiance_w_m2"] = np.maximum(0, np.round(clear_sky_ghi * cloud_factor, 1))

        # Wind
        df["wind_speed_m_s"] = np.maximum(0.5, np.round(wind_base + self.rng.weibull(1.9, hours) * 1.6, 2))

        # Capacities
        solar_capacity_kw = 450.0
        temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
        df["solar_generation_kw"] = np.round((df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2)

        wind_capacity_kw = 250.0
        ws = df["wind_speed_m_s"]
        wind_factor = np.where(
            ws < 3.0, 0.0,
            np.where(ws < 11.5, ((ws - 3.0) / 8.5) ** 3,
                     np.where(ws <= 25.0, 1.0, 0.0))
        )
        df["wind_generation_kw"] = np.maximum(0, np.round(wind_factor * wind_capacity_kw, 2))
        df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

        # MSEDCL Tariffs (₹/kWh)
        base_tariff = 7.80
        tariff_series = np.zeros(hours)
        for i, h in enumerate(df["hour"]):
            if 18 <= h <= 22:
                surcharge = 3.20 if scenario == "extreme_peak_pricing" else 2.20
                tariff_series[i] = base_tariff + surcharge
            elif 6 <= h <= 9:
                tariff_series[i] = base_tariff + 0.85
            elif 12 <= h <= 16:
                tariff_series[i] = base_tariff - 0.85
            elif 22 <= h or h <= 5:
                tariff_series[i] = base_tariff - 1.50
            else:
                tariff_series[i] = base_tariff

        spikes = (self.rng.random(hours) < price_spike_prob).astype(float) * self.rng.uniform(2.0, 4.5, hours)
        df["electricity_price_kwh"] = np.round(tariff_series + spikes, 2)

        # CEA Carbon Intensity (gCO2/kWh)
        base_carbon = 705.0
        carbon_diurnal = (
            170.0 * np.sin(2 * np.pi * (df["hour"] - 16) / 24)
            - 155.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        )
        df["grid_carbon_intensity"] = np.maximum(
            410.0, np.round(base_carbon + carbon_diurnal + self.rng.normal(0, 10, hours), 1)
        )

        facility_base = 155.0 + 65.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 7) / 13)
        df["baseline_demand_kw"] = np.maximum(45.0, np.round(facility_base + self.rng.normal(0, 6, hours), 1))

        return df
