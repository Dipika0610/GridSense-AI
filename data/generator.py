"""
GridSense AI - Realistic Energy Data Generator & Synthesizer
Generates synthetic and realistic benchmark data modeling:
- Solar generation (kW) based on solar zenith, temperature, cloud cover
- Wind generation (kW) based on Weibull-distributed wind speeds
- Real-time Electricity Pricing ($/kWh or ₹/kWh) reflecting TOU and peak scarcity
- Dynamic Grid Carbon Intensity (gCO2/kWh) reflecting marginal grid generation mix
- Facility baseline demand (kW)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class EnergyDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_historical_dataset(self, days: int = 90, start_date: str = "2026-08-01") -> pd.DataFrame:
        """
        Generates hourly historical time-series data for training the forecasting core.
        Features include weather variables and target variables:
        - solar_generation_kw
        - wind_generation_kw
        - total_renewable_kw
        - electricity_price
        - grid_carbon_intensity
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

        # 1. Weather features
        # Temperature follows daily diurnal cycle with seasonal drift
        base_temp = 25.0 + 5.0 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)
        diurnal_temp = 7.0 * np.sin(2 * np.pi * (df["hour"] - 8) / 24)
        noise_temp = self.rng.normal(0, 1.5, hours)
        df["temperature_c"] = np.round(base_temp + diurnal_temp + noise_temp, 2)

        # Cloud cover (0% to 100%) with auto-correlated persistence
        raw_clouds = np.zeros(hours)
        val = 0.25
        for t in range(hours):
            val = 0.92 * val + 0.08 * self.rng.uniform(0.0, 1.0) + self.rng.normal(0, 0.05)
            val = np.clip(val, 0.0, 1.0)
            raw_clouds[t] = val
        df["cloud_cover_pct"] = np.round(raw_clouds * 100, 1)

        # Solar irradiance (GHI: W/m^2)
        # Clear sky profile: peaks at solar noon (~hour 12)
        solar_elevation = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        solar_elevation = np.where((df["hour"] >= 6) & (df["hour"] <= 18), solar_elevation, 0.0)
        clear_sky_ghi = 950.0 * (solar_elevation ** 1.1)
        cloud_factor = 1.0 - 0.75 * (df["cloud_cover_pct"] / 100.0)
        df["solar_irradiance_w_m2"] = np.maximum(0, clear_sky_ghi * cloud_factor + self.rng.normal(0, 15, hours))
        df["solar_irradiance_w_m2"] = np.round(df["solar_irradiance_w_m2"], 1)

        # Wind speed (m/s): Weibull-like distribution with higher night/evening breezes
        base_wind = 6.0 + 1.8 * np.cos(2 * np.pi * (df["hour"] - 2) / 24)
        wind_noise = self.rng.weibull(2.0, hours) * 2.0
        df["wind_speed_m_s"] = np.maximum(0.5, np.round(base_wind + wind_noise, 2))

        # 2. Generation targets
        # Solar plant capacity = 500 kW rated
        solar_capacity_kw = 500.0
        panel_efficiency = 0.18
        # Derating from temperature: ~0.4% drop per degree above 25C
        temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
        df["solar_generation_kw"] = np.round(
            (df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2
        )

        # Wind plant capacity = 300 kW rated
        # Power curve: cut-in at 3 m/s, rated at 12 m/s, cut-out at 25 m/s
        wind_capacity_kw = 300.0
        ws = df["wind_speed_m_s"]
        wind_power_factor = np.where(
            ws < 3.0, 0.0,
            np.where(ws < 12.0, ((ws - 3.0) / 9.0) ** 3,
                     np.where(ws <= 25.0, 1.0, 0.0))
        )
        df["wind_generation_kw"] = np.round(
            wind_power_factor * wind_capacity_kw * (1.0 + self.rng.normal(0, 0.04, hours)), 2
        )
        df["wind_generation_kw"] = np.maximum(0, df["wind_generation_kw"])

        df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

        # 3. Grid Signals: Real-time Electricity Price ($/kWh)
        # Base tariff + TOU peak from 17:00 to 21:00 + morning ramp 07:00 to 09:00
        # Renewable duck curve discount during midday
        base_price = 0.12  # $0.12/kWh average
        tou_mult = np.ones(hours)
        for i, h in enumerate(df["hour"]):
            if 17 <= h <= 21:      # Evening peak
                tou_mult[i] = 2.4 + self.rng.uniform(-0.15, 0.25)
            elif 7 <= h <= 9:      # Morning ramp
                tou_mult[i] = 1.5 + self.rng.uniform(-0.1, 0.1)
            elif 11 <= h <= 15:    # Midday solar surge (Duck curve dip)
                tou_mult[i] = 0.6 + self.rng.uniform(-0.08, 0.08)
            elif 0 <= h <= 5:      # Deep off-peak overnight
                tou_mult[i] = 0.5 + self.rng.uniform(-0.05, 0.05)
            else:
                tou_mult[i] = 1.0

        # Scarcity pricing spike (random events during heatwaves/surges)
        spikes = (self.rng.random(hours) < 0.02).astype(float) * self.rng.uniform(0.15, 0.40, hours)
        df["electricity_price_kwh"] = np.round(base_price * tou_mult + spikes, 4)

        # 4. Grid Carbon Intensity (gCO2 / kWh)
        # High when peaker fossil plants turn on (evenings 18:00 - 22:00: ~450 - 550 gCO2)
        # Low when grid solar/wind is strong (midday: ~120 - 200 gCO2)
        base_carbon = 320.0
        carbon_diurnal = (
            140.0 * np.sin(2 * np.pi * (df["hour"] - 15) / 24)  # Peaks around 21:00
            - 90.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 7) / 10))  # Suppressed by midday solar
        )
        carbon_noise = self.rng.normal(0, 18, hours)
        df["grid_carbon_intensity"] = np.maximum(
            90.0, np.round(base_carbon + carbon_diurnal + carbon_noise, 1)
        )

        # 5. Facility Baseline Inflexible Load (kW)
        # Offices / campus / base demand
        facility_base = 180.0 + 80.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 6) / 14)
        weekend_factor = np.where(df["is_weekend"] == 1, 0.55, 1.0)
        df["baseline_demand_kw"] = np.maximum(
            50.0, np.round((facility_base * weekend_factor) + self.rng.normal(0, 10, hours), 1)
        )

        return df

    def generate_forecast_horizon(self, hours: int = 24, scenario: str = "normal") -> pd.DataFrame:
        """
        Generates the operational decision-making horizon (default 24 hours),
        representing the upcoming day for scheduling and battery arbitrage.
        Scenarios: 'normal', 'high_solar', 'cloudy_storm', 'extreme_peak_pricing'
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        timestamps = [now + timedelta(hours=i) for i in range(hours)]
        
        df = pd.DataFrame({"timestamp": timestamps})
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear

        # Weather conditioning based on scenario
        if scenario == "high_solar":
            cloud_cover = np.clip(self.rng.normal(10, 5, hours), 0, 25)
            wind_base = 5.0
            price_spike_prob = 0.0
        elif scenario == "cloudy_storm":
            cloud_cover = np.clip(self.rng.normal(85, 10, hours), 60, 100)
            wind_base = 12.0
            price_spike_prob = 0.05
        elif scenario == "extreme_peak_pricing":
            cloud_cover = np.clip(self.rng.normal(30, 10, hours), 0, 60)
            wind_base = 4.0
            price_spike_prob = 0.15
        else:  # normal
            cloud_cover = np.clip(self.rng.normal(35, 15, hours), 5, 80)
            wind_base = 7.0
            price_spike_prob = 0.02

        df["cloud_cover_pct"] = np.round(cloud_cover, 1)
        df["temperature_c"] = np.round(26.0 + 6.0 * np.sin(2 * np.pi * (df["hour"] - 8) / 24) + self.rng.normal(0, 0.8, hours), 1)
        
        # Irradiance
        solar_elevation = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
        solar_elevation = np.where((df["hour"] >= 6) & (df["hour"] <= 18), solar_elevation, 0.0)
        clear_sky_ghi = 980.0 * (solar_elevation ** 1.1)
        cloud_factor = 1.0 - 0.75 * (df["cloud_cover_pct"] / 100.0)
        df["solar_irradiance_w_m2"] = np.maximum(0, np.round(clear_sky_ghi * cloud_factor, 1))

        # Wind speed
        df["wind_speed_m_s"] = np.maximum(0.5, np.round(wind_base + self.rng.weibull(2.0, hours) * 2.0, 2))

        # Generation
        solar_capacity_kw = 500.0
        temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
        df["solar_generation_kw"] = np.round((df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2)

        wind_capacity_kw = 300.0
        ws = df["wind_speed_m_s"]
        wind_power_factor = np.where(
            ws < 3.0, 0.0,
            np.where(ws < 12.0, ((ws - 3.0) / 9.0) ** 3,
                     np.where(ws <= 25.0, 1.0, 0.0))
        )
        df["wind_generation_kw"] = np.maximum(0, np.round(wind_power_factor * wind_capacity_kw, 2))
        df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

        # Pricing
        base_price = 0.12
        tou_mult = np.ones(hours)
        for i, h in enumerate(df["hour"]):
            if 17 <= h <= 21:
                tou_mult[i] = 2.6 if scenario == "extreme_peak_pricing" else 2.2
            elif 7 <= h <= 9:
                tou_mult[i] = 1.45
            elif 11 <= h <= 15:
                tou_mult[i] = 0.55
            elif 0 <= h <= 5:
                tou_mult[i] = 0.48
            else:
                tou_mult[i] = 1.0
        
        spikes = (self.rng.random(hours) < price_spike_prob).astype(float) * self.rng.uniform(0.25, 0.55, hours)
        df["electricity_price_kwh"] = np.round(base_price * tou_mult + spikes, 4)

        # Carbon Intensity
        base_carbon = 310.0
        carbon_diurnal = (
            150.0 * np.sin(2 * np.pi * (df["hour"] - 15) / 24)
            - 105.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 7) / 10))
        )
        df["grid_carbon_intensity"] = np.maximum(
            85.0, np.round(base_carbon + carbon_diurnal + self.rng.normal(0, 12, hours), 1)
        )

        # Baseline Inflexible Demand
        facility_base = 175.0 + 75.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 6) / 14)
        df["baseline_demand_kw"] = np.maximum(50.0, np.round(facility_base + self.rng.normal(0, 8, hours), 1))

        return df

