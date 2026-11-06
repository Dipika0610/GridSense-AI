"""
GridSense AI - Weather and Grid API Client (India / Pune & Maharashtra Region)
Provides connectors for:
- Open-Meteo Weather API (solar radiation, temperature, wind speed, cloud cover)
- Calibrated to Western India / Maharashtra coordinates (Default: PCCOE / Pune: 18.6279° N, 73.8131° E)
- MSEDCL ToD tariffs (₹/kWh) and CEA Carbon baseline (gCO₂/kWh)
"""

import logging
from typing import Optional
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.generator import EnergyDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherGridClient:
    """
    Fetches real-time and forecasted weather data from Open-Meteo for Indian coordinates,
    with an automatic fallback to the high-fidelity simulator.
    """
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float = 18.6279, longitude: float = 73.8131):  # PCCOE / Pune coordinates
        self.latitude = latitude
        self.longitude = longitude
        self.generator = EnergyDataGenerator()

    def fetch_weather_forecast(self, hours: int = 24) -> pd.DataFrame:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,cloud_cover,direct_normal_irradiance,shortwave_radiation,wind_speed_10m",
            "forecast_days": 2,
            "timezone": "Asia/Kolkata"
        }
        try:
            response = requests.get(self.OPEN_METEO_URL, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                times = pd.to_datetime(hourly.get("time", []))[:hours]
                
                df = pd.DataFrame({
                    "timestamp": times,
                    "temperature_c": hourly.get("temperature_2m", [])[:hours],
                    "cloud_cover_pct": hourly.get("cloud_cover", [])[:hours],
                    "solar_irradiance_w_m2": hourly.get("shortwave_radiation", [])[:hours],
                    "wind_speed_m_s": hourly.get("wind_speed_10m", [])[:hours]
                })
                
                df["hour"] = df["timestamp"].dt.hour
                df["day_of_week"] = df["timestamp"].dt.dayofweek
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

                # Solar & Wind generation
                solar_capacity_kw = 450.0
                temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
                df["solar_generation_kw"] = np.round(
                    (df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2
                )

                wind_capacity_kw = 250.0
                ws = df["wind_speed_m_s"]
                wind_factor = np.where(
                    ws < 3.0, 0.0,
                    np.where(ws < 11.5, ((ws - 3.0) / 8.5) ** 3,
                             np.where(ws <= 25.0, 1.0, 0.0))
                )
                df["wind_generation_kw"] = np.maximum(0, np.round(wind_factor * wind_capacity_kw, 2))
                df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

                # Indian MSEDCL ToD tariffs (₹/kWh)
                base_tariff = 7.80
                tariff_series = []
                for h in df["hour"]:
                    if 18 <= h <= 22:
                        tariff_series.append(base_tariff + 2.20)
                    elif 6 <= h <= 9:
                        tariff_series.append(base_tariff + 0.85)
                    elif 12 <= h <= 16:
                        tariff_series.append(base_tariff - 0.85)
                    elif 22 <= h or h <= 5:
                        tariff_series.append(base_tariff - 1.50)
                    else:
                        tariff_series.append(base_tariff)
                df["electricity_price_kwh"] = np.round(tariff_series, 2)

                # CEA Indian National Grid Carbon Intensity
                base_carbon = 705.0
                carbon_diurnal = (
                    170.0 * np.sin(2 * np.pi * (df["hour"] - 16) / 24)
                    - 155.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
                )
                df["grid_carbon_intensity"] = np.maximum(410.0, np.round(base_carbon + carbon_diurnal, 1))

                facility_base = 155.0 + 65.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 7) / 13)
                df["baseline_demand_kw"] = np.maximum(45.0, np.round(facility_base, 1))

                logger.info("Successfully fetched live weather from Open-Meteo API for Pune/PCCOE.")
                return df
        except Exception as e:
            logger.warning(f"Could not connect to Open-Meteo API ({e}). Falling back to simulation engine.")
        
        return self.generator.generate_forecast_horizon(hours=hours, scenario="normal")
