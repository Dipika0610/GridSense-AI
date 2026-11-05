"""
GridSense AI - Weather and Grid API Client
Provides connectors for:
- Open-Meteo Weather API (solar radiation, temperature, wind speed, cloud cover)
- Offline synthetic fallback when offline or sandboxed
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
    Fetches real-time and forecasted weather data from Open-Meteo,
    with an automatic fallback to the high-fidelity simulator.
    """
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float = 19.0760, longitude: float = 72.8777):  # Default Mumbai / Maharashtra
        self.latitude = latitude
        self.longitude = longitude
        self.generator = EnergyDataGenerator()

    def fetch_weather_forecast(self, hours: int = 24) -> pd.DataFrame:
        """
        Attempts to fetch live solar and wind weather from Open-Meteo.
        Falls back to the simulator if network is unavailable or request fails.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,cloud_cover,direct_normal_irradiance,shortwave_radiation,wind_speed_10m",
            "forecast_days": 2,
            "timezone": "auto"
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

                # Derive generation from live weather
                solar_capacity_kw = 500.0
                temp_loss = 1.0 - np.maximum(0, (df["temperature_c"] - 25.0) * 0.004)
                df["solar_generation_kw"] = np.round(
                    (df["solar_irradiance_w_m2"] / 1000.0) * solar_capacity_kw * temp_loss, 2
                )

                wind_capacity_kw = 300.0
                ws = df["wind_speed_m_s"]
                wind_power_factor = np.where(
                    ws < 3.0, 0.0,
                    np.where(ws < 12.0, ((ws - 3.0) / 9.0) ** 3,
                             np.where(ws <= 25.0, 1.0, 0.0))
                )
                df["wind_generation_kw"] = np.maximum(0, np.round(wind_power_factor * wind_capacity_kw, 2))
                df["total_renewable_kw"] = np.round(df["solar_generation_kw"] + df["wind_generation_kw"], 2)

                # Synthetic grid price and carbon intensity matching time-of-use
                base_price = 0.12
                tou_mult = [
                    2.3 if 17 <= h <= 21 else (1.4 if 7 <= h <= 9 else (0.55 if 11 <= h <= 15 else 0.5 if h <= 5 else 1.0))
                    for h in df["hour"]
                ]
                df["electricity_price_kwh"] = np.round(base_price * np.array(tou_mult), 4)

                base_carbon = 320.0
                carbon_diurnal = (
                    140.0 * np.sin(2 * np.pi * (df["hour"] - 15) / 24)
                    - 90.0 * np.maximum(0, np.sin(np.pi * (df["hour"] - 7) / 10))
                )
                df["grid_carbon_intensity"] = np.maximum(90.0, np.round(base_carbon + carbon_diurnal, 1))

                facility_base = 180.0 + 80.0 * np.sin(np.pi * np.maximum(0, df["hour"] - 6) / 14)
                df["baseline_demand_kw"] = np.maximum(50.0, np.round(facility_base, 1))

                logger.info("Successfully fetched live weather from Open-Meteo API.")
                return df
        except Exception as e:
            logger.warning(f"Could not connect to Open-Meteo API ({e}). Falling back to simulation engine.")
        
        # Fallback
        return self.generator.generate_forecast_horizon(hours=hours, scenario="normal")

