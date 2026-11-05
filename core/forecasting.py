"""
GridSense AI - Layer 1: Shared Forecasting Core
Implements the multi-model forecasting pipeline described in PPT Slide 2 & 3:
- Random Forest: Robust non-linear decision tree ensemble
- SVR (Support Vector Regressor): Margin-based baseline
- Gradient Boosting / Neural Regressor (MLP): Sequence and temporal peak capture
Outputs 24-hour predictions for:
1. Solar & Wind Renewable Generation (kW)
2. Real-Time Electricity Price ($/kWh)
3. Grid Carbon Intensity (gCO2/kWh)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class ForecastingCore:
    """
    Unified Forecasting Engine.
    Trains and coordinates models across all critical grid and generation targets.
    """

    FEATURE_COLS = [
        "hour", "day_of_week", "is_weekend", "temperature_c",
        "cloud_cover_pct", "solar_irradiance_w_m2", "wind_speed_m_s"
    ]

    TARGETS = [
        "solar_generation_kw",
        "wind_generation_kw",
        "total_renewable_kw",
        "electricity_price_kwh",
        "grid_carbon_intensity"
    ]

    def __init__(self, model_type: str = "random_forest"):
        """
        model_type: 'random_forest', 'svr', 'neural_net', or 'ensemble'
        """
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.is_trained = False

    def _get_base_regressor(self, target: str):
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
        elif self.model_type == "svr":
            return SVR(C=10.0, epsilon=0.1, kernel="rbf")
        elif self.model_type == "neural_net":
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True)
        elif self.model_type == "ensemble":
            # Gradient boosting captures peak generation and pricing spikes best
            return GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, historical_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Trains models for all targets and logs validation performance.
        Splits chronologically (80% train, 20% test) to reflect real forecasting.
        """
        X = historical_df[self.FEATURE_COLS].copy()
        split_idx = int(len(historical_df) * 0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

        for target in self.TARGETS:
            y = historical_df[target].copy()
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            reg = self._get_base_regressor(target)
            
            # SVR and MLP need scaled features
            if self.model_type in ["svr", "neural_net"]:
                reg.fit(X_train_scaled, y_train)
                y_pred = reg.predict(X_test_scaled)
            else:
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

            # Ensure generation is non-negative
            if "kw" in target:
                y_pred = np.maximum(0, y_pred)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.models[target] = reg
            self.scalers[target] = scaler
            self.metrics[target] = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 3)
            }

        self.is_trained = True
        return self.metrics

    def predict(self, horizon_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates forecast predictions for the operational decision horizon.
        Returns a DataFrame with timestamps, actuals (if present), and forecasted columns.
        """
        if not self.is_trained:
            raise RuntimeError("ForecastingCore must be trained before predicting.")

        out_df = horizon_df.copy()
        X = horizon_df[self.FEATURE_COLS].copy()

        for target in self.TARGETS:
            reg = self.models[target]
            if self.model_type in ["svr", "neural_net"]:
                X_scaled = self.scalers[target].transform(X)
                pred = reg.predict(X_scaled)
            else:
                pred = reg.predict(X)

            if "kw" in target:
                pred = np.maximum(0, pred)
            
            out_df[f"forecast_{target}"] = np.round(pred, 2 if "price" not in target else 4)

        # Consistent total renewable check
        out_df["forecast_total_renewable_kw"] = np.round(
            out_df["forecast_solar_generation_kw"] + out_df["forecast_wind_generation_kw"], 2
        )

        return out_df

    def compare_models(self, historical_df: pd.DataFrame, horizon_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Trains and compares Random Forest, SVR, Neural Network, and Ensemble on the target dataset.
        Returns prediction series from all models for side-by-side visualization.
        """
        model_names = ["random_forest", "svr", "neural_net", "ensemble"]
        comparisons = {}

        for m_name in model_names:
            core = ForecastingCore(model_type=m_name)
            core.train(historical_df)
            pred_df = core.predict(horizon_df)
            comparisons[m_name] = {
                "predictions": pred_df,
                "metrics": core.metrics
            }

        return comparisons

