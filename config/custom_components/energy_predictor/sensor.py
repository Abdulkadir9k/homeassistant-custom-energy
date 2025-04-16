"""Sensor platform for Energy Predictor using asynchronous updates."""
from datetime import datetime, timedelta
import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

# --- Configuration & Constants ---
_LOGGER = logging.getLogger(__name__)
DOMAIN = "energy_predictor" # Match the domain name in manifest.json and __init__.py

# Default update interval: 5 minutes
DEFAULT_SCAN_INTERVAL = timedelta(minutes=5)

# Define path relative to this file for model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
MODEL_FILENAME = "energy_prediction_pipeline_hybrid_tuned.joblib"
INFO_FILENAME = "preprocessing_info_hybrid_tuned.joblib"
BG_FILENAME = "shap_background_data.joblib"
# SAMPLE_FILENAME = "sample_input_structure.joblib" # Optional validation file

# Define input helper entity IDs (match your configuration.yaml)
INPUT_APPLIANCE = "input_select.appliance_type"
INPUT_TEMP = "input_number.outdoor_temperature"
INPUT_HH_SIZE = "input_number.household_size"

# Define required base columns for input_data dict before engineering
REQUIRED_INPUT_COLS = [
    'Appliance Type', 'Outdoor Temperature (°C)', 'Household Size', 'Season', 'timestamp'
]

# --- Platform Schema ---
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        # No specific platform config needed here as inputs are separate entities.
    }
)

# === Feature Engineering Function (Synchronous for Executor Job) ===
# IMPORTANT: Keep this function's logic identical to the one used in training.
def engineer_features_sync(df: pd.DataFrame, low_energy_appliances: list[str], lag_periods: list[int]) -> pd.DataFrame | None:
    """Applies feature engineering steps. Runs synchronously within executor.
    Handles missing columns/data gracefully for single instance prediction.
    """
    # --- (Function content remains IDENTICAL to the previous version) ---
    _LOGGER.debug("Applying feature engineering...")
    if df is None or df.empty:
        _LOGGER.error("Input DataFrame to engineer_features_sync is None or empty.")
        return None
    df = df.copy() # Work on a copy

    # --- Low Energy Appliance Feature ---
    if 'Appliance Type' in df.columns:
        df['Is_Low_Energy'] = df['Appliance Type'].apply(
            lambda x: 1 if x in low_energy_appliances else 0
        )
    else:
        df['Is_Low_Energy'] = 0
        _LOGGER.warning("Missing 'Appliance Type' column for 'Is_Low_Energy'. Defaulting to 0.")

    # --- Interaction Feature ---
    if 'Outdoor Temperature (°C)' in df.columns and 'Is_Low_Energy' in df.columns:
         df['Temp_Is_Low_Energy'] = df['Outdoor Temperature (°C)'] * df['Is_Low_Energy']
    else:
        df['Temp_Is_Low_Energy'] = 0
        _LOGGER.warning("Missing required columns for 'Temp_Is_Low_Energy'. Defaulting to 0.")

    # --- Datetime Feature Extraction & Cyclical Encoding ---
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']) and not df['timestamp'].isnull().any():
        try:
            dt = df['timestamp'].iloc[0] # Get the single timestamp
            df['Hour'] = dt.hour
            df['Day'] = dt.day
            df['Month'] = dt.month
            df['Weekday'] = dt.weekday()
            df['Year'] = dt.year # Keep Year (often treated as non-cyclical trend)

            max_vals = {'Hour': 23, 'Day': 31, 'Month': 12, 'Weekday': 6}
            for col, max_val in max_vals.items():
                if col in df.columns: # Check if base column (Hour, Day..) exists
                    if max_val > 0:
                         if pd.api.types.is_numeric_dtype(df[col]): # Ensure column is numeric
                             df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
                             df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
                         else:
                             _LOGGER.warning(f"Column '{col}' is not numeric, cannot apply sin/cos. Setting defaults.")
                             df[f'{col}_sin'] = 0; df[f'{col}_cos'] = 1
                    else:
                         df[f'{col}_sin'] = 0; df[f'{col}_cos'] = 1
        except Exception as e:
            _LOGGER.error(f"Error extracting datetime features: {e}. Setting defaults for cyclical features.")
            cyclical_cols = ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
                             'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos', 'Year']
            for col in cyclical_cols: df[col] = 0
    else:
        _LOGGER.warning("Timestamp missing, invalid, or contains NaNs; cyclical features using defaults.")
        cyclical_cols = ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
                         'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos', 'Year']
        for col in cyclical_cols: df[col] = 0

    # --- Lag Features (Add as Placeholders with NaNs) ---
    if lag_periods:
        target_col_base = 'Energy Consumption (kWh)' # Base name used in training
        for period in lag_periods:
            lag_col_name = f'{target_col_base}_lag_{period}h'
            if lag_col_name not in df.columns:
                df[lag_col_name] = np.nan
                _LOGGER.debug(f"Added placeholder NaN for missing lag: {lag_col_name}")

    _LOGGER.debug("Feature engineering complete.")
    return df


# === Sensor Setup Function ===
async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the sensor platform."""
    # --- (Function content remains IDENTICAL to the previous version) ---
    _LOGGER.info("Setting up Energy Predictor sensor platform asynchronously.")
    try:
        predictor_sensor = EnergyPredictionSensor(hass)
        async_add_entities([predictor_sensor], True)
        _LOGGER.info("EnergyPredictionSensor added to HA.")
    except Exception as e:
        _LOGGER.error(f"Fatal error during EnergyPredictionSensor initialization: {e}", exc_info=True)


# === Sensor Entity Class ===
class EnergyPredictionSensor(SensorEntity):
    """Representation of an Energy Prediction Sensor."""

    _attr_should_poll = True

    def __init__(self, hass: HomeAssistant):
        """Initialize the sensor."""
        # --- (Initialization logic remains IDENTICAL, including loading artifacts) ---
        self._hass = hass
        self._attr_name = "Energy Consumption Predictor"
        self._attr_unique_id = f"{DOMAIN}_predictor_main"
        self._attr_native_unit_of_measurement = "kWh"
        self._attr_icon = "mdi:chart-line"
        self._attr_attribution = "Energy prediction based on XGBoost model"
        self._attr_native_value = None
        self._attr_extra_state_attributes: dict[str, Any] = {"status": "Initializing"}
        self._model = None
        self._preprocessing_info = None
        self._explainer = None
        self._expected_features_list: list[str] | None = None
        self._low_energy_appliances: list[str] = []
        self._lag_periods: list[int] = []
        self._base_value: float | None = None
        self._load_artifacts()
        self._initialize_explainer()
        self._attr_extra_state_attributes["status"] = "Initialized"


    def _load_artifacts(self):
        """Load model and config files synchronously during init."""
        # --- (Method content remains IDENTICAL) ---
        _LOGGER.info("Loading prediction artifacts...")
        try:
            model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            self._model = joblib.load(model_path)
            _LOGGER.debug(f"Model loaded from {model_path}")

            preprocessing_path = os.path.join(MODEL_DIR, INFO_FILENAME)
            self._preprocessing_info = joblib.load(preprocessing_path)
            self._expected_features_list = self._preprocessing_info.get('engineered_features_list')
            self._low_energy_appliances = self._preprocessing_info.get('low_energy_appliances', [])
            self._lag_periods = self._preprocessing_info.get('lag_periods', [])
            _LOGGER.debug(f"Preprocessing info loaded. Expected features: {len(self._expected_features_list) if self._expected_features_list else 'N/A'}")

            _LOGGER.info("Prediction artifacts loaded successfully.")
            self._attr_extra_state_attributes["artifact_status"] = "Loaded"

        except FileNotFoundError as e:
            _LOGGER.error(f"Artifact file not found: {e}. Ensure files are in {MODEL_DIR}")
            self._attr_extra_state_attributes["artifact_status"] = f"Error: {e}"
            raise
        except Exception as e:
            _LOGGER.error(f"Error loading artifacts: {e}", exc_info=True)
            self._attr_extra_state_attributes["artifact_status"] = f"Error: {e}"
            raise

    def _initialize_explainer(self):
        """Initialize SHAP explainer synchronously during init."""
         # --- (Method content remains IDENTICAL) ---
        if not self._model:
            _LOGGER.error("Cannot initialize SHAP: Model not loaded.")
            return
        _LOGGER.debug("Initializing SHAP explainer...")
        try:
            xgb_model = self._model.named_steps['regressor']
            self._explainer = shap.TreeExplainer(xgb_model)
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                 base_value = base_value.mean()
            self._base_value = round(float(base_value), 3)
            self._attr_extra_state_attributes["base_value"] = self._base_value
            _LOGGER.info(f"SHAP explainer initialized. Base value: {self._base_value:.3f}")
        except KeyError:
            _LOGGER.error("Cannot initialize SHAP: 'regressor' step not found in pipeline.")
            self._explainer = None
        except Exception as e:
            _LOGGER.error(f"Error initializing SHAP: {e}", exc_info=True)
            self._explainer = None

    @property
    def scan_interval(self) -> timedelta:
        """Return the polling interval."""
        # --- (Method content remains IDENTICAL) ---
        return DEFAULT_SCAN_INTERVAL

    async def async_update(self) -> None:
        """Fetch new state data for the sensor asynchronously."""
        # --- (Method content remains IDENTICAL) ---
        if self._model is None or self._preprocessing_info is None:
            _LOGGER.error("Cannot update sensor: Artifacts not loaded correctly.")
            self._attr_native_value = None
            self._attr_extra_state_attributes["status"] = "Error - Artifacts missing"
            return

        _LOGGER.debug(f"Scheduling update task for {self.entity_id}")
        self._attr_extra_state_attributes["status"] = "Updating"

        try:
            results = await self._hass.async_add_executor_job(
                self._perform_prediction_and_explanation
            )
            if results:
                self._attr_native_value = results.get("prediction")
                self._attr_extra_state_attributes.update(results.get("attributes", {}))
                self._attr_extra_state_attributes["status"] = "OK"
                _LOGGER.debug(f"Update successful for {self.entity_id}. New state: {self._attr_native_value}")
            else:
                _LOGGER.warning(f"Prediction/explanation task for {self.entity_id} returned no results.")
                self._attr_native_value = None
                self._attr_extra_state_attributes["status"] = "Error - No results from task"
        except Exception as e:
            _LOGGER.error(f"Error during async update execution for {self.entity_id}: {e}", exc_info=True)
            self._attr_native_value = None
            self._attr_extra_state_attributes["status"] = f"Error - {e}"


    # --- SYNCHRONOUS HELPER RUN IN EXECUTOR ---
    def _perform_prediction_and_explanation(self) -> dict[str, Any] | None:
        """Performs data fetching, processing, prediction, and explanation.
        Returns None on failure, or dict {"prediction": ..., "attributes": ...} on success.
        """
        _LOGGER.debug("Executor job started.")
        start_time = time.monotonic()

        # --- 1. Get Input Data ---
        input_data = self._get_ha_input_data_sync()
        if input_data is None:
            _LOGGER.warning("Executor job: Failed to get valid input data.")
            # Return structure indicating failure
            return {"prediction": None, "attributes": {"status": "Error - Input data unavailable", "error": "Failed to get valid input data from HA states."}}

        # --- 2. Engineer Features & Align Columns ---
        try:
            processed_df = engineer_features_sync(
                pd.DataFrame([input_data]),
                self._low_energy_appliances,
                self._lag_periods
            )
            if processed_df is None:
                 raise ValueError("Feature engineering function returned None.")
            if self._expected_features_list:
                current_cols = processed_df.columns.tolist()
                missing_cols = set(self._expected_features_list) - set(current_cols)
                for col in missing_cols: processed_df[col] = np.nan
                extra_cols = set(current_cols) - set(self._expected_features_list)
                if extra_cols: processed_df = processed_df.drop(columns=list(extra_cols))
                processed_df = processed_df[self._expected_features_list] # Reorder
            else:
                 _LOGGER.error("Cannot align columns: 'engineered_features_list' not available.")
                 raise ValueError("Missing expected features list.")
        except Exception as e:
            _LOGGER.error(f"Executor job: Error during feature engineering: {e}", exc_info=True)
            return {"prediction": None, "attributes": {"error": f"Feature Engineering Error: {e}", "status": "Error"}}

        # --- 3. Prediction ---
        try:
            prediction = self._model.predict(processed_df)[0]
            prediction_rounded = round(float(prediction), 3)
        except Exception as e:
            _LOGGER.error(f"Executor job: Error during prediction: {e}", exc_info=True)
            _LOGGER.error(f"Input to predict:\n{processed_df.to_string()}")
            return {"prediction": None, "attributes": {"error": f"Prediction Error: {e}", "status": "Error"}}

        # --- 4. Explanation (Structured + Natural Language) ---
        explanation_dict = {"status": "OK"} # Start with OK status for details
        nl_summary = "Explanation not available." # Default NL text

        if self._explainer:
            try:
                start_shap_time = time.monotonic()
                preprocessor = self._model.named_steps['preprocessor']
                transformed_input = preprocessor.transform(processed_df)
                shap_values = self._explainer.shap_values(transformed_input)[0]

                if hasattr(preprocessor, 'get_feature_names_out'):
                     feature_names = preprocessor.get_feature_names_out()
                else: feature_names = [f'feature_{i}' for i in range(len(shap_values))]

                # Create structured details
                feature_impacts = []
                for i, name in enumerate(feature_names):
                     clean_name = name.replace('num__', '').replace('cat__', '').replace('remainder__', '')
                     feature_impacts.append({
                         'feature': clean_name,
                         'impact': round(float(shap_values[i]), 4)
                     })
                feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
                explanation_dict["top_factors_detail"] = feature_impacts[:5] # Store top 5 structured

                shap_time = round(time.monotonic() - start_shap_time, 2)
                explanation_dict["shap_calc_time_sec"] = shap_time
                _LOGGER.debug(f"SHAP explanation generated in {shap_time} seconds.")

                # *** NEW: Generate Natural Language Summary ***
                nl_summary = self._generate_nl_explanation(
                    prediction=prediction_rounded,
                    top_factors=explanation_dict["top_factors_detail"],
                    input_values=input_data # Pass original inputs for context
                )
                _LOGGER.debug("Natural language explanation generated.")

            except Exception as e:
                _LOGGER.error(f"Executor job: Error generating SHAP explanation: {e}", exc_info=True)
                explanation_dict = {"error": f"SHAP Error: {e}", "status": "Error"}
                nl_summary = f"Explanation Error: {e}"
        else:
            _LOGGER.warning("Executor job: SHAP explainer not available, skipping explanation.")
            explanation_dict = {"info": "SHAP explainer not initialized", "status": "Warning"}
            nl_summary = "SHAP explainer not initialized."

        # --- 5. Prepare Results for async_update ---
        end_time = time.monotonic()
        processing_time = round(end_time - start_time, 2)
        _LOGGER.debug(f"Executor job finished in {processing_time} seconds.")

        attributes_update = {
            "last_updated": datetime.now().isoformat(),
            "prediction_time_sec": processing_time,
            "explanation_details": explanation_dict, # Structured data
            "explanation_summary": nl_summary,       # <<< NEW: Natural Language Text
            "raw_inputs": input_data,
            "error": None, # Clear previous error
            "status": explanation_dict.get("status", "OK") # Reflect SHAP status here too
            # base_value is already in attributes from init
        }

        return {"prediction": prediction_rounded, "attributes": attributes_update}

    # --- Moved _get_ha_input_data_sync here (no changes within the function needed) ---
    def _get_ha_input_data_sync(self) -> dict[str, Any] | None:
        """Synchronously fetches required data from HA states. Runs in executor."""
        # --- (Method content remains IDENTICAL to the previous working version) ---
        _LOGGER.debug("Executor job: Getting HA input states.")
        try:
            appliance_state_obj = self._hass.states.get(INPUT_APPLIANCE)
            temp_state_obj = self._hass.states.get(INPUT_TEMP)
            size_state_obj = self._hass.states.get(INPUT_HH_SIZE)
            if not all([appliance_state_obj, temp_state_obj, size_state_obj]):
                _LOGGER.warning("One or more input entities not found.")
                return None
            appliance_value = appliance_state_obj.state
            temp_value = temp_state_obj.state
            size_value = size_state_obj.state
            if any(s in [None, 'unknown', 'unavailable'] for s in [appliance_value, temp_value, size_value]):
                 _LOGGER.warning("One or more input entities has an invalid state...")
                 return None
            now = datetime.now()
            month = now.month
            season = ('Summer' if 5 <= month <= 8 else
                      'Winter' if month >= 11 or month <= 2 else
                      'Spring/Fall')
            input_data = {
                'Appliance Type': appliance_value,
                'Outdoor Temperature (°C)': float(temp_value),
                'Household Size': int(float(size_value)),
                'Season': season,
                'timestamp': pd.Timestamp(now)
            }
            if not all(key in input_data for key in REQUIRED_INPUT_COLS):
                 _LOGGER.error("Constructed input data is missing required keys...")
                 return None
            _LOGGER.debug(f"Successfully retrieved and processed input data: {input_data}")
            return input_data
        except ValueError as e:
             _LOGGER.error(f"Error converting HA state value to required number type: {e}")
             return None
        except Exception as e:
            _LOGGER.error(f"Unexpected error getting HA input data: {e}", exc_info=True)
            return None

    # *** NEW: Natural Language Explanation Helper Method ***
    def _generate_nl_explanation(self, prediction: float, top_factors: list[dict], input_values: dict) -> str:
        """Generates a human-readable explanation string."""
        if self._base_value is None:
            base_text = "unavailable"
        else:
            base_text = f"{self._base_value:.2f} kWh"

        # Start building the natural language explanation
        nl_lines = [
            f"Predicted consumption: {prediction:.2f} kWh (compared to a baseline average of {base_text})."
        ]

        if not top_factors:
             nl_lines.append("No specific factors identified for explanation.")
             return "\n".join(nl_lines)

        nl_lines.append("Key factors influencing this prediction:")

        for factor in top_factors:
            # Skip factors with negligible impact
            if abs(factor['impact']) < 0.005: # Adjusted threshold slightly higher
                 continue

            feature_clean = factor['feature'] # Already cleaned name from dict
            impact = factor['impact']
            base_feature_name = feature_clean.split('_')[0] # Try to get base name (e.g., 'Appliance Type' from 'Appliance Type_HVAC')
            value_text = ""

            # Try to get the actual input value for context for non-encoded features
            if '_' not in feature_clean and feature_clean in input_values:
                 input_val = input_values[feature_clean]
                 if isinstance(input_val, (float, int)):
                      value_text = f" ({input_val:.1f})" # Format numeric values
                 elif isinstance(input_val, str):
                      value_text = f" ('{input_val}')"

            # Handle one-hot encoded features better
            display_name = feature_clean
            if base_feature_name == 'Appliance Type' and '_' in feature_clean:
                display_name = f"Appliance = '{feature_clean.split('_')[-1]}'"
            elif base_feature_name == 'Season' and '_' in feature_clean:
                 display_name = f"Season = '{feature_clean.split('_')[-1]}'"
            elif base_feature_name in ['Hour', 'Day', 'Month', 'Weekday'] and ('sin' in feature_clean or 'cos' in feature_clean):
                 # Make cyclical features more readable
                 time_part = feature_clean.split('_')[0]
                 trig_part = feature_clean.split('_')[1]
                 display_name = f"Time Factor ({time_part} {trig_part})" # Simpler representation
            # Add value context if available and not already part of name
            elif value_text:
                 display_name += value_text


            # Determine direction and strength
            direction = "increased" if impact > 0 else "decreased"
            magnitude = abs(impact)
            if magnitude > 0.5: strength = "significantly"
            elif magnitude > 0.1: strength = "moderately"
            else: strength = "slightly" # Removed "minimally" for brevity

            nl_lines.append(
                 f"- {display_name}: {strength} {direction} prediction by {magnitude:.3f} kWh."
            )

        # Add a concluding remark if few factors shown
        if len(nl_lines) <= 2 : # Only header + baseline line
             nl_lines.append("Overall impact of factors is minimal or complex.")
        elif len(nl_lines) < 6: # Fewer than 5 factors shown
             nl_lines.append("Other factors had smaller contributions.")


        return "\n".join(nl_lines)
