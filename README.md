# AI Energy Predictor & Explainer - Home Assistant Integration

## Overview

This project integrates a machine learning model, trained to predict energy consumption for specific household appliances, into Home Assistant. It leverages the SHAP library to provide explanations for these predictions, aiming to increase user awareness of energy usage patterns.

The core goals of this integration are:
1.  **Predict Energy Consumption:** Provide periodic forecasts for selected appliance energy usage based on user inputs (appliance type, temperature, household size).
2.  **Explain Predictions:** Offer human-readable explanations detailing the key factors driving the current prediction (e.g., "High temperature significantly increased prediction").
3.  **Visualize & Alert:** Display the prediction and explanation within the Home Assistant Lovelace UI and trigger alerts when predicted consumption exceeds a threshold.
4.  **Foundation for Control:** Lay the groundwork for future AI-driven actions (e.g., adjusting thermostats, delaying tasks) based on predictions, with user override capabilities.

## Technology Stack

*   **Machine Learning:** Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Joblib
*   **Integration Platform:** Home Assistant Core
*   **HA Integration Method:** Custom Component (`sensor` platform)

## Model Training Overview

The prediction model used by this integration was trained separately to forecast energy consumption (kWh) for individual appliance usage events.

*   **Data Source:** Utilized a smart home energy consumption dataset containing timestamps, appliance types, outdoor temperatures, household size, and energy readings.
*   **Features:** Key features included cyclical encoding of time components (hour, day, month, weekday), appliance type (one-hot encoded), temperature, household size, low-energy flags, temperature interactions, and lagged energy consumption values (e.g., 1-hour and 24-hour lags).
*   **Model:** The final model is an XGBoost Regressor (`xgboost.XGBRegressor`).
*   **Tuning:** Hyperparameters were optimized using a hybrid approach: initial broad search with `RandomizedSearchCV` followed by a focused `GridSearchCV`, both employing `TimeSeriesSplit` cross-validation suitable for time-series data.
*   **Evaluation:** Model selection was primarily based on minimizing Mean Absolute Error (MAE) on the time-series test split.
*   **Explainability:** The SHAP (SHapley Additive exPlanations) library was used during development and integrated here to understand feature contributions to individual predictions.

## Home Assistant Integration Approach

The integration is implemented as a **Custom Component** within Home Assistant. This involves creating a new `sensor` platform named `energy_predictor`.

Key implementation details:
*   **Asynchronous Updates:** The sensor (`sensor.energy_consumption_predictor`) uses Home Assistant's asynchronous architecture (`async_update` and `async_add_executor_job`).
*   **Background Processing:** Computationally intensive tasks (model prediction and SHAP explanation generation) are run in Home Assistant's executor thread pool to avoid blocking the main event loop, ensuring HA remains responsive.
*   **Polling:** The sensor polls for input states and updates its prediction on a regular schedule (default 5 minutes).

## Key Components within Home Assistant

1.  **Custom Sensor (`sensor.energy_consumption_predictor`):**
    *   **State:** The predicted energy consumption in kWh for the next interval.
    *   **Attributes:** Contains detailed information:
        *   `base_value`: The average prediction baseline (from SHAP).
        *   `explanation_summary`: A **human-readable text summary** explaining the key factors influencing the current prediction.
        *   `explanation_details`: Structured data with top SHAP factors and their numerical impact (for potential advanced use).
        *   `raw_inputs`: The specific input values (Appliance, Temp, Size) used for the last prediction.
        *   `status`: Indicates the operational status ("OK", "Updating", "Error").
        *   `prediction_time_sec`: Time taken for the last prediction/explanation cycle.
        *   `last_updated`: Timestamp of the last update.

2.  **Input Helpers (User Controls):**
    *   `input_select.appliance_type`: Allows the user to select which appliance the prediction should target. **Options must match training data categories.**
    *   `input_number.outdoor_temperature`: Allows the user to input or link to the current outdoor temperature.
    *   `input_number.household_size`: Allows the user to set the household size used by the model.

3.  **Override Helper (Preparation for Phase 3):**
    *   `input_boolean.energy_ai_control_enabled`: A toggle switch allowing the user to enable or disable potential future AI-driven actions (e.g., device control). Initially `off`.

4.  **Automation (`automation.energy_prediction_high_alert`):**
    *   **Trigger:** Activates when `sensor.energy_consumption_predictor` goes above a defined threshold.
    *   **Condition:** Only runs if `input_boolean.energy_ai_control_enabled` is `on`.
    *   **Action:** Sends a notification (currently `persistent_notification`) containing the prediction value and the *human-readable prediction explanation summary* from the sensor's attributes.

## Setup Instructions within Home Assistant

**(Assumes HA Core on Linux/venv and model artifacts are available locally)**

1.  **Create Directories:**
    ```bash
    # Navigate to HA config directory (e.g., ~/core/config or /config)
    cd <ha_config_dir>
    mkdir -p custom_components/energy_predictor/model_files
    ```

2.  **Copy Artifacts:** Place the following files into `custom_components/energy_predictor/model_files/`:
    *   `energy_prediction_pipeline_hybrid_tuned.joblib`
    *   `preprocessing_info_hybrid_tuned.joblib`
    *   `shap_background_data.joblib`
    *   `sample_input_structure.joblib` (Optional)

3.  **Install Dependencies:** Activate your HA virtual environment and install specific versions:
    ```bash
    # Example: cd ~/core && source venv/bin/activate
    source <path_to_your_venv>/bin/activate
    pip install --upgrade pip
    pip install xgboost==2.1.4 scikit-learn==1.6.1 pandas==2.2.2 shap==0.47.1 joblib==1.4.2
    # Deactivate if desired
    ```

4.  **Create Custom Component Files:**
    *   **`custom_components/energy_predictor/__init__.py`:**
        ```python
        """The Energy Predictor custom component."""
        DOMAIN = "energy_predictor"
        ```
    *   **`custom_components/energy_predictor/manifest.json`:**
        ```json
        {
          "domain": "energy_predictor",
          "name": "Energy Consumption Predictor",
          "version": "1.1.0",
          "documentation": "https://github.com/Abdulkadir9k/homeassistant-custom-energy",
          "requirements": [
            "xgboost==2.1.4",
            "scikit-learn==1.6.1",
            "pandas==2.2.2",
            "shap==0.47.1",
            "joblib==1.4.2"
          ],
          "dependencies": [],
          "codeowners": ["Abdulkadir9k"],
          "iot_class": "calculated",
          "config_flow": false
        }
        ```
        *(Replace placeholders)*
    *   **`custom_components/energy_predictor/sensor.py`:** Place the full Python code for the `sensor.py` file (including the `EnergyPredictionSensor` class, `engineer_features_sync`, and `async_setup_platform`) developed in our previous steps into this file.

5.  **Configure Home Assistant (`configuration.yaml`):** Add/merge the following blocks into your main configuration file:
    ```yaml
    # --- Input Helpers ---
    input_select:
      appliance_type:
        name: Appliance Type Select
        options: # <-- Customize these options EXACTLY as per your model training!
          - HVAC
          - Fridge
          - Lights
          # ... etc ...
        initial: HVAC
        icon: mdi:power-plug

    input_number:
      household_size:
        name: Household Size Input
        min: 1
        max: 10
        step: 1
        initial: 3
        unit_of_measurement: "people"
        mode: box
        icon: mdi:account-group
      outdoor_temperature:
        name: Outdoor Temperature Input
        min: -20
        max: 55
        step: 0.1
        initial: 20.0
        unit_of_measurement: "°C"
        mode: box
        icon: mdi:thermometer

    input_boolean: # For future actions override
      energy_ai_control_enabled:
        name: Enable Energy AI Control Actions
        initial: off
        icon: mdi:robot-confused

    # --- Sensor Platform ---
    sensor:
      # --- Keep any existing sensor platforms ---
      - platform: energy_predictor
        # scan_interval can be optionally overridden here, but defaults to 5 mins in code
    ```

6.  **Configure Automation (`automations.yaml`):** Add the automation YAML block (triggering on `sensor.energy_consumption_predictor`, conditioning on `input_boolean.energy_ai_control_enabled`, action notifying with explanation) to your automations file or configure via UI.

7.  **Restart Home Assistant:** A full restart is required to load the custom component and configuration changes.

## Lovelace Configuration

Add the following cards to a manual Lovelace dashboard (or use the UI editor) to interact with and display the sensor:

```yaml
type: vertical-stack
cards:
  - type: entities
    title: Prediction Inputs
    show_header_toggle: false
    entities:
      - entity: input_select.appliance_type
        name: Appliance Target
      - entity: input_number.outdoor_temperature
        name: Outdoor Temp (°C)
      - entity: input_number.household_size
        name: Household Size
      - entity: input_boolean.energy_ai_control_enabled # Add the override toggle
        name: Allow AI Actions
  - type: gauge
    entity: sensor.energy_consumption_predictor
    name: Predicted Energy (Next Interval)
    # ... (rest of gauge config: unit, min, max, severity) ...
  - type: markdown
    title: Prediction Explanation
    content: >
      {# ... (Paste the final Markdown content displaying explanation_summary etc.) ... #}