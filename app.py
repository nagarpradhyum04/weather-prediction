import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier # New Model for Classification
from sklearn.metrics import accuracy_score
import numpy as np
import joblib 

# Define the filename.
CSV_FILE = 'weather.csv'
TEMP_MODEL_FILE = 'weather_temp_forecaster.joblib'
SUMMARY_MODEL_FILE = 'weather_summary_forecaster.joblib'

# --- 1. Data Loading and Feature Engineering ---

@st.cache_data
def load_data(file_path):
    """Loads the CSV data and cleans the date column."""
    try:
        df = pd.read_csv(file_path)
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
        df = df.set_index('Formatted Date')
        df.columns = df.columns.str.strip()
        
        # Select key columns for prediction and visualization
        required_cols = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Apparent Temperature (C)', 'Summary']
        df_clean = df[required_cols].dropna()

        return df_clean, df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 2. Model Training Functions ---

def create_lagged_data(df_clean, future_hours, target_col):
    """Creates the time-lagged features and target for forecasting."""
    # The 'shift(-N)' operation aligns the current row's features with the target N hours later.
    df_clean[f'Future_{target_col}'] = df_clean[target_col].shift(-future_hours)
    df_model = df_clean.dropna()
    
    X = df_model[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
    y = df_model[f'Future_{target_col}']
    
    return X, y, df_model

@st.cache_resource
def train_temp_forecasting_model(df_clean, future_hours):
    """Trains a Linear Regression model to predict Temperature (C) N hours into the future."""
    if future_hours <= 0: return None, 0.0, "Please select a prediction window greater than 0 hours."

    X, y, df_model = create_lagged_data(df_clean, future_hours, 'Temperature (C)')

    if df_model.empty: return None, 0.0, "Not enough data for this future window."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    joblib.dump(model, TEMP_MODEL_FILE)
    return model, mse, f"Temperature Model (T+{future_hours}h) Trained. MSE: {mse:.4f}"

@st.cache_resource
def train_summary_forecasting_model(df_clean, future_hours):
    """Trains a RandomForestClassifier model to predict Summary N hours into the future."""
    if future_hours <= 0: return None, 0.0, "Please select a prediction window greater than 0 hours."

    X, y, df_model = create_lagged_data(df_clean, future_hours, 'Summary')

    if df_model.empty: return None, 0.0, "Not enough data for this future window."

    # Use a Random Forest Classifier for discrete categories (Summary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(model, SUMMARY_MODEL_FILE)
    return model, accuracy, f"Summary Model (T+{future_hours}h) Trained. Accuracy: {accuracy:.2f}"

# --- 3. Streamlit App Layout and Logic ---

st.set_page_config(layout="wide", page_title="Weather Forecasting App")
st.title("Time Series Weather Forecasting Dashboard ⏳")
st.markdown("This dashboard predicts both the **future temperature (Regression)** and the **weather summary (Classification)**.")

# Load data
df_clean, df_raw = load_data(CSV_FILE)

if not df_clean.empty:
    st.sidebar.header("Forecasting Parameters")
    
    # User selects the time into the future to predict (N hours)
    future_hours = st.sidebar.slider(
        "Prediction Window (N hours ahead)", 
        min_value=1, 
        max_value=24, 
        value=6, 
        step=1
    )
    
    # Train both models based on the selected future_hours
    temp_predictor, temp_mse, temp_status = train_temp_forecasting_model(df_clean, future_hours)
    summary_predictor, summary_acc, summary_status = train_summary_forecasting_model(df_clean, future_hours)
    
    st.sidebar.markdown("---")
    st.sidebar.info(temp_status)
    st.sidebar.success(summary_status)
    
    
    # --- PREDICTION INTERFACE ---
    st.header(f"1. Forecast for {future_hours} Hours Ahead")
    st.markdown("Set the **current** weather conditions to get the forecasted **Temperature** and **Summary**.")
    
    if temp_predictor is not None and summary_predictor is not None:
        
        # Find sensible default ranges for sliders from the *clean* data
        temp_min, temp_max = df_clean['Temperature (C)'].min(), df_clean['Temperature (C)'].max()
        humidity_min, humidity_max = df_clean['Humidity'].min(), df_clean['Humidity'].max()
        wind_min, wind_max = df_clean['Wind Speed (km/h)'].min(), df_clean['Wind Speed (km/h)'].max()

        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_temp = st.slider(
                "Current Temperature (°C)", 
                min_value=float(np.floor(temp_min)), 
                max_value=float(np.ceil(temp_max)), 
                value=float(df_clean['Temperature (C)'].iloc[-1]), # Default to last recorded value
                step=0.1
            )
        with col2:
            input_humidity = st.slider(
                "Current Humidity (0.0 to 1.0)", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(df_clean['Humidity'].iloc[-1]), # Default to last recorded value
                step=0.01
            )
        with col3:
            input_wind = st.slider(
                "Current Wind Speed (km/h)", 
                min_value=0.0, 
                max_value=float(np.ceil(wind_max)), 
                value=float(df_clean['Wind Speed (km/h)'].iloc[-1]), # Default to last recorded value
                step=0.1
            )
            
        # Prepare input data for both models
        new_data = pd.DataFrame({
            'Temperature (C)': [input_temp],
            'Humidity': [input_humidity],
            'Wind Speed (km/h)': [input_wind]
        })
        
        # Make predictions
        predicted_temp = temp_predictor.predict(new_data)[0]
        predicted_summary = summary_predictor.predict(new_data)[0] # New Summary Prediction
        
        st.markdown("---")
        
        # Display Results
        col_summary, col_temp = st.columns(2)
        
        with col_temp:
            st.info(f"## 🌡️ Forecasted Temp: {predicted_temp:.2f} °C")
            
        with col_summary:
            st.success(f"## ☁️ Forecasted Summary: {predicted_summary}")
        
        st.markdown("---")
        
    else:
        st.warning("Prediction models could not be loaded or trained. Check data and future hours setting.")

    # --- VISUALIZATION SECTION ---
    st.header("2. Historical Data Trend")
    st.markdown("Visualize the historical data.")
    
    # Plot 1: Temperature over Time (Line Chart)
    st.subheader("Temperature Trend Over Time")
    
    min_date = df_raw.index.min().to_pydatetime()
    max_date = df_raw.index.max().to_pydatetime()
    
    date_range = st.slider(
        "Select Date Range for Visualization:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    filtered_time_df = df_raw[(df_raw.index >= pd.Timestamp(date_range[0])) & (df_raw.index <= pd.Timestamp(date_range[1]))]
    
    # Use Summary column to color the line for visual context
    color_col = 'Summary' if 'Summary' in filtered_time_df.columns else None

    temp_fig = px.line(
        filtered_time_df,
        y=['Temperature (C)', 'Apparent Temperature (C)'],
        color=color_col,
        title=f"Actual vs. Apparent Temperature Trend",
        labels={'value': 'Temperature (°C)', 'Formatted Date': 'Date'},
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(temp_fig, use_container_width=True)

else:
    st.error(f"Cannot run prediction. Please ensure the file `{CSV_FILE}` is present and correctly formatted.")
