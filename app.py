import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io
from sklearn.linear_model import LinearRegression
try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    st.warning("xlsxwriter is not installed. Excel export will be disabled.")
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("fpdf is not installed. PDF export will be disabled.")
import base64

# Minimal CSS for styling
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
    color: #333;
}
h1, h2, h3 {
    color: #2c3e50;
}
.section {
    margin: 1em 0;
    padding: 1em;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #ffffff;
}
.st-expander {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 5px;
}
.stMetric {
    background-color: #ffffff !important;
    padding: 1em;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #333 !important;
}
.stMetric label {
    color: #333 !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #333 !important;
}
.key-metrics-section {
    background-color: #ffffff !important;
    padding: 1em;
    color: #333 !important;
}
.key-metrics-section * {
    color: #333 !important;
}
.logout-button {
    position: fixed;
    bottom: 10px;
    width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for login and filtered data
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None

# Login Page
if not st.session_state.logged_in:
    st.title("Health Claim Cost Prediction - Login")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Load the dataset and model with error handling
    try:
        data = pd.read_csv("final_merged_synthea_cleaned98.csv")
        model = joblib.load("xgb_model_new.pkl")
    except Exception as e:
        st.error(f"Error loading dataset or model: {e}")
        st.stop()

    # Preprocess the data
    try:
        # Define required columns based on your dataset
        required_columns = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                            "ENCOUNTERCLASS", "CODE", "TOTAL_CLAIM_COST"]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dataset is missing required columns: {missing_columns}")
            st.stop()

        # Handle patient ID column
        patient_id_column = None
        possible_patient_columns = ["PATIENTID", "PATIENT_ID", "Id", "ID", "PATIENT"]
        for col in possible_patient_columns:
            if col in data.columns:
                patient_id_column = col
                break
        
        if patient_id_column:
            data = data.rename(columns={patient_id_column: "PATIENT"})
        else:
            st.warning("No patient ID column found. Creating a placeholder PATIENT column.")
            data["PATIENT"] = [f"patient_{i}" for i in range(len(data))]
        
        required_columns = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                            "ENCOUNTERCLASS", "CODE", "TOTAL_CLAIM_COST", "PATIENT"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dataset is missing required columns: {missing_columns}")
            st.stop()

        # Map TOTAL_CLAIM_COST to TOTALCOST for compatibility with your code
        data = data.rename(columns={"TOTAL_CLAIM_COST": "TOTALCOST"})

        # Calculate encounter duration if START and STOP are present; otherwise, set to 0
        if "START" in data.columns and "STOP" in data.columns:
            data["ENCOUNTER_DURATION"] = (pd.to_datetime(data["STOP"]) - pd.to_datetime(data["START"])).dt.days
        else:
            data["ENCOUNTER_DURATION"] = 0

        # Add missing columns expected by the model with default values
        model_expected_columns = [
            "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
            "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2"
        ]
        for col in model_expected_columns:
            if col not in data.columns:
                if col == "STATE":
                    data[col] = "Unknown"
                else:
                    data[col] = 0

        data = data.fillna(0)

        # Extract year for filtering and forecasting
        if "START" in data.columns:
            data["START_YEAR"] = pd.to_datetime(data["START"]).dt.year
        else:
            data["START_YEAR"] = 2025

        # Prepare features for one-hot encoding
        features = [
            "AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", "ENCOUNTERCLASS", "CODE", "ENCOUNTER_DURATION",
            "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
            "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2"
        ]
        X = data[features]

        # Define categorical columns
        categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]

        # Convert categorical columns to string
        for col in categorical_cols:
            X[col] = X[col].astype(str)

        # Perform one-hot encoding
        X_encoded = pd.get_dummies(X, columns=categorical_cols)

        # Store the encoded feature names
        model_features = X_encoded.columns.tolist()

        # Store categories for each categorical column
        categories = {col: data[col].astype(str).unique() for col in categorical_cols}
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        st.stop()

    # Initialize filtered data if not already set
    if st.session_state.filtered_data is None:
        st.session_state.filtered_data = data.copy()

    # Main App
    st.title("Health Claim Cost Prediction")

    # Add logout button in the sidebar at the bottom
    with st.sidebar:
        st.markdown("<div class='logout-button'>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Data Filters", "Key Metrics", "Claim Forecast", "Data Visualizations", "Resource Allocation", "Race Claim Distributions", "Prediction Cost", "Data Export"])

    # Tab 1: Data Filters
    with tab1:
        st.header("Data Filters")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        years = list(range(1950, 2026))
        start_year = st.selectbox("Start Year:", years, index=1)
        end_year = st.selectbox("End Year:", years, index=len(years)-1)

        if st.button("Apply Year Range"):
            try:
                filtered_data = data[(data["START_YEAR"] >= start_year) & (data["START_YEAR"] <= end_year)]
                st.session_state.filtered_data = filtered_data
                st.write(f"Filtered Data: {len(filtered_data)} records")
            except Exception as e:
                st.error(f"Error applying year range filter: {e}")
        else:
            st.write(f"Filtered Data: {len(st.session_state.filtered_data)} records")

        st.write("**Debugging Information:**")
        st.write(f"Unique Races in Original Data: {data['RACE'].unique()}")
        st.write(f"Unique Races in Filtered Data: {st.session_state.filtered_data['RACE'].unique()}")
        st.write(f"Unique Encounter Classes in Original Data: {data['ENCOUNTERCLASS'].unique()}")
        st.write(f"Unique Encounter Classes in Filtered Data: {st.session_state.filtered_data['ENCOUNTERCLASS'].unique()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Key Metrics
    with tab2:
        st.header("Key Metrics")
        st.markdown("<div class='section key-metrics-section'>", unsafe_allow_html=True)

        filtered_data = st.session_state.filtered_data
        try:
            avg_claim_cost = filtered_data["TOTALCOST"].mean()
            total_claims = len(filtered_data)
            avg_age = filtered_data["AGE"].mean()
            total_patients = filtered_data["PATIENT"].nunique()
            avg_encounter_duration = filtered_data["ENCOUNTER_DURATION"].mean()

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Average Claim Cost", value=f"${avg_claim_cost:.2f}")
                st.metric(label="Total Number of Claims", value=total_claims)
            
            with col2:
                st.metric(label="Average Patient Age", value=f"{avg_age:.1f} years")
                st.metric(label="Total Number of Patients", value=total_patients)
            
            with col3:
                st.metric(label="Average Encounter Duration", value=f"{avg_encounter_duration:.1f} days")
        except Exception as e:
            st.error(f"Error calculating key metrics: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Claim Forecast
    with tab3:
        st.header("Claim Forecast")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            yearly_costs = data.groupby("START_YEAR")["TOTALCOST"].sum().reset_index()
            X = yearly_costs["START_YEAR"].values.reshape(-1, 1)
            y = yearly_costs["TOTALCOST"].values
            forecast_model = LinearRegression()
            forecast_model.fit(X, y)
            current_year = data["START_YEAR"].max()
            future_years = np.array([current_year + i for i in range(1, 6)]).reshape(-1, 1)
            forecasted_costs = forecast_model.predict(future_years)
            
            st.write("**Claim Cost Forecast for the Next 5 Years:**")
            forecast_df = pd.DataFrame({
                "Year": future_years.flatten(),
                "Forecasted Cost ($)": forecasted_costs
            })
            st.write(forecast_df)
            
            historical_df = pd.DataFrame({
                "Year": yearly_costs["START_YEAR"],
                "Cost": yearly_costs["TOTALCOST"],
                "Type": "Historical"
            })
            forecast_df_plot = pd.DataFrame({
                "Year": future_years.flatten(),
                "Cost": forecasted_costs,
                "Type": "Forecasted"
            })
            plot_df = pd.concat([historical_df, forecast_df_plot])
            st.line_chart(plot_df.set_index("Year")["Cost"])
        except Exception as e:
            st.error(f"Error generating claim forecast: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Data Visualizations
    with tab4:
        st.header("Data Visualizations")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.subheader("Total Cost Distribution")
        try:
            st.bar_chart(data["TOTALCOST"].value_counts())
        except Exception as e:
            st.error(f"Error creating Total Cost Distribution chart: {e}")

        st.subheader("Age vs Total Cost")
        try:
            st.scatter_chart(data[["AGE", "TOTALCOST"]])
        except Exception as e:
            st.error(f"Error creating Age vs Total Cost chart: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Resource Allocation
    with tab5:
        st.header("Resource Allocation")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            unique_encounter_classes = st.session_state.filtered_data["ENCOUNTERCLASS"].unique()
            st.write(f"**Unique Encounter Classes in Filtered Data:** {unique_encounter_classes}")

            resource_allocation = st.session_state.filtered_data.groupby("ENCOUNTERCLASS")["TOTALCOST"].sum().reset_index()
            st.write("**Total Claim Costs by Encounter Class:**")
            if resource_allocation.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(resource_allocation)
                st.bar_chart(resource_allocation.set_index("ENCOUNTERCLASS")["TOTALCOST"])
            
            avg_cost_by_encounter = st.session_state.filtered_data.groupby("ENCOUNTERCLASS")["TOTALCOST"].mean().reset_index()
            st.write("**Average Claim Cost by Encounter Class:**")
            if avg_cost_by_encounter.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(avg_cost_by_encounter)
        except Exception as e:
            st.error(f"Error analyzing resource allocation: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 6: Race Claim Distributions
    with tab6:
        st.header("Race Claim Distributions")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            unique_races = st.session_state.filtered_data["RACE"].unique()
            st.write(f"**Unique Races in Filtered Data:** {unique_races}")

            race_distribution = st.session_state.filtered_data.groupby("RACE")["TOTALCOST"].sum().reset_index()
            st.write("**Total Claim Costs by Race (Proxy for Region):**")
            if race_distribution.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(race_distribution)
                st.bar_chart(race_distribution.set_index("RACE")["TOTALCOST"])
            
            avg_cost_by_race = st.session_state.filtered_data.groupby("RACE")["TOTALCOST"].mean().reset_index()
            st.write("**Average Claim Cost by Race (Proxy for Region):**")
            if avg_cost_by_race.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(avg_cost_by_race)
        except Exception as e:
            st.error(f"Error analyzing regional claim distributions: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 7: Prediction Cost
    with tab7:
        st.header("Prediction Cost")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.subheader("Enter Patient Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 0, 100, 30, key="pred_age")
            gender = st.selectbox("Gender", categories["GENDER"], key="pred_gender")
            race = st.selectbox("Race", categories["RACE"], key="pred_race")
            ethnicity = st.selectbox("Ethnicity", categories["ETHNICITY"], key="pred_ethnicity")
        
        with col2:
            income = st.slider("Income ($)", 0, 100000, 50000, key="pred_income")
            encounter_class = st.selectbox("Encounter Class", categories["ENCOUNTERCLASS"], key="pred_encounter_class")
            code = st.selectbox("Procedure Code", categories["CODE"], key="pred_code")
            encounter_duration = st.slider("Encounter Duration (days)", 0, 30, 1, key="pred_encounter_duration")

        # Create input data with all model-expected columns
        input_data = pd.DataFrame({
            "AGE": [age],
            "GENDER": [str(gender)],
            "RACE": [str(race)],
            "ETHNICITY": [str(ethnicity)],
            "INCOME": [income],
            "ENCOUNTERCLASS": [str(encounter_class)],
            "CODE": [str(code)],
            "ENCOUNTER_DURATION": [encounter_duration],
            "PAYER_COVERAGE": [0],
            "BASE_ENCOUNTER_COST": [0],
            "AVG_CLAIM_COST": [0],
            "STATE": ["Unknown"],
            "NUM_DIAG1": [0],
            "HEALTHCARE_EXPENSES": [0],
            "NUM_ENCOUNTERS": [0],
            "NUM_DIAG2": [0]
        })

        try:
            # Show input data for transparency
            st.write("**Input Data:**")
            st.write(input_data)

            # Define categorical columns
            categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]

            # Perform one-hot encoding
            input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)

            # Reindex to match model_features
            input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)

            # Verify that original categorical columns are not in input_data_encoded
            remaining_categorical = [col for col in categorical_cols if col in input_data_encoded.columns]
            if remaining_categorical:
                st.warning(f"Original categorical columns still present after encoding: {remaining_categorical}")
        except Exception as e:
            st.error(f"Error encoding input data: {e}")
            st.stop()

        if st.button("Predict Cost"):
            try:
                # Ensure input_data_encoded is a numpy array for prediction
                prediction_input = input_data_encoded.values
                prediction = model.predict(prediction_input)[0]
                st.write(f"### Predicted Claim Cost: ${prediction:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 8: Data Export
    with tab8:
        st.header("Data Export")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.write("Select the export format and click the button to download the filtered dataset.")
        export_formats = ["CSV"]
        if XLSXWRITER_AVAILABLE:
            export_formats.append("Excel")
        if FPDF_AVAILABLE:
            export_formats.append("PDF")
        export_format = st.selectbox("Select Export Format", export_formats)

        if st.button("Export Data"):
            try:
                if export_format == "CSV":
                    buffer = io.BytesIO()
                    st.session_state.filtered_data.to_csv(buffer, index=False)
                    buffer.seek(0)
                    st.download_button(
                        label="Download CSV File",
                        data=buffer,
                        file_name="filtered_data_export.csv",
                        mime="text/csv",
                        key="export_csv_button",
                        use_container_width=True
                    )
                elif export_format == "Excel" and XLSXWRITER_AVAILABLE:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        st.session_state.filtered_data.to_excel(writer, index=False, sheet_name="Sheet1")
                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel File",
                        data=buffer,
                        file_name="filtered_data_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="export_excel_button",
                        use_container_width=True
                    )
                elif export_format == "PDF" and FPDF_AVAILABLE:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for i, row in st.session_state.filtered_data.head(10).iterrows():
                        pdf.cell(200, 10, txt=str(row.to_dict()), ln=True)
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    st.download_button(
                        label="Download PDF File",
                        data=pdf_output,
                        file_name="filtered_data_export.pdf",
                        mime="application/pdf",
                        key="export_pdf_button",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error exporting data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
