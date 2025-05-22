import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="EV Impact Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2196F3;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to create database connection
def get_db_connection():
    """Create a database connection to Azure SQL DB"""
    try:
        # Get the connection string from environment variable
        connection_string = os.getenv('AZURE_SQL_CONNECTIONSTRING')
        if not connection_string:
            st.error("Azure SQL connection string not found in environment variables.")
            return None
            
        # Create SQLAlchemy engine
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to load data from Azure SQL Database
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data_from_azure():
    """Load data from Azure SQL Database"""
    try:
        engine = get_db_connection()
        if engine is None:
            return None
            
        data = {}
        
        # Load dimension tables
        data["dim_time"] = pd.read_sql_query("SELECT * FROM dim_time", engine)
        data["dim_suburb"] = pd.read_sql_query("SELECT * FROM dim_suburb", engine)
        data["dim_vehicle_type"] = pd.read_sql_query("SELECT * FROM dim_vehicle_type", engine)
        data["dim_fuel_type"] = pd.read_sql_query("SELECT * FROM dim_fuel_type", engine)
        
        # Load fact tables
        data["fact_ev_impact"] = pd.read_sql_query("SELECT * FROM fact_ev_impact", engine)
        data["fact_energy_pollution"] = pd.read_sql_query("SELECT * FROM fact_energy_pollution", engine)
        
        return data
    except Exception as e:
        st.error(f"Error loading data from Azure SQL Database: {e}")
        return None

# Function to join dimension and fact tables
def join_tables(data):
    """Join dimension and fact tables for analysis"""
    try:
        # Join suburb dimension with EV impact fact
        ev_impact_with_suburb = pd.merge(
            data["fact_ev_impact"],
            data["dim_suburb"],
            left_on="id_suburb",  # Using the correct join column from your schema
            right_on="id_suburb",
            how="left"
        )
        
        # Join suburb dimension with energy pollution fact
        energy_pollution_with_suburb = pd.merge(
            data["fact_energy_pollution"],
            data["dim_suburb"],
            left_on="id_suburb",  # Using the correct join column from your schema
            right_on="id_suburb",
            how="left"
        )
        
        return ev_impact_with_suburb, energy_pollution_with_suburb
    except Exception as e:
        st.error(f"Error joining tables: {e}")
        return None, None

# Main function
def main():
    st.markdown('<p class="main-header">EV Impact & Environmental Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.write("""
    This dashboard analyzes the relationship between electric vehicle adoption, 
    electricity consumption, and pollution levels across different suburbs.
    """)
    
    # Add a connection status indicator
    with st.sidebar:
        st.subheader("Database Connection")
        if get_db_connection() is not None:
            st.success("✅ Connected to Azure SQL Database")
        else:
            st.error("❌ Not connected to Azure SQL Database")
    
    # Load data from Azure SQL Database
    data = load_data_from_azure()
    
    if data is None:
        st.error("Failed to load data from Azure SQL Database. Please check your connection settings.")
        return
    
    # Join tables
    ev_impact_with_suburb, energy_pollution_with_suburb = join_tables(data)
    
    if ev_impact_with_suburb is None or energy_pollution_with_suburb is None:
        st.error("Failed to join tables. Please check your data schema.")
        return
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "EV Adoption Overview", 
        "Environmental Impact", 
        "Suburb Analysis",
        "Data Explorer"
    ])
    
    with tab1:
        st.markdown('<p class="sub-header">EV Adoption Metrics</p>', unsafe_allow_html=True)
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total EVs", f"{int(ev_impact_with_suburb['TOTAL_EVS'].sum())}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Battery EVs (BEV)", f"{int(ev_impact_with_suburb['BEV_COUNT'].sum())}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Plug-in Hybrid EVs (PHEV)", f"{int(ev_impact_with_suburb['PHEV_COUNT'].sum())}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            bev_percentage = (ev_impact_with_suburb['BEV_COUNT'].sum() / 
                            ev_impact_with_suburb['TOTAL_EVS'].sum()) * 100
            st.metric("BEV Percentage", f"{bev_percentage:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # EV Distribution by Suburb
        st.subheader("EV Distribution by Suburb")
        
        # Sort for better visualization
        ev_by_suburb = ev_impact_with_suburb.sort_values(by="TOTAL_EVS", ascending=False)
        
        fig = px.bar(
            ev_by_suburb,
            x="SUBURB_NAME",
            y=["BEV_COUNT", "PHEV_COUNT"],
            title="EV Distribution Across Suburbs",
            labels={"value": "Number of Vehicles", "SUBURB_NAME": "Suburb", "variable": "EV Type"},
            color_discrete_map={"BEV_COUNT": "#2196F3", "PHEV_COUNT": "#4CAF50"},
            barmode="stack"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # EV Price and Range Analysis
        st.subheader("EV Price and Range Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                ev_impact_with_suburb,
                x="AVG_PRICE",
                y="TOTAL_EVS",
                size="TOTAL_EVS",
                color="SUBURB_NAME",
                hover_name="SUBURB_NAME",
                title="EV Adoption vs Average Price",
                labels={"AVG_PRICE": "Average EV Price ($)", "TOTAL_EVS": "Number of EVs"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(
                ev_impact_with_suburb,
                x="AVG_RANGE_KM",
                y="TOTAL_EVS",
                size="TOTAL_EVS",
                color="SUBURB_NAME",
                hover_name="SUBURB_NAME",
                title="EV Adoption vs Average Range",
                labels={"AVG_RANGE_KM": "Average Range (km)", "TOTAL_EVS": "Number of EVs"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<p class="sub-header">Environmental Impact Analysis</p>', unsafe_allow_html=True)
        
        # Filter for 2023 data for current analysis
        energy_2023 = energy_pollution_with_suburb[energy_pollution_with_suburb['YEAR'] == 2023]
        
        # Energy consumption vs NO2 levels
        st.subheader("Energy Consumption vs NO2 Pollution (2023)")
        
        # Create scatter plot without size parameter
        fig = px.scatter(
            energy_2023,
            x="ENERGY_CONSUMPTION", 
            y="NO2_LEVEL",
            color="SUBURB_NAME",
            hover_name="SUBURB_NAME",
            title="Energy Consumption vs NO2 Levels by Suburb",
            labels={
                "ENERGY_CONSUMPTION": "Energy Consumption (kWh)",
                "NO2_LEVEL": "NO2 Levels (μg/m³)"
            },
            text="SUBURB_NAME"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Yearly comparison
        st.subheader("Year-over-Year Environmental Changes")
        
        # Prepare data
        years = energy_pollution_with_suburb['YEAR'].unique()
        suburbs = energy_pollution_with_suburb['SUBURB_NAME'].unique()
        
        # Let user select suburb for detailed view
        # Make sure suburbs is not empty
        if len(suburbs) > 0:
            selected_suburb = st.selectbox("Select Suburb for Detailed Analysis", suburbs)
            # Filter data for selected suburb
            suburb_data = energy_pollution_with_suburb[energy_pollution_with_suburb['SUBURB_NAME'] == selected_suburb]
        else:
            st.error("No suburb data available.")
            suburb_data = pd.DataFrame()
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Only add traces if we have data
        if not suburb_data.empty:
            # Add energy consumption trace
            fig.add_trace(
                go.Bar(
                    x=suburb_data['YEAR'],
                    y=suburb_data['ENERGY_CONSUMPTION'],
                    name="Energy Consumption",
                    marker_color='#2196F3'
                ),
                secondary_y=False
            )
            
            # Add NO2 level trace
            fig.add_trace(
                go.Scatter(
                    x=suburb_data['YEAR'],
                    y=suburb_data['NO2_LEVEL'],
                    name="NO2 Level",
                    marker_color='#FF5722',
                    mode='lines+markers'
                ),
                secondary_y=True
            )
        
        # Set titles
        fig.update_layout(
            title=f"Energy Consumption and NO2 Levels in {selected_suburb} (2022-2023)",
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Energy Consumption (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="NO2 Level (μg/m³)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Change percentage analysis
        change_data = energy_2023.sort_values(by="NO2_CHANGE_PCT")
        
        fig = px.bar(
            change_data,
            x="SUBURB_NAME",
            y="NO2_CHANGE_PCT",
            color="NO2_CHANGE_PCT",
            color_continuous_scale=px.colors.diverging.RdYlGn_r,  # Red for increase, green for decrease
            title="NO2 Pollution Change (%) from 2022 to 2023",
            labels={"NO2_CHANGE_PCT": "NO2 Change (%)", "SUBURB_NAME": "Suburb"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<p class="sub-header">Suburb Comparison</p>', unsafe_allow_html=True)
        
        # Join EV impact and energy pollution for combined analysis
        # First ensure both dataframes have id_suburb and at least some data
        if ('id_suburb' in ev_impact_with_suburb.columns and 
            'id_suburb' in energy_2023.columns and 
            energy_2023.shape[0] > 0):
            
            combined_data = pd.merge(
                ev_impact_with_suburb,
                energy_2023[['id_suburb', 'ENERGY_CONSUMPTION', 'NO2_LEVEL', 'NO2_CHANGE_PCT']],
                on="id_suburb",
                how="left"
            )
        else:
            # Create an empty dataframe with the needed columns
            combined_data = pd.DataFrame(columns=[
                'id_suburb', 'SUBURB_NAME', 'TOTAL_EVS', 'BEV_COUNT', 'PHEV_COUNT',
                'AVG_RANGE_KM', 'AVG_PRICE', 'EV_ADOPTION_SCORE', 'ENERGY_CONSUMPTION',
                'NO2_LEVEL', 'NO2_CHANGE_PCT'
            ])
        
        # Calculate EV adoption score and normalize it if we have data
        if not combined_data.empty and 'EV_ADOPTION_SCORE' in combined_data.columns:
            # Check if we have more than one value and they're not all the same
            if (combined_data['EV_ADOPTION_SCORE'].nunique() > 1):
                combined_data['EV_ADOPTION_NORMALIZED'] = (combined_data['EV_ADOPTION_SCORE'] - 
                                                      combined_data['EV_ADOPTION_SCORE'].min()) / \
                                                     (combined_data['EV_ADOPTION_SCORE'].max() - 
                                                      combined_data['EV_ADOPTION_SCORE'].min()) * 100
            else:
                # If all values are the same, set to 50
                combined_data['EV_ADOPTION_NORMALIZED'] = 50
        
        # Create radar chart for suburb comparison
        st.subheader("Suburb Comparison - Key Metrics")
        
        # Select suburbs to compare
        selected_suburbs = st.multiselect(
            "Select Suburbs to Compare",
            options=combined_data['SUBURB_NAME'].unique(),
            default=combined_data['SUBURB_NAME'].unique()[:3]  # Default to first 3 suburbs
        )
        
        if selected_suburbs:
            # Filter data for selected suburbs
            radar_data = combined_data[combined_data['SUBURB_NAME'].isin(selected_suburbs)]
            
            # Create radar chart
            fig = go.Figure()
            
            # Metrics to include in radar chart
            metrics = [
                "TOTAL_EVS", 
                "AVG_RANGE_KM", 
                "AVG_PRICE", 
                "ENERGY_CONSUMPTION", 
                "NO2_LEVEL"
            ]
            
            # Normalize metrics for better visualization
            radar_normalized = pd.DataFrame()
            for metric in metrics:
                max_val = radar_data[metric].max()
                min_val = radar_data[metric].min()
                if max_val == min_val:
                    radar_normalized[metric] = 50  # Set to middle value if all are the same
                else:
                    if metric in ["NO2_LEVEL", "AVG_PRICE"]:  # Lower is better
                        radar_normalized[metric] = 100 - ((radar_data[metric] - min_val) / (max_val - min_val) * 100)
                    else:  # Higher is better
                        radar_normalized[metric] = ((radar_data[metric] - min_val) / (max_val - min_val) * 100)
            
            radar_normalized['SUBURB_NAME'] = radar_data['SUBURB_NAME'].values
            
            # Better labels for radar chart
            radar_labels = {
                "TOTAL_EVS": "EV Adoption",
                "AVG_RANGE_KM": "EV Range",
                "AVG_PRICE": "Price Affordability",
                "ENERGY_CONSUMPTION": "Energy Usage",
                "NO2_LEVEL": "Air Quality"
            }
            
            # Add traces for each suburb
            for suburb in selected_suburbs:
                suburb_data = radar_normalized[radar_normalized['SUBURB_NAME'] == suburb]
                
                if not suburb_data.empty:
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            suburb_data['TOTAL_EVS'].values[0],
                            suburb_data['AVG_RANGE_KM'].values[0],
                            suburb_data['AVG_PRICE'].values[0],
                            suburb_data['ENERGY_CONSUMPTION'].values[0],
                            suburb_data['NO2_LEVEL'].values[0]
                        ],
                        theta=list(radar_labels.values()),
                        fill='toself',
                        name=suburb
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Suburb Comparison (Higher is Better for All Metrics)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # EV Adoption vs NO2 Reduction
            st.subheader("EV Adoption vs NO2 Reduction")
            
            fig = px.scatter(
                combined_data,
                x="TOTAL_EVS",
                y="NO2_CHANGE_PCT",
                color="SUBURB_NAME",
                size="ENERGY_CONSUMPTION",
                hover_name="SUBURB_NAME",
                title="Relationship between EV Adoption and NO2 Change",
                labels={
                    "TOTAL_EVS": "Total EVs",
                    "NO2_CHANGE_PCT": "NO2 Change (%)",
                    "ENERGY_CONSUMPTION": "Energy Consumption"
                },
                text="SUBURB_NAME"
            )
            
            # Add a horizontal line at y=0 to show the boundary between increase and decrease
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Add explanatory text
            fig.add_annotation(
                x=combined_data['TOTAL_EVS'].max() * 0.8,
                y=5,
                text="NO2 Increase",
                showarrow=False,
                font=dict(color="red")
            )
            
            fig.add_annotation(
                x=combined_data['TOTAL_EVS'].max() * 0.8,
                y=-5,
                text="NO2 Decrease",
                showarrow=False,
                font=dict(color="green")
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<p class="sub-header">Data Explorer</p>', unsafe_allow_html=True)
        
        # Display raw data tables from Azure SQL Database
        st.subheader("Raw Data Tables")
        
        # Select which table to view
        table_option = st.selectbox(
            "Select Table to View",
            options=[
                "EV Impact Fact Table",
                "Energy Pollution Fact Table",
                "Suburb Dimension Table",
                "Vehicle Type Dimension Table",
                "Fuel Type Dimension Table",
                "Time Dimension Table"
            ]
        )
        
        # Display selected table
        if table_option == "EV Impact Fact Table":
            st.dataframe(data["fact_ev_impact"])
        elif table_option == "Energy Pollution Fact Table":
            st.dataframe(data["fact_energy_pollution"])
        elif table_option == "Suburb Dimension Table":
            st.dataframe(data["dim_suburb"])
        elif table_option == "Vehicle Type Dimension Table":
            st.dataframe(data["dim_vehicle_type"])
        elif table_option == "Fuel Type Dimension Table":
            st.dataframe(data["dim_fuel_type"])
        elif table_option == "Time Dimension Table":
            st.dataframe(data["dim_time"])
            
        # Custom query option for advanced users
        st.subheader("Custom SQL Query")
        
        # SQL query input
        custom_query = st.text_area(
            "Enter SQL Query",
            "SELECT TOP 10 * FROM fact_ev_impact",
            height=100
        )
        
        if st.button("Run Query"):
            try:
                engine = get_db_connection()
                if engine is not None:
                    # Execute the query
                    query_result = pd.read_sql_query(custom_query, engine)
                    st.dataframe(query_result)
                else:
                    st.error("No database connection available.")
            except Exception as e:
                st.error(f"Error executing query: {e}")
        
        # Custom visualization section
        st.subheader("Custom Analysis")
        
        st.write("""
        Select dimensions and metrics to create a custom visualization.
        """)
        
        # Let user select table to analyze
        analysis_table = st.radio(
            "Select primary table for analysis",
            options=["EV Impact", "Energy & Pollution"]
        )
        
        if analysis_table == "EV Impact":
            df_to_analyze = ev_impact_with_suburb
            x_options = ["SUBURB_NAME", "BEV_COUNT", "PHEV_COUNT", "AVG_RANGE_KM", "AVG_PRICE"]
            y_options = ["TOTAL_EVS", "BEV_COUNT", "PHEV_COUNT", "AVG_RANGE_KM", "AVG_PRICE"]
            if "EV_ADOPTION_SCORE" in df_to_analyze.columns:
                y_options.append("EV_ADOPTION_SCORE")
            color_options = ["SUBURB_NAME", "BEV_COUNT", "PHEV_COUNT", "AVG_RANGE_KM", "AVG_PRICE"]
        else:
            df_to_analyze = energy_pollution_with_suburb
            x_options = ["SUBURB_NAME", "YEAR", "ENERGY_CONSUMPTION", "NO2_LEVEL"]
            y_options = ["ENERGY_CONSUMPTION"]
            if "ENERGY_CHANGE_PCT" in df_to_analyze.columns:
                y_options.append("ENERGY_CHANGE_PCT")
            y_options.extend(["NO2_LEVEL"])
            if "NO2_CHANGE" in df_to_analyze.columns:
                y_options.append("NO2_CHANGE")
            if "NO2_CHANGE_PCT" in df_to_analyze.columns:
                y_options.append("NO2_CHANGE_PCT")
            color_options = ["SUBURB_NAME", "YEAR"]
            if "NO2_CHANGE_PCT" in df_to_analyze.columns:
                color_options.append("NO2_CHANGE_PCT")
            if "ENERGY_CHANGE_PCT" in df_to_analyze.columns:
                color_options.append("ENERGY_CHANGE_PCT")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-Axis", options=x_options)
        
        with col2:
            y_axis = st.selectbox("Y-Axis", options=y_options)
        
        with col3:
            color_by = st.selectbox("Color By", options=color_options)
        
        # Chart type selection
        chart_type = st.radio("Chart Type", options=["Bar", "Line", "Scatter"])
        
        # Create the chart
        if not df_to_analyze.empty:
            # Make sure the columns exist
            if x_axis in df_to_analyze.columns and y_axis in df_to_analyze.columns and color_by in df_to_analyze.columns:
                if chart_type == "Bar":
                    fig = px.bar(
                        df_to_analyze,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=f"{y_axis} by {x_axis}",
                        labels={x_axis: x_axis.replace("_", " "), y_axis: y_axis.replace("_", " ")}
                    )
                elif chart_type == "Line":
                    fig = px.line(
                        df_to_analyze,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=f"{y_axis} by {x_axis}",
                        labels={x_axis: x_axis.replace("_", " "), y_axis: y_axis.replace("_", " ")}
                    )
                else:  # Scatter
                    fig = px.scatter(
                        df_to_analyze,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=f"{y_axis} vs {x_axis}",
                        labels={x_axis: x_axis.replace("_", " "), y_axis: y_axis.replace("_", " ")}
                    )
            else:
                st.error(f"One or more selected columns not found in the data.")
                fig = go.Figure()
        else:
            st.error("No data available for visualization.")
            fig = go.Figure()
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()