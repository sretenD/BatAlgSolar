import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import BytesIO
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="🔋 Battery Sim",
    layout="wide",
    initial_sidebar_state="expanded",
)

image_path = "https://www.eimv.si/img/EIMV_logo_png.png"
col1, col2, col3 = st.columns([1, 2, 1])
col1.empty()
col3.empty()
col2.image(image_path, width=300)

st.title("🔋 Battery Sim")

# Sidebar for inputs
st.sidebar.header("1. Upload Power Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file with power data", type=["csv"])

st.sidebar.header("2. Battery Parameters")
battery_power = st.sidebar.slider("Battery Power Capacity (MW)", 0.1, 20.0, 1.0, step=0.1)
battery_capacity = st.sidebar.slider("Battery Energy Capacity (MWh)", 1, 500, 1, step=1)
battery_efficiency = st.sidebar.slider("Battery Efficiency (%)", 80, 100, 93, step=1)

st.sidebar.header("3. Financial Parameters")
battery_cost_per_mwh = st.sidebar.number_input("Battery Cost (€ per MWh)", value=200000)
battery_cost_per_mw = st.sidebar.number_input("Battery Power Cost (€ per MW)", value=500000)
operational_cost_per_mwh = st.sidebar.number_input("Operational Cost (€ per MWh)", value=6500)
savings_per_mwh = st.sidebar.number_input("Savings from Unserved Energy (€ per MWh)", value=110)

DEFAULT_DATA_POINTS = 1000
DEFAULT_POWER = np.random.normal(loc=2.5, scale=1.0, size=DEFAULT_DATA_POINTS)
DEFAULT_TIME = pd.date_range(start='2023-01-01', periods=DEFAULT_DATA_POINTS, freq='15min')

# Load data
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, sep='\t', parse_dates=['Timestamp'])
        if data.shape[1] < 2:
            st.error("The CSV file must contain at least two columns: Timestamp and Power(kW).")
            st.stop()
        if set(data.columns) != set(['Timestamp', 'Power(kW)']):
            st.error("The CSV file must have columns named 'Timestamp' and 'Power(kW)'.")
            st.stop()
        # data = data.sort_values('Timestamp',ascending=True)
        data.reset_index(drop=True, inplace=True)
        data['Power(MW)'] = data['Power(kW)'] / 1e3
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    st.info("Using default synthetic data. Upload your own data for more accurate results.")
    data = pd.DataFrame({
        'Timestamp': DEFAULT_TIME,
        'Power(kW)': DEFAULT_POWER * 1000  # Convert MW to kW
    })
    data['Power(MW)'] = data['Power(kW)'] / 1e3

st.header("📊 Power Data Preview")
st.write(data.head())

st.header("⚙️ Simulation Parameters")

Pp = battery_power  # The threshold to not exceed

# Calculate Adjusted Power:
# Adjusted_Power(MW) = Power(MW) - Pp
# Positive => deficit (need discharge)
# Negative => surplus (need charge)
data['Adjusted_Power(MW)'] = data['Power(MW)'] - 0

st.subheader("Consumption Diagram vs Threshold")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Timestamp'], y=data['Power(MW)'],
                   mode='lines', name='Consumption Diagram'))
fig1.add_trace(go.Scatter(x=data['Timestamp'], y=data['Adjusted_Power(MW)'],
                   mode='lines', name='Adjusted Power (Deficit/Surplus)'))
fig1.add_trace(go.Scatter(x=data['Timestamp'], y=[0]*len(data),
                   mode='lines', name='Zero Line', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=data['Timestamp'], y=[Pp]*len(data),
                   mode='lines', name='Power Threshold', line=dict(dash='dot', color='red')))
fig1.add_trace(go.Scatter(x=data['Timestamp'], y=[-Pp]*len(data),
                   mode='lines', name='Power Threshold', line=dict(dash='dot', color='red')))
fig1.update_layout(title="Power vs Threshold",
                   xaxis_title="Time",
                   yaxis_title="Power (MW)")
st.plotly_chart(fig1, use_container_width=True)

def simulate_battery(agc, moc_shr, C, eff, int_interval=4):
    n = len(agc)
    ba = np.zeros(n+1)  # Battery SoC in MWh
    ba[0] = 0.0  # Start empty
    served_energy = np.zeros(n)
    unserved_energy = np.zeros(n)
    dt = 1.0 / int_interval


    for i in range(1, n+1):
            power_needed = min(agc[i-1], moc_shr)  # Limit discharge to actual load

            if power_needed > 0:
                # DEFICIT scenario: Need to DISCHARGE battery
                deficit = power_needed * dt
                # Max we can discharge this interval
                max_discharge = min(moc_shr * dt, ba[i-1], deficit)  # Added deficit limit
                served_energy[i-1] = max_discharge
                unserved_energy[i-1] = max(0, deficit - max_discharge)
                ba[i] = ba[i-1] - max_discharge

            elif power_needed < 0:
                # SURPLUS scenario: Need to CHARGE battery
                surplus = abs(power_needed) * dt
                available_capacity = C - ba[i-1]
                stored_energy = min(surplus * eff, available_capacity)
                ba[i] = ba[i-1] + stored_energy
                unserved_energy[i-1] = max(0, surplus - (stored_energy / eff))
                served_energy[i-1] = -stored_energy

            else:
                ba[i] = ba[i-1]
                served_energy[i-1] = 0
                unserved_energy[i-1] = 0

    soc = (ba[1:] / C) * 100
    total_unserved_positive = np.sum(unserved_energy[unserved_energy > 0])
    total_unserved_negative = np.sum(unserved_energy[unserved_energy < 0])
    total_served_positive = np.sum(served_energy[served_energy > 0])
    total_served_negative = np.sum(served_energy[served_energy < 0])
    percentage_unserved = (np.count_nonzero(unserved_energy) / len(unserved_energy)) * 100

    results = {
        'SoC': soc,
        'Served_Energy': served_energy,
        'Unserved_Energy': unserved_energy,
        'Total_Unserved_Positive': total_unserved_positive,
        'Total_Unserved_Negative': total_unserved_negative,
        'Total_Served_Positive': total_served_positive,
        'Total_Served_Negative': total_served_negative,
        'Percentage_Unserved': percentage_unserved
    }

    return results

st.header("🔄 Run Simulation")

if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        try:
            agc = data['Adjusted_Power(MW)'].values
            simulation_results = simulate_battery(
                agc=agc,
                moc_shr=battery_power,
                C=battery_capacity,
                eff=battery_efficiency / 100,
                int_interval=4  # 15-minute intervals
            )
        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")
            st.stop()

    st.success("Simulation completed!")

    # Display SoC over time
    # st.subheader("🔋 Battery State of Charge (SoC)")
    # fig_soc = go.Figure()
    # fig_soc.add_trace(go.Scatter(x=data['Timestamp'], y=simulation_results['SoC'],
    #                              mode='lines', name='SoC (%)'))
    # fig_soc.update_layout(title="Battery State of Charge Over Time",
    #                       xaxis_title="Time",
    #                       yaxis_title="SoC (%)")
    # st.plotly_chart(fig_soc, use_container_width=True)

    # # Display Served and Unserved Energy
    # st.subheader("📈 Battery Energy")
    # fig_energy = go.Figure()
    # fig_energy.add_trace(go.Scatter(x=data['Timestamp'], y=simulation_results['Served_Energy'],
    #                                 mode='lines', name='Served Energy (MWh)', line=dict(color='green')))
    # fig_energy.add_trace(go.Scatter(x=data['Timestamp'], y=simulation_results['Unserved_Energy'],
    #                                 mode='lines', name='Unserved Energy (MWh)', line=dict(color='red')))
    # fig_energy.update_layout(title="Battery Energy Over Time",
    #                          xaxis_title="Time",
    #                          yaxis_title="Energy (MWh)")
    # st.plotly_chart(fig_energy, use_container_width=True)

    # Replace the existing energy visualization with this power-focused visualization
    # st.subheader("📈 Battery Power")
    # fig_power = go.Figure()
    # fig_power.add_trace(go.Bar(x=data['Timestamp'], 
    #                         y=simulation_results['Served_Energy'] * 4,  # Convert MWh to MW (4 intervals per hour)
    #                         name='Battery Power (MW)', 
    #                         marker_color='green'))
    # fig_power.add_trace(go.Bar(x=data['Timestamp'], 
    #                         y=simulation_results['Unserved_Energy'] * 4,  # Convert MWh to MW (4 intervals per hour)
    #                         name='Unserved Power (MW)', 
    #                         marker_color='red'))
    # fig_power.update_layout(title="Battery Power Over Time",
    #                     xaxis_title="Time",
    #                     yaxis_title="Power (MW)",
    #                     barmode='overlay')
    # st.plotly_chart(fig_power, use_container_width=True)

    # Replace the existing separate plots with this combined subplot
    st.subheader("🔋 Battery Performance")
    fig_combined = make_subplots(rows=2, cols=1, 
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=("Battery State of Charge (SoC)", "Battery Power"))

    # Add SoC trace to first subplot
    fig_combined.add_trace(
        go.Scatter(x=data['Timestamp'], 
                y=simulation_results['SoC'],
                mode='lines', 
                name='SoC (%)'),
        row=1, col=1
    )

    # Add Battery Power traces to second subplot
    fig_combined.add_trace(
        go.Bar(x=data['Timestamp'], 
            y=simulation_results['Served_Energy'] * 4,
            name='Battery Power (MW)', 
            marker_color='green'),
        row=2, col=1
    )

    fig_combined.add_trace(
        go.Bar(x=data['Timestamp'], 
            y=simulation_results['Unserved_Energy'] * 4,
            name='Unserved Power (MW)', 
            marker_color='red'),
        row=2, col=1
    )

    # Update layout
    fig_combined.update_layout(
        height=800,  # Increase height to accommodate both plots
        showlegend=True,
        barmode='overlay'
    )

    # Update y-axes labels
    fig_combined.update_yaxes(title_text="SoC (%)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Power (MW)", row=2, col=1)

    st.plotly_chart(fig_combined, use_container_width=True)


    # Summary Metrics
    st.subheader("📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Served Energy (+)", f"{simulation_results['Total_Served_Positive']:.2f} MWh")
    col1.metric("Total Served Energy (-)", f"{simulation_results['Total_Served_Negative']:.2f} MWh")
    col2.metric("Total Unserved Energy (+)", f"{simulation_results['Total_Unserved_Positive']:.2f} MWh")
    col2.metric("Total Unserved Energy (-)", f"{simulation_results['Total_Unserved_Negative']:.2f} MWh")
    col3.metric("Percentage of Time Unserved", f"{simulation_results['Percentage_Unserved']:.2f} %")

    # Financial Analysis
    st.header("💰 Financial Analysis")

    capex = (battery_cost_per_mwh * battery_capacity) + (battery_cost_per_mw * battery_power)
    opex = operational_cost_per_mwh * (simulation_results['Total_Served_Positive'] + simulation_results['Total_Served_Negative'])
    savings = savings_per_mwh * (simulation_results['Total_Unserved_Positive'] + abs(simulation_results['Total_Unserved_Negative']))
    net_savings = savings - opex
    payback_period = capex / net_savings if net_savings > 0 else np.nan
    roi = (net_savings / capex) * 100 if capex > 0 else np.nan

    col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)
    col_f1.metric("Capital Expenditure (CapEx)", f"€{capex:,.2f}")
    col_f2.metric("Operational Expenditure (OpEx)", f"€{opex:,.2f}")
    col_f3.metric("Savings from Unserved Energy", f"€{savings:,.2f}")
    col_f4.metric("Net Savings", f"€{net_savings:,.2f}")
    col_f5.metric("Payback Period (Years)", f"{payback_period:.2f}" if not np.isnan(payback_period) else "N/A")
    col_f6.metric("Return on Investment (ROI)", f"{roi:.2f} %" if not np.isnan(roi) else "N/A")

    st.subheader("📋 Financial Summary")
    financial_summary = pd.DataFrame({
        'Metric': ['Capital Expenditure (CapEx)', 
                   'Operational Expenditure (OpEx)', 
                   'Savings from Unserved Energy', 
                   'Net Savings', 
                   'Payback Period (Years)', 
                   'Return on Investment (ROI)'],
        'Value': [f"€{capex:,.2f}",
                  f"€{opex:,.2f}",
                  f"€{savings:,.2f}",
                  f"€{net_savings:,.2f}",
                  f"{payback_period:.2f}" if not np.isnan(payback_period) else "N/A",
                  f"{roi:.2f} %" if not np.isnan(roi) else "N/A"]
    })
    st.table(financial_summary)

    # Download Simulation Results
    st.header("📥 Download Results")

    results_df = data.copy()
    results_df['Served_Energy(MWh)'] = simulation_results['Served_Energy']
    results_df['Unserved_Energy(MWh)'] = simulation_results['Unserved_Energy']
    results_df['SoC(%)'] = simulation_results['SoC']

    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Simulation Results')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(results_df)

    st.download_button(
        label="📥 Download Simulation Results as Excel",
        data=excel_data,
        file_name='simulation_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Prepare financial summary for download
    financial_excel = BytesIO()
    with pd.ExcelWriter(financial_excel, engine='xlsxwriter') as writer:
        financial_summary.to_excel(writer, index=False, sheet_name='Financial Summary')
    financial_excel.seek(0)
    st.download_button(
        label="📥 Download Financial Summary as Excel",
        data=financial_excel.getvalue(),
        file_name='financial_summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.markdown("""
---
*Battery Power Sim App*
""")
