import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(‘ignore’)

# Page configuration

st.set_page_config(
page_title=“Expense Dashboard”,
page_icon=“💰”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# Custom CSS for better styling

st.markdown(”””

<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .filter-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>

“””, unsafe_allow_html=True)

@st.cache_data
def load_data():
“””
Load and preprocess the data from ‘03.my_data.csv’.
If the file is not found or its structure is unexpected, sample data is created.
“””
try:
df = pd.read_csv(‘03.my_data.csv’)

```
    # Define the required column names as they appear in the CSV (case-sensitive initially)
    required_columns_csv_case = ['Type', 'Amount', 'Subtype', 'Description', 'Timestamp'] 
    
    # Convert DataFrame column NAMES to lowercase for consistent internal handling
    df.columns = df.columns.str.lower()
    
    # Define the expected lowercase column names for internal use
    required_columns_lower = [col.lower() for col in required_columns_csv_case]

    # Check if all expected lowercase columns are present in the DataFrame
    if not all(col in df.columns for col in required_columns_lower):
        st.warning("CSV structure not as expected or some required columns are missing. Creating sample data for demonstration.")
        return create_sample_data()

    # Convert values in the 'type' column to lowercase to match filtering logic
    if 'type' in df.columns:
        df['type'] = df['type'].astype(str).str.lower()

    # Ensure 'timestamp' column exists and is converted to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        st.error("Missing 'Timestamp' column. Cannot proceed without date information. Creating sample data.")
        return create_sample_data()

    # Ensure 'amount' column exists and is numeric
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        st.error("Missing 'Amount' column. Cannot proceed without amount information. Creating sample data.")
        return create_sample_data()
    
    # Drop rows where 'timestamp' or 'amount' is NaN after conversion
    df.dropna(subset=['timestamp', 'amount'], inplace=True)
    
    # Create year-month column for grouping
    if 'timestamp' in df.columns:
        df['year_month'] = df['timestamp'].dt.to_period('M')
    else:
        df['year_month'] = None

    return df
except FileNotFoundError:
    st.error("Data file '03.my_data.csv' not found. Creating sample data for demonstration.")
    return create_sample_data()
except Exception as e:
    st.error(f"An error occurred while loading or processing data: {e}. Creating sample data for demonstration.")
    return create_sample_data()
```

def create_sample_data():
“””
Creates a sample DataFrame for demonstration purposes
if the actual CSV file is not found or malformed.
“””
np.random.seed(42)
dates = pd.date_range(start=‘2023-01-01’, end=‘2024-12-31’, freq=‘D’)

```
data = []
for current_date in dates:
    # Income entries (10% chance per day)
    if np.random.random() < 0.1:  
        data.append({
            'timestamp': current_date,
            'type': 'income',
            'subtype': np.random.choice(['salary', 'freelance', 'investment', 'other']),
            'description': np.random.choice(['Monthly salary', 'Project payment', 'Dividend', 'Bonus']),
            'amount': np.random.randint(1000000, 15000000)   # VND
        })
    
    # Outcome entries (30% chance per day)
    if np.random.random() < 0.3:  
        data.append({
            'timestamp': current_date,
            'type': 'outcome',
            'subtype': np.random.choice(['food', 'transport', 'utilities', 'entertainment', 'shopping']),
            'description': np.random.choice(['Groceries', 'Gas', 'Electric bill', 'Movie', 'Clothes']),
            'amount': -np.random.randint(50000, 500000)   # Negative for expenses
        })
    
    # Savings entries (5% chance per day)
    if np.random.random() < 0.05:  
        data.append({
            'timestamp': current_date,
            'type': 'saving',
            'subtype': np.random.choice(['emergency', 'investment', 'retirement']),
            'description': np.random.choice(['Emergency fund', 'Stock investment', 'Pension']),
            'amount': np.random.randint(500000, 2000000)
        })

df = pd.DataFrame(data)
if not df.empty:
    df['year_month'] = df['timestamp'].dt.to_period('M')
return df
```

def format_vnd(amount):
“””
Formats a numerical amount into VND currency string, using ‘M’ for millions.
“””
if abs(amount) >= 1_000_000:
return f”{amount / 1_000_000:,.2f} M VND”
return f”{amount:,.0f} VND”

def create_filters(df, key_prefix=””):
“””
Creates filter widgets (Type, Date Range, Subtype) in the Streamlit app.
“””
st.markdown(’<div class="filter-container">’, unsafe_allow_html=True)
st.subheader(“📊 Filters”)

```
col1, col2, col3 = st.columns(3)

with col1:
    type_options = df['type'].unique() if 'type' in df.columns and not df['type'].empty else []
    type_filter = st.multiselect(
        "Type",
        options=list(type_options),
        default=list(type_options),
        key=f"{key_prefix}_type"
    )

with col2:
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']) and not df['timestamp'].empty:
        min_date = df['timestamp'].min().date() 
        max_date = df['timestamp'].max().date()
        date_range = st.date_input(
            "Timestamp Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_date"
        )
    else:
        st.info("No valid 'Timestamp' column found for date filtering.")
        date_range = (None, None)
        
with col3:
    subtype_options = []
    if 'subtype' in df.columns and 'type' in df.columns and not df.empty:
        if type_filter:
            filtered_by_type_df = df[df['type'].isin(type_filter)]
            if not filtered_by_type_df.empty:
                subtype_options = filtered_by_type_df['subtype'].unique()
        else:
            subtype_options = df['subtype'].unique()
        
    subtype_filter = st.multiselect(
        "Subtype",
        options=list(subtype_options),
        default=list(subtype_options),
        key=f"{key_prefix}_subtype"
    )

st.markdown('</div>', unsafe_allow_html=True)

return type_filter, date_range, subtype_filter
```

def apply_filters(df, type_filter, date_range, subtype_filter):
“””
Applies the selected filters to the DataFrame.
“””
filtered_df = df.copy()

```
if not filtered_df.empty:
    if type_filter and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    
    if len(date_range) == 2 and date_range[0] is not None and 'timestamp' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) & 
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    if subtype_filter and 'subtype' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['subtype'].isin(subtype_filter)]

return filtered_df
```

def create_metrics(df, overall_df=None):
“””
Creates and displays metric cards for current balance, total income, outcome, and savings.
“””
if df.empty or ‘amount’ not in df.columns or ‘type’ not in df.columns:
total_income = 0
total_outcome = 0
total_savings = 0
else:
total_income = df[df[‘type’] == ‘income’][‘amount’].sum()
total_outcome = abs(df[df[‘type’] == ‘outcome’][‘amount’].sum())
total_savings = df[df[‘type’] == ‘saving’][‘amount’].sum()

```
if overall_df is not None and 'amount' in overall_df.columns and 'type' in overall_df.columns:
    overall_income = overall_df[overall_df['type'] == 'income']['amount'].sum()
    overall_outcome = abs(overall_df[overall_df['type'] == 'outcome']['amount'].sum())
    current_balance = overall_income - overall_outcome
else:
    current_balance = df['amount'].sum()
    
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="💰 Current Balance", value=format_vnd(current_balance))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="📈 Total Income", value=format_vnd(total_income))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="📉 Total Outcome", value=format_vnd(total_outcome))
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="🏦 Total Savings", value=format_vnd(total_savings))
    st.markdown('</div>', unsafe_allow_html=True)
```

def create_line_chart(df, title=“📈 Daily Trends by Category”):
“””
Creates a line chart showing daily trends by category.
“””
if df.empty or ‘timestamp’ not in df.columns or ‘type’ not in df.columns or ‘amount’ not in df.columns:
fig = go.Figure().update_layout(title=title, annotations=[dict(text=“No data available for this chart.”, showarrow=False)])
return fig

```
daily_data = df.groupby(['timestamp', 'type'])['amount'].sum().reset_index()

fig = px.line(
    daily_data, 
    x='timestamp', 
    y='amount', 
    color='type',
    title=title,
    labels={'amount': 'Amount (VND)', 'timestamp': 'Timestamp'}
)

fig.update_traces(mode='lines', fill='tozeroy')
fig.update_layout(height=400, hovermode='x unified')

return fig
```

def create_treemap(df, title=“Tree Map”):
“””
Creates a treemap visualizing the distribution of subtypes within types.
“””
if df.empty or ‘type’ not in df.columns or ‘subtype’ not in df.columns or ‘amount’ not in df.columns:
fig = go.Figure().update_layout(title=title, annotations=[dict(text=“No data available for this chart.”, showarrow=False)])
return fig

```
treemap_data = df.copy()
treemap_data['amount_abs'] = treemap_data['amount'].abs()

fig = px.treemap(
    treemap_data,
    path=['type', 'subtype'],
    values='amount_abs',
    title=title,
    color='amount',
    color_continuous_scale='RdYlBu'
)

fig.update_layout(height=400)
return fig
```

def create_monthly_table(df):
“””
Creates a table summarizing monthly income, outcome, and savings.
“””
if df.empty or ‘year_month’ not in df.columns or ‘type’ not in df.columns or ‘amount’ not in df.columns:
st.info(“Not enough data or missing columns for monthly summary table (year_month, type, amount).”)
return pd.DataFrame()

```
monthly_data = df.groupby(['year_month', 'type'])['amount'].sum().reset_index()
monthly_pivot = monthly_data.pivot(index='year_month', columns='type', values='amount').fillna(0)
monthly_pivot = monthly_pivot.sort_index(ascending=False)

for col_type in ['income', 'outcome', 'saving']: 
    if col_type in monthly_pivot.columns:
        if col_type == 'outcome':
            monthly_pivot[col_type] = monthly_pivot[col_type].abs().apply(format_vnd)
        else:
            monthly_pivot[col_type] = monthly_pivot[col_type].apply(format_vnd)

return monthly_pivot
```

def create_stacked_column_chart(df, group_by, title):
“””
Creates a stacked column chart grouped by a specified column and type.
“””
if df.empty or group_by not in df.columns or ‘type’ not in df.columns or ‘amount’ not in df.columns:
fig = go.Figure().update_layout(title=title, annotations=[dict(text=“No data available for this chart.”, showarrow=False)])
return fig

```
grouped_data = df.copy()
grouped_data['amount_abs'] = grouped_data['amount'].abs()

grouped_data_summed = grouped_data.groupby([group_by, 'type'])['amount_abs'].sum().reset_index()

sorted_categories = grouped_data.groupby(group_by)['amount_abs'].sum().sort_values(ascending=False).index.tolist()

fig = px.bar(
    grouped_data_summed,
    x=group_by,
    y='amount_abs',
    color='type',
    title=title,
    labels={'amount_abs': 'Amount (VND)'},
    category_orders={group_by: sorted_categories}
)

fig.update_layout(height=400)
return fig
```

def overview_page(df):
“”“Displays the Overview Dashboard page.”””
st.title(“🏠 Overview Dashboard”)

```
# Reload button
col_reload, col_status = st.columns([0.2, 0.8])
with col_reload:
    if st.button("🔄 Reload Data", key="reload_overview"):
        st.cache_data.clear()
        st.rerun()

# Filters
type_filter, date_range, subtype_filter = create_filters(df, "overview")
filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)

# All-time metrics
st.subheader("📊 All Time Summary")
create_metrics(df, overall_df=df)

# Current filtered metrics
if not df.empty and not filtered_df.equals(df):
    st.subheader("🔍 Filtered Summary")
    total_income_filtered = filtered_df[filtered_df['type'] == 'income']['amount'].sum() if 'type' in filtered_df.columns else 0
    total_outcome_filtered = abs(filtered_df[filtered_df['type'] == 'outcome']['amount'].sum()) if 'type' in filtered_df.columns else 0
    total_savings_filtered = filtered_df[filtered_df['type'] == 'saving']['amount'].sum() if 'type' in filtered_df.columns else 0

    col1_filtered, col2_filtered, col3_filtered, col4_filtered = st.columns(4)
    with col1_filtered:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        overall_balance = df[df['type'] == 'income']['amount'].sum() - abs(df[df['type'] == 'outcome']['amount'].sum())
        st.metric(label="💰 Current Balance (Overall)", value=format_vnd(overall_balance))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2_filtered:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="📈 Filtered Income", value=format_vnd(total_income_filtered))
        st.markdown('</div>', unsafe_allow_html=True)
    with col3_filtered:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="📉 Filtered Outcome", value=format_vnd(total_outcome_filtered))
        st.markdown('</div>', unsafe_allow_html=True)
    with col4_filtered:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="🏦 Filtered Savings", value=format_vnd(total_savings_filtered))
        st.markdown('</div>', unsafe_allow_html=True)

# Charts
st.markdown("---")
st.subheader("📈 Daily Trends by Category")

if 'type' in filtered_df.columns:
    income_df_ov = filtered_df[filtered_df['type'] == 'income'].copy()
    outcome_df_ov = filtered_df[filtered_df['type'] == 'outcome'].copy()
    saving_df_ov = filtered_df[filtered_df['type'] == 'saving'].copy()
else:
    st.info("Cannot categorize daily trends: 'type' column is missing in filtered data.")
    income_df_ov = pd.DataFrame(columns=filtered_df.columns)
    outcome_df_ov = pd.DataFrame(columns=filtered_df.columns)
    saving_df_ov = pd.DataFrame(columns=filtered_df.columns)

st.markdown("#### Income Trends")
line_fig_income = create_line_chart(income_df_ov, "📈 Income Daily Trends")
st.plotly_chart(line_fig_income, use_container_width=True)

st.markdown("#### Outcome Trends")
line_fig_outcome = create_line_chart(outcome_df_ov, "📉 Outcome Daily Trends")
st.plotly_chart(line_fig_outcome, use_container_width=True)

st.markdown("#### Savings Trends")
line_fig_saving = create_line_chart(saving_df_ov, "🏦 Savings Daily Trends")
st.plotly_chart(line_fig_saving, use_container_width=True)

st.markdown("---")
st.subheader("🗺️ Subtypes Distribution by Category")

col1_dist, col2_dist, col3_dist = st.columns(3)

with col1_dist:
    treemap_fig_income = create_treemap(income_df_ov, "🗺️ Income Distribution")
    st.plotly_chart(treemap_fig_income, use_container_width=True)

with col2_dist:
    treemap_fig_outcome = create_treemap(outcome_df_ov, "🗺️ Outcome Distribution")
    st.plotly_chart(treemap_fig_outcome, use_container_width=True)

with col3_dist:
    treemap_fig_saving = create_treemap(saving_df_ov, "🗺️ Savings Distribution")
    st.plotly_chart(treemap_fig_saving, use_container_width=True)

# Monthly table
st.markdown("---")
st.subheader("📅 Monthly Summary Table")
monthly_table = create_monthly_table(filtered_df)
if not monthly_table.empty:
    st.dataframe(monthly_table, use_container_width=True)
else:
    st.info("No data available for the monthly summary table with current filters.")
```

def income_page(df):
“”“Displays the Income Dashboard page.”””
st.title(“📈 Income Dashboard”)

```
# Reload button
col_reload, col_status = st.columns([0.2, 0.8])
with col_reload:
    if st.button("🔄 Reload Data", key="reload_income"):
        st.cache_data.clear()
        st.rerun()

income_df = df[df['type'] == 'income'].copy() if 'type' in df.columns and 'income' in df['type'].unique() else pd.DataFrame(columns=df.columns)

type_filter, date_range, subtype_filter = create_filters(income_df, "income")
filtered_df = apply_filters(income_df, type_filter, date_range, subtype_filter)

st.markdown("---")
st.subheader("💰 Income Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
    overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
    st.metric(label="💰 Current Overall Balance", value=format_vnd(overall_income - overall_outcome)) 
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("All Time Income", format_vnd(income_df['amount'].sum() if 'amount' in income_df.columns else 0))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Filtered Income", format_vnd(filtered_df['amount'].sum() if 'amount' in filtered_df.columns else 0))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Income Breakdown")
col1, col2 = st.columns(2)

with col1:
    subtype_fig = create_stacked_column_chart(filtered_df, 'subtype', "📊 Income by Subtype")
    st.plotly_chart(subtype_fig, use_container_width=True)

with col2:
    desc_fig = create_stacked_column_chart(filtered_df, 'description', "📝 Income by Description")
    st.plotly_chart(desc_fig, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    treemap_fig = create_treemap(filtered_df, "🗺️ Income Distribution")
    st.plotly_chart(treemap_fig, use_container_width=True)

with col4:
    st.subheader("📋 Income Details")
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['timestamp', 'subtype', 'description', 'amount']):
        income_table = filtered_df[['timestamp', 'subtype', 'description', 'amount']].sort_values('timestamp', ascending=False)
        income_table['amount_formatted'] = income_table['amount'].apply(format_vnd)
        st.dataframe(income_table[['timestamp', 'subtype', 'description', 'amount_formatted']], use_container_width=True)
    else:
        st.info("Missing required columns or no data for income details table.")
```

def outcome_page(df):
“”“Displays the Outcome Dashboard page.”””
st.title(“📉 Outcome Dashboard”)

```
# Reload button
col_reload, col_status = st.columns([0.2, 0.8])
with col_reload:
    if st.button("🔄 Reload Data", key="reload_outcome"):
        st.cache_data.clear()
        st.rerun()

outcome_df = df[df['type'] == 'outcome'].copy() if 'type' in df.columns and 'outcome' in df['type'].unique() else pd.DataFrame(columns=df.columns)
if 'amount' in outcome_df.columns:
    outcome_df['amount_display'] = outcome_df['amount'].abs()
else:
    outcome_df['amount_display'] = 0

type_filter, date_range, subtype_filter = create_filters(outcome_df, "outcome")
filtered_df = apply_filters(outcome_df, type_filter, date_range, subtype_filter)
if 'amount' in filtered_df.columns:
    filtered_df['amount_display'] = filtered_df['amount'].abs() 
else:
    filtered_df['amount_display'] = 0

st.markdown("---")
st.subheader("💸 Outcome Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
    overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
    st.metric(label="💰 Current Overall Balance", value=format_vnd(overall_income - overall_outcome))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("All Time Outcome", format_vnd(outcome_df['amount_display'].sum()))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Filtered Outcome", format_vnd(filtered_df['amount_display'].sum()))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Outcome Breakdown")
col1, col2 = st.columns(2)

with col1:
    subtype_fig = create_stacked_column_chart(filtered_df, 'subtype', "📊 Outcome by Subtype")
    st.plotly_chart(subtype_fig, use_container_width=True)

with col2:
    desc_fig = create_stacked_column_chart(filtered_df, 'description', "📝 Outcome by Description")
    st.plotly_chart(desc_fig, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    treemap_fig = create_treemap(filtered_df, "🗺️ Outcome Distribution")
    st.plotly_chart(treemap_fig, use_container_width=True)

with col4:
    st.subheader("📅 Monthly Outcome Summary")
    if not filtered_df.empty and 'year_month' in filtered_df.columns and 'amount_display' in filtered_df.columns:
        monthly_outcome = filtered_df.groupby('year_month')['amount_display'].sum().reset_index()
        monthly_outcome = monthly_outcome.sort_values('year_month', ascending=False)
        monthly_outcome['amount_formatted'] = monthly_outcome['amount_display'].apply(format_vnd)
        st.dataframe(monthly_outcome[['year_month', 'amount_formatted']], use_container_width=True)
    else:
        st.info("No data or missing required columns for monthly outcome summary.")
```

def savings_page(df):
“”“Displays the Savings Dashboard page.”””
st.title(“🏦 Savings Dashboard”)

```
# Reload button
col_reload, col_status = st.columns([0.2, 0.8])
with col_reload:
    if st.button("🔄 Reload Data", key="reload_savings"):
        st.cache_data.clear()
        st.rerun()

savings_df = df[df['type'] == 'saving'].copy() if 'type' in df.columns and 'saving' in df['type'].unique() else pd.DataFrame(columns=df.columns)

type_filter, date_range, subtype_filter = create_filters(savings_df, "savings")
filtered_df = apply_filters(savings_df, type_filter, date_range, subtype_filter)

# Metrics
    st.markdown("---")
    st.subheader("🏦 Savings Summary")
    col1, col2, col3 = st.columns(3) # Added a column for current balance
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Display overall balance from the full dataframe
        overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
        overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
        st.metric(label="💰 Current Overall Balance", value=format_vnd(overall_income - overall_outcome))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("All Time Savings", format_vnd(savings_df['amount'].sum() if 'amount' in savings_df.columns else 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Filtered Savings", format_vnd(filtered_df['amount'].sum() if 'amount' in filtered_df.columns else 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    st.subheader("📊 Savings Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        # Use create_stacked_column_chart directly which now handles stacking
        subtype_fig = create_stacked_column_chart(
            filtered_df, 'subtype', 
            "📊 Savings by Subtype"
        )
        st.plotly_chart(subtype_fig, use_container_width=True)
    
    with col2:
        # Use create_stacked_column_chart directly which now handles stacking
        desc_fig = create_stacked_column_chart(
            filtered_df, 'description', 
            "📝 Savings by Description"
        )
        st.plotly_chart(desc_fig, use_container_width=True)
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        treemap_fig = create_treemap(filtered_df, "🗺️ Savings Distribution")
        st.plotly_chart(treemap_fig, use_container_width=True)
    
    with col4:
        # Monthly savings table
        st.subheader("📅 Monthly Savings Summary")
        if not filtered_df.empty and 'year_month' in filtered_df.columns and 'amount' in filtered_df.columns:
            monthly_savings = filtered_df.groupby('year_month')['amount'].sum().reset_index()
            monthly_savings = monthly_savings.sort_values('year_month', ascending=False)
            monthly_savings['amount_formatted'] = monthly_savings['amount'].apply(format_vnd)
            st.dataframe(monthly_savings[['year_month', 'amount_formatted']], use_container_width=True)
        else:
            st.info("No data or missing required columns for monthly savings summary.")

def main():
    """Main application entry point."""
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("💰 Expense Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📈 Income", "📉 Outcome", "🏦 Savings"]
    )
    
    # Display selected page
    if page == "🏠 Overview":
        overview_page(df)
    elif page == "📈 Income":
        income_page(df)
    elif page == "📉 Outcome":
        outcome_page(df)
    elif page == "🏦 Savings":
        savings_page(df)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    if not df.empty and 'timestamp' in df.columns and 'type' in df.columns:
        st.sidebar.write(f"Total Records: {len(df):,}")
        st.sidebar.write(f"Timestamp Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        st.sidebar.write(f"Categories: {', '.join(df['type'].unique())}")
    else:
        st.sidebar.info("No data loaded. Showing sample data summary (if generated).")


if __name__ == "__main__":
    main()