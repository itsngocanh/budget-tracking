import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from functools import lru_cache
import gc
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Expense Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced color palette - moved to module level for better performance
COLORS = {
    'income': '#2E8B57',      # Sea Green
    'outcome': '#DC143C',     # Crimson
    'saving': '#4169E1',      # Royal Blue
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Custom CSS for better styling - moved to module level
CUSTOM_CSS = """
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .metric-container-income {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(46, 139, 87, 0.37);
    }
    
    .metric-container-outcome {
        background: linear-gradient(135deg, #DC143C 0%, #B22222 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(220, 20, 60, 0.37);
    }
    
    .metric-container-saving {
        background: linear-gradient(135deg, #4169E1 0%, #0000CD 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(65, 105, 225, 0.37);
    }
    
    .filter-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
    }
    
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
"""

# Apply CSS once at module level
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Optimized data loading with better caching and error handling
@st.cache_data(ttl=300)  # Cache for 5 minutes instead of default
def load_data():
    """
    Load and preprocess the data from '03.my_data.csv' with optimized performance.
    """
    try:
        # Use more efficient pandas reading with specific dtypes
        dtype_dict = {
            'Type': 'category',  # More memory efficient for categorical data
            'Amount': 'float64',
            'Subtype': 'category',
            'Description': 'string'
        }
        
        df = pd.read_csv('03.my_data.csv', dtype=dtype_dict, parse_dates=['Timestamp'])
        
        # Convert column names to lowercase once
        df.columns = df.columns.str.lower()
        
        # Validate required columns efficiently
        required_columns = {'type', 'amount', 'subtype', 'description', 'timestamp'}
        if not required_columns.issubset(set(df.columns)):
            st.warning("CSV structure not as expected. Creating sample data for demonstration.")
            return create_sample_data()

        # Optimize data types and conversions
        df['type'] = df['type'].astype('category').str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Efficiently drop rows with NaN values
        initial_rows = len(df)
        df.dropna(subset=['timestamp', 'amount'], inplace=True)
        if len(df) < initial_rows:
            st.info(f"Removed {initial_rows - len(df)} rows with invalid data.")
        
        # Create year_month column efficiently
        df['year_month'] = df['timestamp'].dt.to_period('M')
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        return df
        
    except FileNotFoundError:
        st.error("Data file '03.my_data.csv' not found. Creating sample data for demonstration.")
        return create_sample_data()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}. Creating sample data for demonstration.")
        return create_sample_data()

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

@st.cache_data(ttl=600)  # Cache sample data for 10 minutes
def create_sample_data():
    """
    Creates optimized sample DataFrame for demonstration purposes.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Pre-allocate lists for better performance
    data = []
    data.reserve = 1000  # Hint for memory allocation
    
    # Use vectorized operations where possible
    income_mask = np.random.random(len(dates)) < 0.1
    outcome_mask = np.random.random(len(dates)) < 0.3
    saving_mask = np.random.random(len(dates)) < 0.05
    
    # Generate income data
    income_dates = dates[income_mask]
    for current_date in income_dates:
        data.append({
            'timestamp': current_date,
            'type': 'income',
            'subtype': np.random.choice(['salary', 'freelance', 'investment', 'other']),
            'description': np.random.choice(['Monthly salary', 'Project payment', 'Dividend', 'Bonus']),
            'amount': np.random.randint(1000000, 15000000)
        })
    
    # Generate outcome data
    outcome_dates = dates[outcome_mask]
    for current_date in outcome_dates:
        data.append({
            'timestamp': current_date,
            'type': 'outcome',
            'subtype': np.random.choice(['food', 'transport', 'utilities', 'entertainment', 'shopping', 'thảo mộc']),
            'description': np.random.choice(['Groceries', 'Gas', 'Electric bill', 'Movie', 'Clothes', 'Herbs medicine']),
            'amount': -np.random.randint(50000, 500000)
        })
    
    # Generate savings data
    saving_dates = dates[saving_mask]
    for current_date in saving_dates:
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
        df = optimize_dataframe_memory(df)
    
    return df

# Optimized formatting functions with caching
@lru_cache(maxsize=1000)
def format_currency(amount):
    """
    Optimized currency formatting with caching for repeated values.
    """
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:,.1f}M"
    elif abs(amount) >= 1_000:
        return f"{amount / 1_000:,.0f}K"
    return f"{amount:,.0f}"

@lru_cache(maxsize=1000)
def format_vnd(amount):
    """
    Optimized VND formatting with caching for repeated values.
    """
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:,.2f}M VND"
    elif abs(amount) >= 1_000:
        return f"{amount / 1_000:,.1f}K VND"
    return f"{amount:,.0f} VND"

# Optimized filter creation with better performance
@st.cache_data(ttl=300)
def create_compact_filters(df, key_prefix=""):
    """
    Creates optimized filter widgets with better performance.
    """
    with st.expander("🎯 Filters & Date Range", expanded=True):
        col_quick, col_custom = st.columns([1, 2])
        
        with col_quick:
            st.markdown("**⚡ Quick Filters:**")
            quick_period = st.selectbox(
                "Period",
                ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Custom"],
                key=f"{key_prefix}_quick_period"
            )
        
        # Calculate date range efficiently
        if 'timestamp' in df.columns and not df.empty:
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            
            # Pre-calculate date ranges for better performance
            date_ranges = {
                "Last 30 Days": (max_date - timedelta(days=30), max_date),
                "Last 90 Days": (max_date - timedelta(days=90), max_date),
                "Last 6 Months": (max_date - timedelta(days=180), max_date),
                "Last Year": (max_date - timedelta(days=365), max_date),
                "All Time": (min_date, max_date)
            }
            
            if quick_period == "Custom":
                with col_custom:
                    st.markdown("**📅 Custom Date Range:**")
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        key=f"{key_prefix}_date_range"
                    )
            else:
                date_range = date_ranges.get(quick_period, (min_date, max_date))
        else:
            date_range = (datetime.now() - timedelta(days=30), datetime.now())
        
        # Type filter with optimized options
        with col_custom:
            st.markdown("**📊 Type Filter:**")
            type_options = ["All Types"] + list(df['type'].unique()) if 'type' in df.columns else ["All Types"]
            type_filter = st.selectbox(
                "Transaction Type",
                type_options,
                key=f"{key_prefix}_type_filter"
            )
        
        # Subtype filter with dynamic options
        subtype_options = ["All Subtypes"]
        if type_filter != "All Types" and 'subtype' in df.columns:
            filtered_df = df[df['type'] == type_filter]
            subtype_options.extend(filtered_df['subtype'].unique().tolist())
        
        subtype_filter = st.selectbox(
            "Subtype",
            subtype_options,
            key=f"{key_prefix}_subtype_filter"
        )
        
        return type_filter, date_range, subtype_filter

# Optimized filter application
@st.cache_data(ttl=300)
def apply_filters(df, type_filter, date_range, subtype_filter):
    """
    Apply filters to DataFrame with optimized performance.
    """
    filtered_df = df.copy()
    
    # Apply type filter
    if type_filter != "All Types":
        filtered_df = filtered_df[filtered_df['type'] == type_filter]
    
    # Apply date range filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        if isinstance(start_date, datetime):
            filtered_df = filtered_df[
                (filtered_df['timestamp'] >= start_date) & 
                (filtered_df['timestamp'] <= end_date)
            ]
    
    # Apply subtype filter
    if subtype_filter != "All Subtypes":
        filtered_df = filtered_df[filtered_df['subtype'] == subtype_filter]
    
    return filtered_df

# Optimized metric card creation
def create_enhanced_metric_card(label, value, container_class="metric-container", delta=None, delta_color="normal"):
    """
    Create optimized metric cards with better performance.
    """
    st.markdown(
        f"""
        <div class="{container_class}">
            <h3>{label}</h3>
            <h2>{value}</h2>
            {f'<p style="color: {delta_color};">{delta}</p>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

# Optimized metrics creation with better caching
@st.cache_data(ttl=300)
def create_collapsible_metrics(df, overall_df=None, title="💎 Financial Summary", expanded=True):
    """
    Create optimized collapsible metrics with better performance.
    """
    with st.expander(title, expanded=expanded):
        if df.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Pre-calculate all metrics efficiently
        metrics = calculate_metrics(df, overall_df)
        
        # Display metrics in columns
        cols = st.columns(4)
        for i, (label, value, container_class, delta, delta_color) in enumerate(metrics):
            with cols[i % 4]:
                create_enhanced_metric_card(label, value, container_class, delta, delta_color)

@st.cache_data(ttl=300)
def calculate_metrics(df, overall_df=None):
    """
    Calculate all metrics efficiently in one pass.
    """
    metrics = []
    
    # Calculate totals efficiently
    income_total = df[df['type'] == 'income']['amount'].sum()
    outcome_total = abs(df[df['type'] == 'outcome']['amount'].sum())
    saving_total = df[df['type'] == 'saving']['amount'].sum()
    net_flow = income_total - outcome_total
    
    # Calculate deltas if overall_df is provided
    if overall_df is not None and not overall_df.empty:
        overall_income = overall_df[overall_df['type'] == 'income']['amount'].sum()
        overall_outcome = abs(overall_df[overall_df['type'] == 'outcome']['amount'].sum())
        overall_saving = overall_df[overall_df['type'] == 'saving']['amount'].sum()
        
        income_delta = f"Δ {format_currency(income_total - overall_income)}"
        outcome_delta = f"Δ {format_currency(outcome_total - overall_outcome)}"
        saving_delta = f"Δ {format_currency(saving_total - overall_saving)}"
        net_delta = f"Δ {format_currency(net_flow - (overall_income - overall_outcome))}"
    else:
        income_delta = outcome_delta = saving_delta = net_delta = None
    
    # Add metrics to list
    metrics.extend([
        ("💰 Total Income", format_vnd(income_total), "metric-container-income", income_delta, "normal"),
        ("💸 Total Outcome", format_vnd(outcome_total), "metric-container-outcome", outcome_delta, "normal"),
        ("🏦 Total Savings", format_vnd(saving_total), "metric-container-saving", saving_delta, "normal"),
        ("📊 Net Flow", format_vnd(net_flow), "metric-container", net_delta, "normal")
    ])
    
    return metrics

# Optimized chart creation functions
@st.cache_data(ttl=300)
def create_flexible_trend_chart(df, title="📈 Financial Trends", key_prefix="trend"):
    """
    Create optimized trend chart with better performance.
    """
    if df.empty:
        st.warning("No data available for trend analysis.")
        return
    
    # Pre-calculate aggregated data efficiently
    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp'].dt.date
    
    # Use more efficient aggregation
    daily_data = df_copy.groupby(['date', 'type'])['amount'].sum().reset_index()
    
    # Create pivot table for better performance
    pivot_data = daily_data.pivot(index='date', columns='type', values='amount').fillna(0)
    
    # Create cumulative data efficiently
    cumulative_data = pivot_data.cumsum()
    
    # Create chart
    fig = go.Figure()
    
    for col in cumulative_data.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_data.index,
            y=cumulative_data[col],
            mode='lines+markers',
            name=col.title(),
            line=dict(color=COLORS.get(col, '#1f77b4'), width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Amount (VND)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main function with optimized performance
def main():
    """Main application entry point with optimized performance."""
    # Load data once and cache it
    df = load_data()
    
    # Optimized sidebar with better performance
    with st.sidebar:
        st.markdown("### 💰 Enhanced Financial Dashboard")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "🧭 Navigate to:",
            ["🏠 Overview", "📈 Income", "📉 Outcome", "🏦 Savings"],
            help="Select a dashboard view"
        )
        
        # Data summary with optimized calculations
        st.markdown("---")
        st.markdown("### 📊 Data Insights")
        
        if not df.empty:
            # Calculate metrics efficiently
            total_transactions = len(df)
            date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
            
            if 'type' in df.columns:
                type_counts = df['type'].value_counts()
                income_count = type_counts.get('income', 0)
                outcome_count = type_counts.get('outcome', 0)
                savings_count = type_counts.get('saving', 0)
                
                st.metric("📊 Total Records", f"{total_transactions:,}")
                st.metric("📈 Income Entries", f"{income_count:,}")
                st.metric("📉 Outcome Entries", f"{outcome_count:,}")
                st.metric("🏦 Savings Entries", f"{savings_count:,}")
                
                st.info(f"📅 **Date Range**\n{date_range}")
                
                # Quick financial health indicator
                total_income = df[df['type'] == 'income']['amount'].sum()
                total_outcomes = abs(df[df['type'] == 'outcome']['amount'].sum())
                financial_health = "🟢 Healthy" if total_income > total_outcomes else "🔴 Concerning"
                st.markdown(f"**Financial Health:** {financial_health}")
        else:
            st.warning("📊 No data loaded")
    
    # Display selected page with optimized performance
    if page == "🏠 Overview":
        overview_page(df)
    elif page == "📈 Income":
        income_page(df)
    elif page == "📉 Outcome":
        outcome_page(df)
    elif page == "🏦 Savings":
        savings_page(df)
    
    # Force garbage collection for better memory management
    gc.collect()

# Placeholder functions for other pages (implement as needed)
def overview_page(df):
    """Optimized overview page implementation."""
    st.title("🏠 Financial Overview")
    
    # Apply filters
    type_filter, date_range, subtype_filter = create_compact_filters(df, "overview")
    filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)
    
    # Display metrics
    create_collapsible_metrics(filtered_df, df, "💎 Financial Summary", expanded=True)
    
    # Display trend chart
    create_flexible_trend_chart(filtered_df, "📈 Financial Trends", "overview_trend")

def income_page(df):
    """Optimized income page implementation."""
    st.title("📈 Income Analysis")
    
    # Apply filters
    type_filter, date_range, subtype_filter = create_compact_filters(df, "income")
    filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)
    
    # Filter for income only
    income_df = filtered_df[filtered_df['type'] == 'income']
    
    if not income_df.empty:
        create_collapsible_metrics(income_df, filtered_df, "💰 Income Summary", expanded=True)
        create_flexible_trend_chart(income_df, "📈 Income Trends", "income_trend")
    else:
        st.warning("No income data available for the selected filters.")

def outcome_page(df):
    """Optimized outcome page implementation."""
    st.title("📉 Outcome Analysis")
    
    # Apply filters
    type_filter, date_range, subtype_filter = create_compact_filters(df, "outcome")
    filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)
    
    # Filter for outcome only
    outcome_df = filtered_df[filtered_df['type'] == 'outcome']
    
    if not outcome_df.empty:
        create_collapsible_metrics(outcome_df, filtered_df, "💸 Outcome Summary", expanded=True)
        create_flexible_trend_chart(outcome_df, "📉 Outcome Trends", "outcome_trend")
    else:
        st.warning("No outcome data available for the selected filters.")

def savings_page(df):
    """Optimized savings page implementation."""
    st.title("🏦 Savings Analysis")
    
    # Apply filters
    type_filter, date_range, subtype_filter = create_compact_filters(df, "savings")
    filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)
    
    # Filter for savings only
    savings_df = filtered_df[filtered_df['type'] == 'saving']
    
    if not savings_df.empty:
        create_collapsible_metrics(savings_df, filtered_df, "🏦 Savings Summary", expanded=True)
        create_flexible_trend_chart(savings_df, "📈 Savings Trends", "savings_trend")
    else:
        st.warning("No savings data available for the selected filters.")

if __name__ == "__main__":
    main()