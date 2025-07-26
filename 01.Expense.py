import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Expense Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced color palette
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

# Custom CSS for better styling
st.markdown("""
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
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load and preprocess the data from '03.my_data.csv'.
    If the file is not found or its structure is unexpected, sample data is created.
    """
    try:
        df = pd.read_csv('03.my_data.csv')

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

def create_sample_data():
    """
    Creates a sample DataFrame for demonstration purposes
    if the actual CSV file is not found or malformed.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

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
                'subtype': np.random.choice(['food', 'transport', 'utilities', 'entertainment', 'shopping', 'thảo mộc']),
                'description': np.random.choice(['Groceries', 'Gas', 'Electric bill', 'Movie', 'Clothes', 'Herbs medicine']),
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

def format_currency(amount):
    """
    Formats a numerical amount with M/K indicators for better readability.
    """
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:,.1f}M"
    elif abs(amount) >= 1_000:
        return f"{amount / 1_000:,.0f}K"
    return f"{amount:,.0f}"

def format_vnd(amount):
    """
    Formats a numerical amount into VND currency string, using 'M' for millions.
    """
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:,.2f}M VND"
    elif abs(amount) >= 1_000:
        return f"{amount / 1_000:,.1f}K VND"
    return f"{amount:,.0f} VND"

def create_compact_filters(df, key_prefix=""):
    """
    Creates compact, easily accessible filter widgets in expandable sections.
    """
    with st.expander("🎯 Filters & Date Range", expanded=True):
        # Quick date range buttons
        col_quick, col_custom = st.columns([1, 2])
        
        with col_quick:
            st.markdown("**⚡ Quick Filters:**")
            quick_period = st.selectbox(
                "Period",
                ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Custom"],
                key=f"{key_prefix}_quick_period"
            )
        
        # Calculate date range based on selection
        if 'timestamp' in df.columns and not df.empty:
            max_date = df['timestamp'].max().date()
            min_date = df['timestamp'].min().date()
            
            if quick_period == "Last 30 Days":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif quick_period == "Last 90 Days":
                start_date = max_date - timedelta(days=90)
                end_date = max_date
            elif quick_period == "Last 6 Months":
                start_date = max_date - timedelta(days=180)
                end_date = max_date
            elif quick_period == "Last Year":
                start_date = max_date - timedelta(days=365)
                end_date = max_date
            elif quick_period == "Custom":
                with col_custom:
                    st.markdown("**📅 Custom Date Range:**")
                    date_range = st.date_input(
                        "Select dates",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key=f"{key_prefix}_custom_date"
                    )
                    start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
            else:  # All Time
                start_date, end_date = min_date, max_date
        else:
            start_date, end_date = None, None

        # Compact filter row
        col1, col2, col3 = st.columns(3)

        with col1:
            type_options = df['type'].unique() if 'type' in df.columns and not df['type'].empty else []
            type_filter = st.multiselect(
                "💼 Type",
                options=list(type_options),
                default=list(type_options),
                key=f"{key_prefix}_type"
            )

        with col2:
            subtype_options = []
            if 'subtype' in df.columns and 'type' in df.columns and not df.empty:
                if type_filter:
                    filtered_by_type_df = df[df['type'].isin(type_filter)]
                    if not filtered_by_type_df.empty:
                        subtype_options = filtered_by_type_df['subtype'].unique()
                else:
                    subtype_options = df['subtype'].unique()
                
            subtype_filter = st.multiselect(
                "🏷️ Category",
                options=list(subtype_options),
                default=list(subtype_options),
                key=f"{key_prefix}_subtype"
            )

        with col3:
            # Show selected period info
            if start_date and end_date:
                st.info(f"📅 {start_date} to {end_date}")

    return type_filter, (start_date, end_date), subtype_filter

def apply_filters(df, type_filter, date_range, subtype_filter):
    """
    Applies the selected filters to the DataFrame.
    """
    filtered_df = df.copy()

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

def create_enhanced_metric_card(label, value, container_class="metric-container", delta=None, delta_color="normal"):
    """Creates an enhanced metric card with better styling"""
    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
    if delta:
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color)
    else:
        st.metric(label=label, value=value)
    st.markdown('</div>', unsafe_allow_html=True)

def create_collapsible_metrics(df, overall_df=None, title="💎 Financial Summary", expanded=True):
    """
    Creates collapsible metric cards for current balance, total income, outcome, and savings.
    """
    with st.expander(title, expanded=expanded):
        if df.empty or 'amount' not in df.columns or 'type' not in df.columns:
            total_income = 0
            total_outcome = 0
            total_savings = 0
        else:
            total_income = df[df['type'] == 'income']['amount'].sum()
            total_outcome = abs(df[df['type'] == 'outcome']['amount'].sum())
            total_savings = df[df['type'] == 'saving']['amount'].sum()

        if overall_df is not None and 'amount' in overall_df.columns and 'type' in overall_df.columns:
            overall_income = overall_df[overall_df['type'] == 'income']['amount'].sum()
            overall_outcome = abs(overall_df[overall_df['type'] == 'outcome']['amount'].sum())
            current_balance = overall_income - overall_outcome
        else:
            current_balance = df['amount'].sum()
            
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            balance_color = "normal" if current_balance >= 0 else "inverse"
            create_enhanced_metric_card("💰 Current Balance", format_vnd(current_balance), 
                                      "metric-container", delta_color=balance_color)

        with col2:
            create_enhanced_metric_card("📈 Total Income", format_vnd(total_income), 
                                      "metric-container-income")

        with col3:
            create_enhanced_metric_card("📉 Total Outcome", format_vnd(total_outcome), 
                                      "metric-container-outcome")

        with col4:
            create_enhanced_metric_card("🏦 Total Savings", format_vnd(total_savings), 
                                      "metric-container-saving")

def create_flexible_trend_chart(df, title="📈 Financial Trends", key_prefix="trend"):
    """
    Creates a flexible trend chart with period options (Daily, Weekly, Monthly, Quarterly, Yearly).
    Mobile-friendly with better number display and improved Y-axis formatting.
    """
    if df.empty or 'timestamp' not in df.columns or 'type' not in df.columns or 'amount' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False, font=dict(size=16, color="gray"))]
        )
        return fig

    # Period selection
    col_period, col_info = st.columns([1, 2])
    with col_period:
        period = st.selectbox(
            "📊 View by:",
            ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
            index=2,  # Default to Monthly
            key=f"{key_prefix}_period"
        )

    # Aggregate data based on selected period
    df_copy = df.copy()
    
    if period == "Daily":
        df_copy['period'] = df_copy['timestamp'].dt.date
        period_format = '%Y-%m-%d'
    elif period == "Weekly":
        df_copy['period'] = df_copy['timestamp'].dt.to_period('W').dt.start_time
        period_format = '%Y-W%U'
    elif period == "Monthly":
        df_copy['period'] = df_copy['timestamp'].dt.to_period('M').dt.start_time
        period_format = '%Y-%m'
    elif period == "Quarterly":
        df_copy['period'] = df_copy['timestamp'].dt.to_period('Q').dt.start_time
        period_format = '%Y-Q%q'
    else:  # Yearly
        df_copy['period'] = df_copy['timestamp'].dt.to_period('Y').dt.start_time
        period_format = '%Y'

    # Group data by period and type
    trend_data = df_copy.groupby(['period', 'type'])['amount'].sum().reset_index()
    
    fig = go.Figure()

    # Add traces for each transaction type
    for transaction_type in trend_data['type'].unique():
        type_data = trend_data[trend_data['type'] == transaction_type].sort_values('period')
        
        fig.add_trace(go.Scatter(
            x=type_data['period'],
            y=type_data['amount'],
            mode='lines+markers+text',
            name=transaction_type.title(),
            line=dict(
                color=COLORS.get(transaction_type, COLORS['primary']),
                width=4,
                shape='spline'
            ),
            marker=dict(
                size=10,
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=[format_currency(val) for val in type_data['amount']],
            textposition='top center',
            textfont=dict(size=12, color=COLORS.get(transaction_type, COLORS['primary'])),
            hovertemplate=f'<b>{transaction_type.title()}</b><br>' +
                         'Period: %{x}<br>' +
                         'Amount: %{text}<br>' +
                         '<extra></extra>'
        ))

    # Mobile-friendly layout with improved Y-axis formatting
    fig.update_layout(
        title=dict(
            text=f"{title} - {period}",
            font=dict(size=18, color=COLORS['dark']),
            x=0.5
        ),
        xaxis=dict(
            title=f"{period} Period",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            zeroline=False,
            tickangle=45 if period in ["Daily", "Weekly"] else 0
        ),
        yaxis=dict(
            title="Amount",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            # Custom tick format function to show M/K format
            tickformat='.0f',
            tickvals=None,  # Let Plotly choose tick values
            ticktext=None   # Will be set programmatically
        ),
        hovermode='x unified',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        # Mobile responsive
        margin=dict(l=50, r=50, t=80, b=80)
    )

    # Custom Y-axis formatting for M/K display
    if not trend_data.empty:
        max_val = trend_data['amount'].max()
        min_val = trend_data['amount'].min()
        
        # Generate appropriate tick values
        range_val = max_val - min_val
        if range_val > 0:
            if max_val >= 1_000_000:
                # Generate ticks in millions
                tick_step = max(1_000_000, int(range_val / 5 / 1_000_000) * 1_000_000)
                tick_vals = list(range(int(min_val), int(max_val) + tick_step, tick_step))
                tick_texts = [format_currency(val) for val in tick_vals]
            elif max_val >= 1_000:
                # Generate ticks in thousands
                tick_step = max(1_000, int(range_val / 5 / 1_000) * 1_000)
                tick_vals = list(range(int(min_val), int(max_val) + tick_step, tick_step))
                tick_texts = [format_currency(val) for val in tick_vals]
            else:
                # Keep original formatting for small numbers
                tick_vals = None
                tick_texts = None
            
            if tick_vals:
                fig.update_yaxes(tickvals=tick_vals, ticktext=tick_texts)

    with col_info:
        if not trend_data.empty:
            total_periods = len(trend_data['period'].unique())
            st.info(f"📊 Showing {total_periods} {period.lower()} periods")

    return fig

def create_enhanced_treemap(df, title="🗺️ Distribution Treemap"):
    """
    Creates an enhanced treemap with better colors and styling.
    """
    if df.empty or 'type' not in df.columns or 'subtype' not in df.columns or 'amount' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available for this chart.", 
                            showarrow=False, font=dict(size=16, color="gray"))]
        )
        return fig

    treemap_data = df.copy()
    treemap_data['amount_abs'] = treemap_data['amount'].abs()
    
    # Create color mapping
    color_values = []
    for _, row in treemap_data.iterrows():
        if row['type'] == 'income':
            color_values.append(1)
        elif row['type'] == 'outcome':
            color_values.append(-1)
        else:
            color_values.append(0)
    
    treemap_data['color_val'] = color_values

    fig = go.Figure(go.Treemap(
        labels=treemap_data['subtype'],
        parents=treemap_data['type'],
        values=treemap_data['amount_abs'],
        textinfo="label+value+percent entry",
        textfont=dict(size=12, color="white"),
        marker=dict(
            colorscale=[[0, COLORS['outcome']], 
                       [0.5, COLORS['saving']], 
                       [1, COLORS['income']]],
            colorbar=dict(
                title="Transaction Type",
                tickvals=[-1, 0, 1],
                ticktext=["Outcome", "Savings", "Income"]
            ),
            line=dict(width=2, color="white")
        ),
        hovertemplate='<b>%{label}</b><br>' +
                     'Amount: %{value:,.0f} VND<br>' +
                     'Percentage: %{percentEntry}<br>' +
                     '<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['dark']),
            x=0.5
        ),
        height=400,
        margin=dict(t=50, l=25, r=25, b=25),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_top3_pie_chart(df, group_by, title="📊 Distribution", show_legend=True):
    """
    Creates a pie chart showing only top 3 categories and 'Others'.
    """
    if df.empty or group_by not in df.columns or 'amount' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False)]
        )
        return fig

    pie_data = df.copy()
    pie_data['amount_abs'] = pie_data['amount'].abs()
    grouped_data = pie_data.groupby(group_by)['amount_abs'].sum().sort_values(ascending=False)
    
    # Keep top 3 and group rest as 'Others'
    if len(grouped_data) > 3:
        top3 = grouped_data.head(3)
        others_sum = grouped_data.tail(-3).sum()
        
        # Create final data
        labels = list(top3.index) + ['Others']
        values = list(top3.values) + [others_sum]
    else:
        labels = list(grouped_data.index)
        values = list(grouped_data.values)
    
    # Create custom colors - vibrant for top 3, gray for others
    colors = [COLORS['income'], COLORS['outcome'], COLORS['saving'], '#CCCCCC'][:len(labels)]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent+value',
        textfont=dict(size=12, color='white'),
        texttemplate='<b>%{label}</b><br>%{percent}<br>%{text}',
        text=[format_currency(v) for v in values],
        hovertemplate='<b>%{label}</b><br>' +
                     'Amount: %{text}<br>' +
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS['dark']),
            x=0.5
        ),
        height=350,
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            x=1.05,
            y=0.5,
            font=dict(size=12)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=25, r=25, b=25),
        # Mobile responsive
        font=dict(size=11)
    )

    return fig

def create_sorted_bar_chart(df, group_by, title, orientation='v'):
    """
    Creates a bar chart sorted from highest to lowest values.
    """
    if df.empty or group_by not in df.columns or 'type' not in df.columns or 'amount' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False)]
        )
        return fig

    grouped_data = df.copy()
    grouped_data['amount_abs'] = grouped_data['amount'].abs()
    
    # Sort by total amount (highest to lowest)
    total_by_group = grouped_data.groupby(group_by)['amount_abs'].sum().sort_values(ascending=False)
    sorted_categories = total_by_group.index.tolist()
    
    # Group by category and type, then sort
    grouped_data_summed = grouped_data.groupby([group_by, 'type'])['amount_abs'].sum().reset_index()

    fig = go.Figure()

    # Add traces for each transaction type
    for transaction_type in grouped_data_summed['type'].unique():
        type_data = grouped_data_summed[grouped_data_summed['type'] == transaction_type]
        
        # Sort type_data according to sorted_categories
        type_data_sorted = []
        for category in sorted_categories:
            category_data = type_data[type_data[group_by] == category]
            if not category_data.empty:
                type_data_sorted.append(category_data.iloc[0])
        
        if type_data_sorted:
            type_data_df = pd.DataFrame(type_data_sorted)
            
            if orientation == 'h':
                fig.add_trace(go.Bar(
                    y=type_data_df[group_by],
                    x=type_data_df['amount_abs'],
                    name=transaction_type.title(),
                    orientation='h',
                    marker=dict(
                        color=COLORS.get(transaction_type, COLORS['primary']),
                        line=dict(color='white', width=1)
                    ),
                    text=[format_currency(val) for val in type_data_df['amount_abs']],
                    textposition='outside',
                    hovertemplate=f'<b>{transaction_type.title()}</b><br>' +
                                 f'{group_by}: %{{y}}<br>' +
                                 'Amount: %{text}<br>' +
                                 '<extra></extra>'
                ))
            else:
                fig.add_trace(go.Bar(
                    x=type_data_df[group_by],
                    y=type_data_df['amount_abs'],
                    name=transaction_type.title(),
                    marker=dict(
                        color=COLORS.get(transaction_type, COLORS['primary']),
                        line=dict(color='white', width=1)
                    ),
                    text=[format_currency(val) for val in type_data_df['amount_abs']],
                    textposition='outside',
                    hovertemplate=f'<b>{transaction_type.title()}</b><br>' +
                                 f'{group_by}: %{{x}}<br>' +
                                 'Amount: %{text}<br>' +
                                 '<extra></extra>'
                ))

    # Update layout for mobile-friendly display
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS['dark']),
            x=0.5
        ),
        barmode='stack',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            title="Categories" if orientation == 'v' else "Amount",
            tickangle=45 if orientation == 'v' else 0,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            title="Amount" if orientation == 'v' else "Categories",
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            y=-0.25 if orientation == 'v' else -0.15,
            x=0.5,
            xanchor='center',
            font=dict(size=12)
        ),
        # Mobile responsive margins
        margin=dict(l=40, r=40, t=60, b=100 if orientation == 'v' else 80)
    )

    return fig

def create_monthly_table(df):
    """
    Creates an enhanced table summarizing monthly income, outcome, and savings.
    """
    if df.empty or 'year_month' not in df.columns or 'type' not in df.columns or 'amount' not in df.columns:
        st.info("Not enough data or missing columns for monthly summary table.")
        return pd.DataFrame()

    monthly_data = df.groupby(['year_month', 'type'])['amount'].sum().reset_index()
    monthly_pivot = monthly_data.pivot(index='year_month', columns='type', values='amount').fillna(0)
    monthly_pivot = monthly_pivot.sort_index(ascending=False)

    # Calculate net savings
    if 'income' in monthly_pivot.columns and 'outcome' in monthly_pivot.columns:
        monthly_pivot['net_savings'] = monthly_pivot['income'] + monthly_pivot['outcome']  # outcome is negative
    
    # Format values
    for col_type in monthly_pivot.columns:
        if col_type == 'outcome':
            monthly_pivot[col_type] = monthly_pivot[col_type].abs().apply(format_currency)
        else:
            monthly_pivot[col_type] = monthly_pivot[col_type].apply(format_currency)

    return monthly_pivot

def create_enhanced_area_chart(df, title="📈 Cumulative Trends"):
    """
    Creates an enhanced area chart showing cumulative trends.
    """
    if df.empty or 'timestamp' not in df.columns or 'type' not in df.columns or 'amount' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No data available", showarrow=False)]
        )
        return fig

    # Prepare cumulative data
    daily_data = df.groupby(['timestamp', 'type'])['amount'].sum().reset_index()
    
    fig = go.Figure()

    for transaction_type in daily_data['type'].unique():
        type_data = daily_data[daily_data['type'] == transaction_type].sort_values('timestamp')
        type_data['cumulative'] = type_data['amount'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=type_data['timestamp'],
            y=type_data['cumulative'],
            mode='lines',
            name=f"Cumulative {transaction_type.title()}",
            fill='tonexty' if transaction_type != 'income' else 'tozeroy',
            line=dict(
                color=COLORS.get(transaction_type, COLORS['primary']),
                width=2,
                shape='spline'
            ),
            fillcolor=f"rgba{(*[int(COLORS.get(transaction_type, COLORS['primary'])[i:i+2], 16) for i in (1, 3, 5)], 0.3)}",
            hovertemplate=f'<b>Cumulative {transaction_type.title()}</b><br>' +
                         'Date: %{x}<br>' +
                         'Amount: %{text}<br>' +
                         '<extra></extra>',
            text=[format_currency(v) for v in type_data['cumulative']]
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS['dark']),
            x=0.5
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title="Cumulative Amount",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor='center'
        )
    )

    return fig

def create_thao_moc_analysis(df):
    """
    Creates detailed analysis for Thảo Mộc category in outcome data.
    """
    if df.empty or 'subtype' not in df.columns:
        st.info("No data available for Thảo Mộc analysis.")
        return
    
    # Filter for Thảo Mộc data
    thao_moc_df = df[df['subtype'].str.lower() == 'thảo mộc'].copy()
    
    if thao_moc_df.empty:
        st.info("No Thảo Mộc transactions found in the data.")
        return
    
    st.subheader("🌿 Deep Dive: Thảo Mộc Analysis")
    
    # Thảo Mộc metrics
    with st.expander("🌿 Thảo Mộc Summary Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        total_thao_moc = thao_moc_df['amount'].abs().sum()
        avg_thao_moc = thao_moc_df['amount'].abs().mean()
        transaction_count = len(thao_moc_df)
        
        with col1:
            create_enhanced_metric_card("🌿 Total Thảo Mộc", format_vnd(total_thao_moc), "metric-container-outcome")
        with col2:
            create_enhanced_metric_card("📊 Avg Transaction", format_vnd(avg_thao_moc), "metric-container")
        with col3:
            create_enhanced_metric_card("🔢 Total Transactions", f"{transaction_count:,}", "metric-container")
        with col4:
            # Calculate percentage of total outcomes
            total_outcomes = df[df['type'] == 'outcome']['amount'].abs().sum() if 'type' in df.columns else total_thao_moc
            percentage = (total_thao_moc / total_outcomes * 100) if total_outcomes > 0 else 0
            create_enhanced_metric_card("📈 % of Total Outcomes", f"{percentage:.1f}%", "metric-container-saving")
    
    # Thảo Mộc charts
    col1_tm, col2_tm = st.columns(2)
    
    with col1_tm:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Thảo Mộc trend over time
        tm_trend = create_flexible_trend_chart(thao_moc_df, "🌿 Thảo Mộc Spending Trends", "thao_moc_trend")
        st.plotly_chart(tm_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_tm:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Thảo Mộc by description if available
        if 'description' in thao_moc_df.columns:
            tm_desc_chart = create_sorted_bar_chart(thao_moc_df, 'description', "🏷️ Thảo Mộc by Description", 'h')
            st.plotly_chart(tm_desc_chart, use_container_width=True)
        else:
            st.info("No description data available for detailed breakdown.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Thảo Mộc transactions
    st.subheader("📋 Recent Thảo Mộc Transactions")
    if 'timestamp' in thao_moc_df.columns:
        recent_tm = thao_moc_df.sort_values('timestamp', ascending=False).head(10)
        display_cols = ['timestamp', 'description', 'amount'] if 'description' in recent_tm.columns else ['timestamp', 'amount']
        
        if display_cols:
            recent_display = recent_tm[display_cols].copy()
            recent_display['amount_formatted'] = recent_display['amount'].abs().apply(format_currency)
            recent_display['date'] = recent_display['timestamp'].dt.strftime('%Y-%m-%d')
            
            display_columns = ['date', 'amount_formatted']
            column_config = {"date": "Date", "amount_formatted": "Amount"}
            
            if 'description' in recent_display.columns:
                display_columns.insert(1, 'description')
                column_config["description"] = "Description"
            
            st.dataframe(
                recent_display[display_columns],
                use_container_width=True,
                height=300,
                column_config=column_config
            )

def overview_page(df):
    """Displays the enhanced Overview Dashboard page."""
    st.markdown('<h1 class="main-header">🏠 Financial Overview Dashboard</h1>', unsafe_allow_html=True)

    # Reload button with better styling
    col_reload, col_info = st.columns([0.2, 0.8])
    with col_reload:
        if st.button("🔄 Reload Data", key="reload_overview", help="Refresh data from source"):
            st.cache_data.clear()
            st.rerun()
    
    with col_info:
        if not df.empty:
            st.info(f"📊 Loaded {len(df):,} transactions | Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")

    # Enhanced compact filters
    type_filter, date_range, subtype_filter = create_compact_filters(df, "overview")
    filtered_df = apply_filters(df, type_filter, date_range, subtype_filter)

    # All-time metrics with enhanced styling - COLLAPSIBLE
    create_collapsible_metrics(df, overall_df=df, title="💎 Financial Summary - All Time", expanded=True)

    # Current filtered metrics
    if not df.empty and not filtered_df.equals(df):
        create_collapsible_metrics(filtered_df, overall_df=df, title="🎯 Filtered Period Analysis", expanded=False)

    # Enhanced Charts Section
    st.markdown("---")
    st.subheader("📈 Comprehensive Financial Analysis")

    # Main trend analysis with flexible periods
    col1_trend, col2_trend = st.columns([2, 1])
    with col1_trend:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        trend_chart = create_flexible_trend_chart(filtered_df, "📈 Financial Trends", "overview_trend")
        st.plotly_chart(trend_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_trend:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Overall distribution pie chart (top 3 + others)
        if 'type' in filtered_df.columns:
            pie_chart = create_top3_pie_chart(filtered_df, 'type', "💼 Transaction Types")
            st.plotly_chart(pie_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Cumulative trends
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    cumulative_chart = create_enhanced_area_chart(filtered_df, "📊 Cumulative Financial Growth")
    st.plotly_chart(cumulative_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Category breakdown by type
    st.markdown("---")
    st.subheader("🗂️ Category-wise Financial Breakdown")

    if 'type' in filtered_df.columns:
        income_df_ov = filtered_df[filtered_df['type'] == 'income'].copy()
        outcome_df_ov = filtered_df[filtered_df['type'] == 'outcome'].copy()
        saving_df_ov = filtered_df[filtered_df['type'] == 'saving'].copy()

        tab1, tab2, tab3 = st.tabs(["💚 Income Analysis", "🔴 Outcome Analysis", "💙 Savings Analysis"])
        
        with tab1:
            col1_inc, col2_inc = st.columns(2)
            with col1_inc:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                income_bar = create_sorted_bar_chart(income_df_ov, 'subtype', "📊 Income by Category")
                st.plotly_chart(income_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2_inc:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                income_pie = create_top3_pie_chart(income_df_ov, 'subtype', "🥧 Top Income Sources")
                st.plotly_chart(income_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            col1_out, col2_out = st.columns(2)
            with col1_out:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                outcome_bar = create_sorted_bar_chart(outcome_df_ov, 'subtype', "📊 Outcome by Category")
                st.plotly_chart(outcome_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2_out:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                outcome_pie = create_top3_pie_chart(outcome_df_ov, 'subtype', "🥧 Top Outcome Categories")
                st.plotly_chart(outcome_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            col1_sav, col2_sav = st.columns(2)
            with col1_sav:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                saving_bar = create_sorted_bar_chart(saving_df_ov, 'subtype', "📊 Savings by Category")
                st.plotly_chart(saving_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2_sav:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                saving_pie = create_top3_pie_chart(saving_df_ov, 'subtype', "🥧 Top Savings Types")
                st.plotly_chart(saving_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Monthly Summary (TABLE ONLY - NO DUPLICATE CHART)
    st.markdown("---")
    st.subheader("📅 Monthly Financial Summary")
    
    monthly_table = create_monthly_table(filtered_df)
    if not monthly_table.empty:
        st.dataframe(
            monthly_table, 
            use_container_width=True,
            height=400
        )
    else:
        st.info("📊 No data available for monthly summary with current filters.")

def income_page(df):
    """Displays the enhanced Income Dashboard page."""
    st.markdown('<h1 class="main-header">📈 Income Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Reload button
    col_reload, col_info = st.columns([0.2, 0.8])
    with col_reload:
        if st.button("🔄 Reload Data", key="reload_income"):
            st.cache_data.clear()
            st.rerun()

    income_df = df[df['type'] == 'income'].copy() if 'type' in df.columns and 'income' in df['type'].unique() else pd.DataFrame(columns=df.columns)

    with col_info:
        if not income_df.empty:
            st.success(f"💰 Found {len(income_df):,} income transactions")

    type_filter, date_range, subtype_filter = create_compact_filters(income_df, "income")
    filtered_df = apply_filters(income_df, type_filter, date_range, subtype_filter)

    # COLLAPSIBLE Income Performance Metrics
    overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
    overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
    all_time_income = income_df['amount'].sum() if 'amount' in income_df.columns else 0
    filtered_income = filtered_df['amount'].sum() if 'amount' in filtered_df.columns else 0

    with st.expander("💎 Income Performance Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_enhanced_metric_card("💰 Net Worth Impact", format_currency(overall_income - overall_outcome), "metric-container")
        with col2:
            create_enhanced_metric_card("📈 Total Income", format_currency(all_time_income), "metric-container-income")
        with col3:
            create_enhanced_metric_card("🎯 Filtered Income", format_currency(filtered_income), "metric-container-income")
        with col4:
            avg_income = all_time_income / len(income_df) if len(income_df) > 0 else 0
            create_enhanced_metric_card("📊 Avg per Transaction", format_currency(avg_income), "metric-container")

    # Enhanced Charts
    st.markdown("---")
    st.subheader("📊 Income Deep Dive Analysis")

    # Main income trend with flexible periods
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    income_trend = create_flexible_trend_chart(filtered_df, "📈 Income Trends Over Time", "income_trend")
    st.plotly_chart(income_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Income breakdown charts
    col1_break, col2_break = st.columns(2)
    
    with col1_break:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        subtype_chart = create_sorted_bar_chart(filtered_df, 'subtype', "💼 Income by Source")
        st.plotly_chart(subtype_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_break:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        pie_chart = create_top3_pie_chart(filtered_df, 'subtype', "🥧 Top Income Sources")
        st.plotly_chart(pie_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Description analysis
    col3_desc, col4_detail = st.columns([1, 1])
    
    with col3_desc:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        desc_chart = create_sorted_bar_chart(filtered_df, 'description', "📝 Income by Description", 'h')
        st.plotly_chart(desc_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4_detail:
        st.subheader("📋 Recent Income Transactions")
        if not filtered_df.empty and all(col in filtered_df.columns for col in ['timestamp', 'subtype', 'description', 'amount']):
            income_display = filtered_df[['timestamp', 'subtype', 'description', 'amount']].copy()
            income_display = income_display.sort_values('timestamp', ascending=False).head(10)
            income_display['amount_formatted'] = income_display['amount'].apply(format_currency)
            income_display['date'] = income_display['timestamp'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                income_display[['date', 'subtype', 'description', 'amount_formatted']],
                use_container_width=True,
                height=400,
                column_config={
                    "date": "Date",
                    "subtype": "Source",
                    "description": "Description",
                    "amount_formatted": "Amount"
                }
            )
        else:
            st.info("📊 No income data available for detailed view.")

def outcome_page(df):
    """Displays the enhanced Outcome Dashboard page."""
    st.markdown('<h1 class="main-header">📉 Outcome Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Reload button
    col_reload, col_info = st.columns([0.2, 0.8])
    with col_reload:
        if st.button("🔄 Reload Data", key="reload_outcome"):
            st.cache_data.clear()
            st.rerun()

    outcome_df = df[df['type'] == 'outcome'].copy() if 'type' in df.columns and 'outcome' in df['type'].unique() else pd.DataFrame(columns=df.columns)
    if 'amount' in outcome_df.columns:
        outcome_df['amount_display'] = outcome_df['amount'].abs()

    with col_info:
        if not outcome_df.empty:
            st.error(f"💸 Found {len(outcome_df):,} outcome transactions")

    type_filter, date_range, subtype_filter = create_compact_filters(outcome_df, "outcome")
    filtered_df = apply_filters(outcome_df, type_filter, date_range, subtype_filter)
    if 'amount' in filtered_df.columns:
        filtered_df['amount_display'] = filtered_df['amount'].abs()

    # COLLAPSIBLE Outcome Analysis Metrics
    overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
    overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
    all_time_outcome = outcome_df['amount_display'].sum() if 'amount_display' in outcome_df.columns else 0
    filtered_outcome = filtered_df['amount_display'].sum() if 'amount_display' in filtered_df.columns else 0

    with st.expander("💸 Outcome Analysis Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_enhanced_metric_card("💰 Remaining Balance", format_currency(overall_income - overall_outcome), "metric-container")
        with col2:
            create_enhanced_metric_card("📉 Total Outcomes", format_currency(all_time_outcome), "metric-container-outcome")
        with col3:
            create_enhanced_metric_card("🎯 Filtered Outcomes", format_currency(filtered_outcome), "metric-container-outcome")
        with col4:
            avg_outcome = all_time_outcome / len(outcome_df) if len(outcome_df) > 0 else 0
            create_enhanced_metric_card("📊 Avg per Transaction", format_currency(avg_outcome), "metric-container")

    # Enhanced Charts
    st.markdown("---")
    st.subheader("📊 Outcome Pattern Analysis")

    # Main outcome trend with flexible periods
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    outcome_trend = create_flexible_trend_chart(filtered_df, "📉 Outcome Trends Over Time", "outcome_trend")
    st.plotly_chart(outcome_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Outcome breakdown
    col1_out, col2_out = st.columns(2)
    
    with col1_out:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        category_chart = create_sorted_bar_chart(filtered_df, 'subtype', "💳 Outcomes by Category")
        st.plotly_chart(category_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_out:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        outcome_pie = create_top3_pie_chart(filtered_df, 'subtype', "🥧 Top Outcome Categories")
        st.plotly_chart(outcome_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ADD THẢO MỘC ANALYSIS SECTION
    st.markdown("---")
    create_thao_moc_analysis(filtered_df)

    # Monthly outcome analysis
    st.markdown("---")
    col3_monthly, col4_top = st.columns(2)
    
    with col3_monthly:
        st.subheader("📅 Monthly Outcome Trends")
        if not filtered_df.empty and 'year_month' in filtered_df.columns:
            monthly_outcomes = filtered_df.groupby('year_month')['amount_display'].sum().reset_index()
            monthly_outcomes = monthly_outcomes.sort_values('year_month', ascending=False).head(12)
            monthly_outcomes['amount_formatted'] = monthly_outcomes['amount_display'].apply(format_currency)
            monthly_outcomes['month'] = monthly_outcomes['year_month'].astype(str)
            
            st.dataframe(
                monthly_outcomes[['month', 'amount_formatted']],
                use_container_width=True,
                height=350,
                column_config={
                    "month": "Month",
                    "amount_formatted": "Total Outcomes"
                }
            )

    with col4_top:
        st.subheader("🔝 Top Outcome Categories")
        if not filtered_df.empty and 'subtype' in filtered_df.columns:
            top_categories = filtered_df.groupby('subtype')['amount_display'].sum().sort_values(ascending=False).head(10)
            top_categories_df = pd.DataFrame({
                'Category': top_categories.index,
                'Amount': top_categories.values
            })
            top_categories_df['Amount_Formatted'] = top_categories_df['Amount'].apply(format_currency)
            
            st.dataframe(
                top_categories_df[['Category', 'Amount_Formatted']],
                use_container_width=True,
                height=350,
                column_config={
                    "Category": "Outcome Category",
                    "Amount_Formatted": "Total Amount"
                }
            )

def savings_page(df):
    """Displays the enhanced Savings Dashboard page."""
    st.markdown('<h1 class="main-header">🏦 Savings Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Reload button
    col_reload, col_info = st.columns([0.2, 0.8])
    with col_reload:
        if st.button("🔄 Reload Data", key="reload_savings"):
            st.cache_data.clear()
            st.rerun()

    savings_df = df[df['type'] == 'saving'].copy() if 'type' in df.columns and 'saving' in df['type'].unique() else pd.DataFrame(columns=df.columns)

    with col_info:
        if not savings_df.empty:
            st.info(f"🏦 Found {len(savings_df):,} savings transactions")

    type_filter, date_range, subtype_filter = create_compact_filters(savings_df, "savings")
    filtered_df = apply_filters(savings_df, type_filter, date_range, subtype_filter)

    # COLLAPSIBLE Savings Growth Metrics
    overall_income = df[df['type'] == 'income']['amount'].sum() if 'type' in df.columns else 0
    overall_outcome = abs(df[df['type'] == 'outcome']['amount'].sum()) if 'type' in df.columns else 0
    all_time_savings = savings_df['amount'].sum() if 'amount' in savings_df.columns else 0
    filtered_savings = filtered_df['amount'].sum() if 'amount' in filtered_df.columns else 0

    with st.expander("🏦 Savings Growth Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            create_enhanced_metric_card("💰 Net Financial Position", format_currency(overall_income - overall_outcome), "metric-container")
        with col2:
            create_enhanced_metric_card("🏦 Total Savings", format_currency(all_time_savings), "metric-container-saving")
        with col3:
            create_enhanced_metric_card("🎯 Filtered Savings", format_currency(filtered_savings), "metric-container-saving")
        with col4:
            savings_rate = (all_time_savings / overall_income * 100) if overall_income > 0 else 0
            create_enhanced_metric_card("📊 Savings Rate", f"{savings_rate:.1f}%", "metric-container")

    # Enhanced Charts
    st.markdown("---")
    st.subheader("📊 Savings Performance Analysis")

    # Savings trend with flexible periods
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    savings_trend = create_flexible_trend_chart(filtered_df, "🏦 Savings Growth Trends", "savings_trend")
    st.plotly_chart(savings_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Savings breakdown
    col1_sav, col2_sav = st.columns(2)
    
    with col1_sav:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        savings_bar = create_sorted_bar_chart(filtered_df, 'subtype', "💼 Savings by Type")
        st.plotly_chart(savings_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_sav:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        savings_pie = create_top3_pie_chart(filtered_df, 'subtype', "🥧 Top Savings Types")
        st.plotly_chart(savings_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Savings goals and monthly analysis
    col3_goals, col4_monthly = st.columns(2)
    
    with col3_goals:
        st.subheader("🎯 Savings Goals Analysis")
        if not filtered_df.empty and 'subtype' in filtered_df.columns:
            savings_by_type = filtered_df.groupby('subtype')['amount'].sum().sort_values(ascending=False)
            
            # Create a simple progress-like visualization
            fig_goals = go.Figure()
            
            colors_list = [COLORS['saving'], COLORS['income'], COLORS['primary']]
            
            for i, (subtype, amount) in enumerate(savings_by_type.items()):
                fig_goals.add_trace(go.Bar(
                    y=[subtype],
                    x=[amount],
                    orientation='h',
                    name=subtype.title(),
                    marker=dict(color=colors_list[i % len(colors_list)]),
                    text=format_currency(amount),
                    textposition='inside'
                ))
            
            fig_goals.update_layout(
                title="Savings by Goal Type",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_goals, use_container_width=True)

    with col4_monthly:
        st.subheader("📅 Monthly Savings Summary")
        if not filtered_df.empty and 'year_month' in filtered_df.columns:
            monthly_savings = filtered_df.groupby('year_month')['amount'].sum().reset_index()
            monthly_savings = monthly_savings.sort_values('year_month', ascending=False).head(12)
            monthly_savings['amount_formatted'] = monthly_savings['amount'].apply(format_currency)
            monthly_savings['month'] = monthly_savings['year_month'].astype(str)
            
            st.dataframe(
                monthly_savings[['month', 'amount_formatted']],
                use_container_width=True,
                height=300,
                column_config={
                    "month": "Month",
                    "amount_formatted": "Savings Amount"
                }
            )

def main():
    """Main application entry point with enhanced styling."""
    # Load data
    df = load_data()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 💰 Enhanced Financial Dashboard")
        st.markdown("---")
        
        # Navigation with enhanced styling
        page = st.selectbox(
            "🧭 Navigate to:",
            ["🏠 Overview", "📈 Income", "📉 Outcome", "🏦 Savings"],
            help="Select a dashboard view"
        )
        
        # Data summary in sidebar
        st.markdown("---")
        st.markdown("### 📊 Data Insights")
        
        if not df.empty:
            total_transactions = len(df)
            date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
            
            if 'type' in df.columns:
                income_count = len(df[df['type'] == 'income'])
                outcome_count = len(df[df['type'] == 'outcome'])
                savings_count = len(df[df['type'] == 'saving'])
                
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
            
        st.markdown("---")
        st.markdown("*💡 Use filters on each page to customize your analysis*")
        
        # Enhanced Features Section
        st.markdown("---")
        st.markdown("### 🚀 Enhanced Features")
        st.markdown("""
        - **🎯 Smart Filters**: Quick date range selection
        - **📊 Flexible Charts**: Multiple period views (Daily/Weekly/Monthly/etc.)
        - **🥧 Top 3 Focus**: Simplified pie charts showing key categories
        - **📱 Mobile Friendly**: Responsive design for all devices
        - **🎨 Modern UI**: Enhanced styling with gradients and effects
        - **⚡ Performance**: Optimized data loading and processing
        - **🌿 Thảo Mộc Analysis**: Special deep-dive for herbal medicine expenses
        - **📊 Collapsible Metrics**: Expandable metric cards for better space management
        """)
    
    # Display selected page
    if page == "🏠 Overview":
        overview_page(df)
    elif page == "📈 Income":
        income_page(df)
    elif page == "📉 Outcome":
        outcome_page(df)
    elif page == "🏦 Savings":
        savings_page(df)

if __name__ == "__main__":
    main()