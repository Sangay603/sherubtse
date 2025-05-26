# Import necessary libraries
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import timedelta, datetime, date
import base64
import os
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# --- Data Loading and Cleaning ---
# This part is adapted from the code you provided
files = {
    "BNBL": "Historical_Data_BNBL.csv",
    "BBPL": "Historical_Data_BBPL.csv",
    "DPNB": "Historical_Data_DPNB.csv",
    "BIL": "Historical_Data_BIL.csv",
    "BPCL": "Historical_Data_BPCL.csv",
    "DFAL": "Historical_Data_DFAL.csv",
    "DPL": "Historical_Data_DPL.csv",
    "DWAL": "Historical_Data_DWAL.csv",
    "GICB": "Historical_Data_GICB.csv",
    "KCL": "Historical_Data_KCL.csv",
    "PCAL": "Historical_Data_PCAL.csv",
    "RICB": "Historical_Data_RICB.csv",
    "STCB": "Historical_Data_STCB.csv",
    "SVL": "Historical_Data_SVL.csv",
    "TBL": "Historical_Data_TBL.csv",
    "BTCL": "Historical_Data_BTCL.csv"
}

def clean_dataframe(df):
    if df.empty or len(df) <= 1:
        return pd.DataFrame(columns=['Date', 'Market Price'])

    try:
        df.columns = df.iloc[0].tolist()
    except Exception as e:
        print(f"Warning: Could not set columns from first row. Error: {e}")
        df = df[1:].reset_index(drop=True)
        df.columns = df.columns.str.strip()

    if len(df) > 1:
        df = df[1:].reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['Date', 'Market Price'])

    df.columns = df.columns.str.strip()

    if 'Date' not in df.columns or 'Market Price' not in df.columns:
        print(f"Warning: 'Date' or 'Market Price' column not found after cleaning.")
        return pd.DataFrame(columns=['Date', 'Market Price'])

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Market Price'] = pd.to_numeric(df['Market Price'], errors='coerce')

    return df[['Date', 'Market Price']].dropna(subset=['Date', 'Market Price'])

# Read, clean, and store cleaned dataframes
cleaned_dataframes = []
for name, path in files.items():
    try:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            continue

        df_raw = pd.read_csv(path)
        df_cleaned = clean_dataframe(df_raw)

        if not df_cleaned.empty:
            df_cleaned['Company'] = name
            cleaned_dataframes.append(df_cleaned)
        else:
            print(f"Warning: No valid data found in {path} after cleaning.")

    except Exception as e:
        print(f"Error processing file {path}: {e}")

# Concatenate all cleaned dataframes
df = pd.DataFrame()
if cleaned_dataframes:
    df = pd.concat(cleaned_dataframes, ignore_index=True)
    print("Data loaded and combined into a single DataFrame 'df'.")
else:
    print("No dataframes were successfully loaded and cleaned.")

# Ensure proper date handling
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    available_companies = sorted(df['Company'].unique())
else:
    min_date = pd.Timestamp('2000-01-01')
    max_date = pd.Timestamp('2030-12-31')
    available_companies = []

# --- Dashboard Layout ---
app = dash.Dash(
    __name__, 
    assets_folder='assets',
    title='Sherubtse College Stock Market Dashboard',
    suppress_callback_exceptions=True
)

# Update the controls section with improved date pickers
controls_section = html.Div([
    html.Div([
        html.Label([
            html.I(className="fas fa-building"), " Select Company"
        ]),
        dcc.Dropdown(
            id='company-dropdown',
            options=[{'label': company, 'value': company} for company in available_companies],
            value=available_companies[0] if available_companies else None,
            clearable=False,
            className="custom-dropdown",
            placeholder="Select a company..."
        )
    ], className="control-item"),

    html.Div([
        html.Label([
            html.I(className="fas fa-calendar"), " Start Date"
        ]),
        dcc.DatePickerSingle(
            id='start-date-picker',
            min_date_allowed=pd.Timestamp('2000-01-01'),
            max_date_allowed=pd.Timestamp('2030-12-31'),
            initial_visible_month=pd.Timestamp('2020-01-01'),
            date=pd.Timestamp('2020-01-01'),
            display_format='YYYY-MM-DD',
            className="custom-datepicker",
            calendar_orientation='vertical',
            show_outside_days=True,
            first_day_of_week=1
        )
    ], className="control-item"),

    html.Div([
        html.Label([
            html.I(className="fas fa-calendar-alt"), " Forecast Until"
        ]),
        dcc.DatePickerSingle(
            id='forecast-date-picker',
            min_date_allowed=pd.Timestamp('2000-01-01'),
            max_date_allowed=pd.Timestamp('2050-12-31'),
            initial_visible_month=pd.Timestamp('2025-01-01'),
            date=pd.Timestamp('2025-01-01'),
            display_format='YYYY-MM-DD',
            className="custom-datepicker",
            calendar_orientation='vertical',
            show_outside_days=True,
            first_day_of_week=1
        )
    ], className="control-item"),

    # Add date range info
    html.Div([
        html.P([
            html.I(className="fas fa-info-circle"), 
            " You can select dates from 2000 to 2050"
        ], className="date-range-info")
    ], className="control-item info")
], className="controls-container")

# Market Analysis Section
market_analysis_section = html.Div([
    html.Div([
        html.H2([html.I(className="fas fa-chart-line"), " Market Analysis"], className="section-title"),
        html.Div([
            html.Button([
                html.I(className="fas fa-sync-alt"), " Refresh Data"
            ], id="refresh-button", className="refresh-button"),
            html.Div(id="last-update-time", className="last-update")
        ], className="market-header-right")
    ], className="market-header"),
    
    html.Div(id='market-analysis-content', className="market-analysis-container")
], id="section-market", className="dashboard-section", style={'display': 'none'})

# Update the app layout
app.layout = html.Div([
    html.Div([
        # Sidebar
        html.Aside([
            html.Div([
                html.Img(src='/assets/images/logo.png', className="college-logo", alt="Sherubtse College Logo"),
                html.H2("Sherubtse College")
            ], className="sidebar-header"),
            html.Nav([
                html.A([html.I(className="fas fa-home"), " Home"], href="#", id="nav-home", className="nav-link active"),
                html.A([html.I(className="fas fa-chart-line"), " Market Analysis"], href="#", id="nav-market", className="nav-link"),
                html.A([html.I(className="fas fa-file-alt"), " Reports"], href="#", id="nav-reports", className="nav-link"),
                html.A([html.I(className="fas fa-cog"), " Settings"], href="#", id="nav-settings", className="nav-link")
            ])
        ], className="sidebar"),

        # Main Content
        html.Main([
            html.Div([
                html.Div([
                    html.Div([
                        html.H1("Stock Market Price Forecast"),
                        html.Div([
                            html.Img(src='/assets/images/logo.png', className="header-logo", alt="Sherubtse College Logo"),
                            html.Div([
                                dcc.Input(
                                    type="text",
                                    placeholder="Search stocks...",
                                    className="search-input"
                                ),
                                html.Button([
                                    html.I(className="fas fa-search"), " Search"
                                ], className="search-button")
                            ], className="search-bar")
                        ], className="header-right")
                    ], className="header-row")
                ], className="header"),

                dcc.Store(id='current-section', data='home'),

                # Home Section
                html.Div([
                    controls_section,
                    html.Div([
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-chart-area"), " Market Price Forecast"
                            ], className="graph-title"),
                            dcc.Loading(
                                id="loading-graph",
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id='forecast-graph',
                                        config={
                                            'displayModeBar': True,
                                            'displaylogo': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'scrollZoom': True
                                        }
                                    )
                                ]
                            ),
                            html.Div(id='forecast-report', className="report-content")
                        ], className="graph-content")
                    ], className="graph-container"),
                ], id="section-home", className="dashboard-section"),

                # Market Analysis Section
                market_analysis_section,

                # Reports Section
                html.Div([
                    html.H2([html.I(className="fas fa-file-alt"), " Reports"]),
                    html.Div([
                        html.Div([
                            html.H3([html.I(className="fas fa-book"), " Annual Reports"]),
                            html.P("Access official annual reports and financial statements of listed companies."),
                            html.Div(className="annual-reports-grid", children=[
                                # BPCL Reports
                                html.Div([
                                    html.H4([html.I(className="fas fa-building"), " BPCL"]),
                                    html.Div(className="report-years", children=[
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2023"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/BPCL_2023.pdf", target="_blank"),
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2022"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/BPCL_2022.pdf", target="_blank")
                                    ])
                                ], className="company-section"),

                                # BTCL Reports
                                html.Div([
                                    html.H4([html.I(className="fas fa-building"), " BTCL"]),
                                    html.Div(className="report-years", children=[
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2023"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/BTCL_2023.pdf", target="_blank"),
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2022"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/BTCL_2022.pdf", target="_blank")
                                    ])
                                ], className="company-section"),

                                # DFAL Reports
                                html.Div([
                                    html.H4([html.I(className="fas fa-building"), " DFAL"]),
                                    html.Div(className="report-years", children=[
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2023"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/DFAL_2023.pdf", target="_blank"),
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2022"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/DFAL_2022.pdf", target="_blank")
                                    ])
                                ], className="company-section"),

                                # DPL Reports
                                html.Div([
                                    html.H4([html.I(className="fas fa-building"), " DPL"]),
                                    html.Div(className="report-years", children=[
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2023"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/DPL_2023.pdf", target="_blank"),
                                        html.A([
                                            html.Div(className="report-year-card", children=[
                                                html.I(className="far fa-file-pdf"),
                                                html.H5("2022"),
                                                html.P("Annual Report")
                                            ])
                                        ], href="annual_reports/DPL_2022.pdf", target="_blank")
                                    ])
                                ], className="company-section")
                            ])
                        ], className="reports-container")
                    ], className="reports-grid")
                ], id="section-reports", className="dashboard-section", style={'display': 'none'}),

                # Settings Section
                html.Div([
                    html.H2([html.I(className="fas fa-cog"), " Settings"]),
                    html.Div([
                        html.Div([
                            html.H3("Display Settings"),
                            html.Div([
                                html.Label("Theme"),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Dark', 'value': 'dark'},
                                        {'label': 'Light', 'value': 'light'}
                                    ],
                                    value='dark',
                                    className="settings-dropdown"
                                )
                            ], className="settings-item")
                        ], className="settings-card")
                    ], className="settings-grid")
                ], id="section-settings", className="dashboard-section", style={'display': 'none'}),

                # Footer
                html.Footer([
                    html.Div([
                        html.Img(src='/assets/images/logo.png', className="footer-logo", alt="Sherubtse College Logo"),
                        html.Div([
                            html.P("© 2024 Sherubtse College Stock Market Forecast Dashboard"),
                            html.P("Developed with ❤️ by Department of Mathematics & Computer Science")
                        ], className="footer-text")
                    ], className="footer-content")
                ], className="footer")
            ], className="main-content")
        ])
    ])
])

# Calculate performance metrics for companies
def calculate_performance_metrics(df, company):
    if df.empty:
        return {
            'current_price': 0,
            'change_1d': 0,
            'change_1w': 0,
            'change_1m': 0,
            'trend': 'neutral',
            'last_update': 'No data'
        }
    
    df = df.sort_values('Date')
    current_price = df['Market Price'].iloc[-1]
    last_update = df['Date'].iloc[-1]
    
    try:
        price_1d_ago = df[df['Date'] <= (last_update - timedelta(days=1))]['Market Price'].iloc[-1]
        change_1d = ((current_price - price_1d_ago) / price_1d_ago) * 100
    except:
        change_1d = 0
        
    try:
        price_1w_ago = df[df['Date'] <= (last_update - timedelta(days=7))]['Market Price'].iloc[-1]
        change_1w = ((current_price - price_1w_ago) / price_1w_ago) * 100
    except:
        change_1w = 0
        
    try:
        price_1m_ago = df[df['Date'] <= (last_update - timedelta(days=30))]['Market Price'].iloc[-1]
        change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
    except:
        change_1m = 0
    
    if change_1w > 2:
        trend = 'bullish'
    elif change_1w < -2:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    return {
        'current_price': current_price,
        'change_1d': change_1d,
        'change_1w': change_1w,
        'change_1m': change_1m,
        'trend': trend,
        'last_update': last_update.strftime('%Y-%m-%d')
    }

@app.callback(
    Output('market-analysis-content', 'children'),
    [Input('refresh-button', 'n_clicks')]
)
def update_market_analysis(n_clicks):
    company_metrics = {}
    for company in available_companies:
        company_df = df[df['Company'] == company]
        company_metrics[company] = calculate_performance_metrics(company_df, company)
    
    return html.Div([
        html.Div([
            html.Div([
                html.H3(company, className="company-title"),
                html.Div([
                    html.Div([
                        html.Span("Nu. ", className="currency-symbol"),
                        html.Span(f"{metrics['current_price']:.2f}", className="price-value")
                    ], className="current-price"),
                    html.Div([
                        html.Span(
                            f"{metrics['change_1d']:+.2f}%",
                            className=f"price-change {'positive' if metrics['change_1d'] > 0 else 'negative' if metrics['change_1d'] < 0 else 'neutral'}"
                        ),
                        html.Span(" (24h)", className="time-period")
                    ], className="price-change-container")
                ], className="price-info"),
                html.Div([
                    html.Div([
                        html.Span("7D: "),
                        html.Span(
                            f"{metrics['change_1w']:+.2f}%",
                            className=f"price-change {'positive' if metrics['change_1w'] > 0 else 'negative' if metrics['change_1w'] < 0 else 'neutral'}"
                        )
                    ], className="metric"),
                    html.Div([
                        html.Span("30D: "),
                        html.Span(
                            f"{metrics['change_1m']:+.2f}%",
                            className=f"price-change {'positive' if metrics['change_1m'] > 0 else 'negative' if metrics['change_1m'] < 0 else 'neutral'}"
                        )
                    ], className="metric"),
                    html.Div([
                        html.I(
                            className=f"fas {'fa-arrow-up' if metrics['trend'] == 'bullish' else 'fa-arrow-down' if metrics['trend'] == 'bearish' else 'fa-minus'}",
                            style={'color': '#00ff00' if metrics['trend'] == 'bullish' else '#ff0000' if metrics['trend'] == 'bearish' else '#888888'}
                        ),
                        html.Span(f" {metrics['trend'].capitalize()}", className="trend-text")
                    ], className="trend")
                ], className="metrics-container"),
                html.Div([
                    html.I(className="far fa-clock"), 
                    f" Last Updated: {metrics['last_update']}"
                ], className="last-update")
            ], className=f"company-card {metrics['trend']}")
        for company, metrics in company_metrics.items()], className="company-grid")
    ])

# Add callback for navigation
@app.callback(
    [Output('section-home', 'style'),
     Output('section-market', 'style'),
     Output('section-reports', 'style'),
     Output('section-settings', 'style'),
     Output('nav-home', 'className'),
     Output('nav-market', 'className'),
     Output('nav-reports', 'className'),
     Output('nav-settings', 'className')],
    [Input('nav-home', 'n_clicks'),
     Input('nav-market', 'n_clicks'),
     Input('nav-reports', 'n_clicks'),
     Input('nav-settings', 'n_clicks')],
    [State('current-section', 'data')]
)
def update_section(home_clicks, market_clicks, reports_clicks, settings_clicks, current_section):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default to home section
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                'nav-link active', 'nav-link', 'nav-link', 'nav-link']
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize all sections as hidden and nav links as inactive
    sections = [{'display': 'none'} for _ in range(4)]
    nav_classes = ['nav-link' for _ in range(4)]
    
    # Show the selected section and activate its nav link
    if button_id == 'nav-home':
        sections[0] = {'display': 'block'}
        nav_classes[0] = 'nav-link active'
    elif button_id == 'nav-market':
        sections[1] = {'display': 'block'}
        nav_classes[1] = 'nav-link active'
    elif button_id == 'nav-reports':
        sections[2] = {'display': 'block'}
        nav_classes[2] = 'nav-link active'
    elif button_id == 'nav-settings':
        sections[3] = {'display': 'block'}
        nav_classes[3] = 'nav-link active'
    
    return sections + nav_classes

# --- Callbacks for Interactivity ---
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-report', 'children')],
    [Input('company-dropdown', 'value'),
     Input('start-date-picker', 'date'),
     Input('forecast-date-picker', 'date')]
)
def update_graph_and_report(selected_company, start_date_str, forecast_until_str):
    if not all([selected_company, start_date_str, forecast_until_str]):
        return go.Figure(layout=dict(template="plotly_dark")), []

    try:
        # Convert dates
        start_date = pd.to_datetime(start_date_str)
        forecast_until = pd.to_datetime(forecast_until_str)

        # Filter data
        df_company = df[df['Company'] == selected_company]
        df_filtered = df_company[df_company['Date'] >= start_date]

        if df_filtered.empty:
            return (
                go.Figure(layout=go.Layout(
                    title=f"No data available for {selected_company} from {start_date_str}",
                    template="plotly_dark"
                )),
                []
            )

        # Prepare data for Prophet
        df_model = df_filtered[['Date', 'Market Price']].rename(columns={'Date': 'ds', 'Market Price': 'y'})
        
        # Train Prophet model with optimized parameters
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            changepoint_range=0.9,
            interval_width=0.95
        )
        model.fit(df_model)

        # Create future dates
        future_dates = pd.date_range(
            start=df_model['ds'].min(),
            end=forecast_until,
            freq='D'
        )
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = model.predict(future)

        # Create figure
        fig = go.Figure()

        # Add historical data
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Market Price'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=1),
            marker=dict(size=4)
        ))

        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#2196f3', width=2)
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(33, 150, 243, 0.1)',
            line=dict(width=0),
            name='Confidence Interval'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Market Price Forecast for {selected_company}",
                font=dict(size=24)
            ),
            xaxis_title="Date",
            yaxis_title="Market Price",
            template="plotly_dark",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )

        # Add range slider and buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )

        # Generate report cards
        last_value = df_filtered['Market Price'].iloc[-1]
        forecast_end = forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_end - last_value) / last_value) * 100
        
        avg_historical = df_filtered['Market Price'].mean()
        avg_forecast = forecast['yhat'].mean()
        trend_direction = "positive" if forecast_change > 0 else "negative"
        
        max_forecast = forecast['yhat_upper'].max()
        min_forecast = forecast['yhat_lower'].min()
        volatility = forecast['yhat'].std()

        report_cards = [
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line"),
                    html.H4("Forecast Change")
                ], className="report-card-header"),
                html.Div(f"{forecast_change:.2f}%", className="report-value"),
                html.Div([
                    html.I(className=f"fas fa-arrow-{'up' if forecast_change > 0 else 'down'}"),
                    html.Span(f"{'Increase' if forecast_change > 0 else 'Decrease'} from current price")
                ], className=f"report-change {trend_direction}")
            ], className="report-card"),

            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-bar"),
                    html.H4("Price Range")
                ], className="report-card-header"),
                html.Div([
                    html.Div(f"Max: {max_forecast:.2f}", className="report-value"),
                    html.Div(f"Min: {min_forecast:.2f}", className="report-value")
                ])
            ], className="report-card"),

            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-area"),
                    html.H4("Market Volatility")
                ], className="report-card-header"),
                html.Div(f"{volatility:.2f}", className="report-value"),
                html.Div("Standard deviation of forecast", className="report-subtitle")
            ], className="report-card"),

            html.Div([
                html.Div([
                    html.I(className="fas fa-balance-scale"),
                    html.H4("Average Prices")
                ], className="report-card-header"),
                html.Div([
                    html.Div(f"Historical: {avg_historical:.2f}", className="report-value"),
                    html.Div(f"Forecast: {avg_forecast:.2f}", className="report-value")
                ])
            ], className="report-card")
        ]

        return fig, report_cards

    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        return (
            go.Figure(layout=go.Layout(
                title=f"Error generating forecast: {str(e)}",
                template="plotly_dark"
            )),
            []
        )

# Add callback for refresh button
@app.callback(
    [Output("last-update-time", "children")],
    [Input("refresh-button", "n_clicks")]
)
def refresh_market_data(n_clicks):
    # Update last refresh time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [f"Last Updated: {current_time}"]

# --- Run the app ---
if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 