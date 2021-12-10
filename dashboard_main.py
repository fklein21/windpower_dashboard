from io import SEEK_CUR
import pandas as pd
import numpy as np

import dash
from datetime import date
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px

from dash.dependencies import Input, Output

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

################################################################################
# HELPER FUNCTIONS
################################################################################
def degrees_to_cardinal(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((d + 11.25)/22.5)
    return dirs[ix % 16]

def card_sorter(column):
    """Sort function"""
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    correspondence = {card: order for order, card in enumerate(wd_card)}
    return column.map(correspondence)

## get the required data
def get_wind_forecast():
    pass

################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

# this is needed by gunicorn command in procfile
server = app.server

################################################################################
# STYLE
################################################################################
# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}
################################################################################
# PLOTS
################################################################################

## get color scheme right
colors = px.colors.qualitative.Plotly
color_dict = {'Zone '+str(z): c for z,c in zip(range(1,11), colors)}

def get_figure_24h(dff, selected_zone, selected_hour=0):
    tmin, tmax = 1,24
    fig = go.Figure()
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    for column in selected_zone:
        color = color_dict[column]
        fig.add_traces(go.Scatter(x=dff['HOUR'], y = df[column], 
            mode = 'lines', line=dict(color=color), name=column)
        )
    fig.update_xaxes(range = [tmin, tmax])
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    fig.layout.showlegend = True
    fig.add_shape(type="line",
        x0=selected_hour, y0=0, x1=selected_hour, y1=1,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    fig.update_layout( 
            title='Forecast Power Output over 24 hours')
    return fig

## get figure for the energy per hour graph 
def get_figure_energy_per_hour(df, selected_zone, selected_hour):
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    df_hour = df[selected_zone]
    df_hour = pd.DataFrame(df_hour.loc[selected_hour:selected_hour])
    cols = df_hour.columns
    cols = [cc for cc in cols if cc.startswith('Zone')]
    dff = df_hour[cols]
    dff = dff.T
    bars = []
    fig = go.Figure()
    for zone in selected_zone:
        color = color_dict[zone]
        fig.add_traces(
            go.Bar(x=[zone], y=[dff.loc[zone][dff.columns[-1]]], 
                marker={'color': color}, showlegend=False)
    )
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    fig.update_layout( 
            title='Forecast Power Output at hour '+str(selected_hour))
    return fig

## wind rose
def plot_windrose(df, selected_zone_nr=1, show_legend=False, show_title=True):
    df = df[df['ZONEID']==selected_zone_nr]
    bins = np.linspace(0,24,13)
    labels = range(0,23,2)

    df['WS100BIN'] = pd.cut(df['WS100'], bins=bins, labels = labels)
    df_grouped = df.groupby(['WD100CARD','WS100BIN']).count().reset_index()
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    wd_zeros = np.zeros(len(wd_card))
    df_all_wd_card = pd.DataFrame([wd_card, wd_zeros, wd_zeros])
    df_all_wd_card = df_all_wd_card.T
    df_all_wd_card.columns = ['WD100CARD', 'WS100BIN', 'FREQUENCY']
    
    data_wind = df_grouped[['WD100CARD', 'WS100BIN', 'TIMESTAMP']]
    data_wind.columns = df_all_wd_card.columns

    datax = pd.concat([data_wind, df_all_wd_card], axis = 0)
    datax = datax.sort_values(by='WD100CARD', key=card_sorter)
    ws_ls = np.arange(0,24,2)
    wind_all_speeds = pd.DataFrame([['N']*len(ws_ls), ws_ls, np.zeros(len(ws_ls))]).T
    wind_all_speeds.columns = ['WD100CARD', 'WS100BIN', 'FREQUENCY']
    wind_all_speeds
    datax = pd.concat([wind_all_speeds, df_all_wd_card, data_wind], axis = 0)

    fig = px.bar_polar(datax, 
        r="FREQUENCY", 
        theta="WD100CARD",
        color="WS100BIN", 
        color_discrete_sequence= px.colors.sequential.Plasma_r,
    )                     
    fig.layout.showlegend = show_legend
    fig.update_layout(
        autosize=True,
        width=300,
        height=300,
        margin=dict(
            l=30,
            r=30,
            b=30,
            t=50,
            pad=4
        )
    )
    if show_title:
        fig.update_layout( 
            title='Zone '+str(selected_zone_nr)
        )
    return fig

################################################################################
# LAYOUT
################################################################################
day='2013-01-01'
filename = 'prediction_for_dashboard.csv'
filename_wind = 'prediction_wind_for_dashboard.csv'
# df, df_wind = make_prediction(day)
# df['HOUR'] = df['TIMESTAMP'].dt.hour
# df_wind['WD100CARD'] = df_wind.WD100.apply(lambda x: degrees_to_cardinal(x))
# print(df_wind.head())
# df.to_csv(filename,index=False)
# df_wind.to_csv(filename_wind,index=False)
df = pd.read_csv(filename)
df_wind = pd.read_csv(filename_wind)
# df_targetvar_yesterday ...


def Header(name, app):
    title = html.H1(name, style={"margin-top": 30, "margin-bottom": 30, "margin-left":30, "margin-right":30})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")
    return dbc.Row([dbc.Col(title, md=8),],justify="center")

controls = dbc.Card(
    [
        html.Div(
            children=[
                html.Div( 
                    dbc.Label("Date\t"),
                ),
                html.Div( 
                    dcc.DatePickerSingle(
                        id='date-picker-single',
                        min_date_allowed=date(2013, 7, 1),
                        max_date_allowed=date(2013, 12, 31),
                        initial_visible_month=date(2013, 7, 5),
                        date=date(2013, 7, 1)
                    ),
                ),
            ]
        ),
        html.Div(''),
        html.Hr(),
        html.Div(''),
        html.Div(
            [
                dbc.Label("Zone selection:"),
                dbc.Checklist(
                    options=[
                        {"label": "Zone 1", "value": "Zone 1"},
                        {"label": "Zone 2", "value": "Zone 2"},
                        {"label": "Zone 3", "value": "Zone 3"},
                        {"label": "Zone 4", "value": "Zone 4"},
                        {"label": "Zone 5", "value": "Zone 5"},
                        {"label": "Zone 6", "value": "Zone 6"},
                        {"label": "Zone 7", "value": "Zone 7"},
                        {"label": "Zone 8", "value": "Zone 8"},
                        {"label": "Zone 9", "value": "Zone 9"},
                        {"label": "Zone 10", "value": "Zone 10"},
                    ],
                    value=["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5",
                        "Zone 6", "Zone 7", "Zone 8", "Zone 9", "Zone 10", ],
                    id="zone-selector",
                ),
            ]
        ),
    ],
    body=True,
)
## sidebar  with controls for date, zone
sidebar = html.Div(
    [
        html.H2('Parameter', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    #style=SIDEBAR_STYLE,
)

title_tab_1 = html.Div('Projected Energy Output', id="title-tab-1")
figure_energy_24h = dcc.Graph(id='figure-energy-24h')
slider_hour = dcc.Slider( 
    id='slider-hour',
    min=1,
    max=23,
    value=1,
    marks={str(hh): str(hh) for hh in range(1,24)},
    step=None
)

figure_energy_per_hour = dcc.Graph(id="figure-energy-per-hour")
maincontent_tab_1 = dbc.Container( 
    [
        dbc.Row( 
            [ 
                dbc.Col( 
                    [ 
                        sidebar,
                    ], md=3,
                ),
                dbc.Col( 
                    [
                        figure_energy_24h,
                        slider_hour,
                        figure_energy_per_hour
                    ], md=9,
                )
            ]
        )
    ], fluid=True,
)

maincontent_tab_2 = dbc.Container( 
    [   
        html.Div(""),
        dbc.Row( 
            [ 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 1"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-1", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 4"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-4", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 7"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-7", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 2"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-2", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 5"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-5", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 8"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-8", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 10"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-10", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 3"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-3", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 6"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-6", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Zone 9"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-9", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
            ]
        )
    ]
)

app.layout = html.Div( 
    [
        Header("Energy Output Forecast for the Next 24 Hours", app),
        html.Hr(),
        dbc.Tabs( 
            [ 
                dbc.Tab(maincontent_tab_1, label="Forecast Energy Output", tab_id="tab-energy-forecast"),
                dbc.Tab(maincontent_tab_2, label="Wind Strength and Direction", tab_id="tab-wind-roses"),
            ],
            id="tabs",
            active_tab="tab-energy-forecast"
        )
    ]
)
################################################################################
# INTERACTION CALLBACKS
################################################################################

@app.callback(
    Output('figure-energy-24h', "figure"),
    Input('zone-selector', 'value'),
    Input('slider-hour', 'value'))
def update_graphs(selected_zone, selected_hour):
    cols = selected_zone.copy()
    cols.append('TIMESTAMP')
    return get_figure_24h(df, selected_zone, selected_hour)

@app.callback(
    Output('figure-energy-per-hour', 'figure'),
    Input('zone-selector', 'value'),
    Input('slider-hour', 'value'))
def update_figure(selected_zone, selected_hour):
    return get_figure_energy_per_hour(df, selected_zone, selected_hour)

@app.callback(
    Output('wind-rose-1', 'figure'),
    Output('wind-rose-2', 'figure'),
    Output('wind-rose-3', 'figure'),
    Output('wind-rose-4', 'figure'),
    Output('wind-rose-5', 'figure'),
    Output('wind-rose-6', 'figure'),
    Output('wind-rose-7', 'figure'),
    Output('wind-rose-8', 'figure'),
    Output('wind-rose-9', 'figure'),
    Output('wind-rose-10', 'figure'),
    Input('date-picker-single', 'date'))
def update_figure_windrose(date):
    print('date:',date)
    return plot_windrose(df_wind, 1, show_title=False),\
           plot_windrose(df_wind, 2, show_title=False),\
           plot_windrose(df_wind, 3, show_title=False),\
           plot_windrose(df_wind, 4, show_title=False),\
           plot_windrose(df_wind, 5, show_title=False),\
           plot_windrose(df_wind, 6, show_title=False),\
           plot_windrose(df_wind, 7, show_title=False),\
           plot_windrose(df_wind, 8, show_title=False),\
           plot_windrose(df_wind, 9, show_title=False),\
           plot_windrose(df_wind, 10, show_title=False)


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
