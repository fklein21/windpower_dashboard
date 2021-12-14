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

from PIL import ImageColor

from dash.dependencies import Input, Output

pd.options.mode.chained_assignment = None  # default='warn'

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


################################################################################
# DEFAULT VALUES FOR FILE NAMES, ...
################################################################################

features_forecast = ['ZONEID', 'TIMESTAMP', 'HOUR', 'WEEKDAY', 'WS100', 'WD100']
features_retro = ['ZONEID', 'TIMESTAMP', 'HOUR', 'WEEKDAY', 'TARGETVAR']

PATH_PREDICTIONS = 'RandomForest_Predictions.csv'
PATH_DATA_ALL = 'raw_data_incl_features_test.csv'

initial_date = '2013-07-01'

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

def add_HOUR_column(df):
    if 'HOUR' not in df.columns and 'TIMESTAMP' in df.columns:
        df['HOUR'] = df['TIMESTAMP'].dt.hour
    if 'HOUR' in df.columns:
        df['HOUR'].replace({0 : 24}, inplace=True)

## get the required data
def get_data(day, 
             data,
             features=None,
             delta_days=None):
    day +=' 1:00:00'
    time_start = pd.to_datetime(day)
    time_end = time_start + pd.Timedelta(value=23, unit='hour')
    if delta_days is not None:
        time_delta = pd.Timedelta(value=1, unit='day')
        time_start += time_delta*delta_days
        time_end += time_delta*delta_days
    data = data[(data['TIMESTAMP']>=time_start) & (data['TIMESTAMP']<=time_end)]
    if features is not None and len(features)>0:
        if all(x in data.columns for x in features):
            data = data[features]
    return data

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
color_dict = {'Wind Farm '+str(z): c for z,c in zip(range(1,11), colors)}
color_dict_light = { k:ImageColor.getcolor(v, "RGB") for k,v in color_dict.items()}
color_dict_light = { k:"rgba({}, {}, {}, {})".format(v[0], v[1], v[2], 0.4) for k,v in color_dict_light.items()}


def get_figure_windspeed(df, selected_zone, selected_hour=1):
    tmin, tmax = 1,24
    fig = go.Figure()
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    
    for column in selected_zone:
        df_zone = df[df['ZONEID'] == column]
        fig.add_traces(go.Scatter(x=df_zone['HOUR'], y = df_zone['WS100'], 
            mode = 'lines', line=dict(color=color_dict[column]), name=column)
        )
    fig.update_xaxes(range = [tmin, tmax])
    fig.update_yaxes(range = [0,24])
    fig.layout.template = 'plotly_white'
    fig.layout.showlegend = True
    fig.add_shape(type="line",
        x0=selected_hour, y0=0, x1=selected_hour, y1=24,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    #fig.update_layout( 
    #        title='Windspeed over 24 hours')
    fig.update_layout( 
            title='Windspeed over 24 hours',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19,20,21,22,23,24],
                ticktext = ['1:00', '2:00','3:00', '4:00','5:00','6:00', '7:00','8:00', '9:00','10:00' ,'11:00','12:00', '13:00','14:00' ,'15:00','16:00', '17:00','18:00', '19:00',
                           '20:00', '21:00','22:00', '23:00', '24.00'],
                #yaxis_title="Windspeed in m/s",           
                tickangle = 45,             
            ),

            yaxis = dict(
                title ='Windspeed in m/s',
            )
    )        
    return fig


def get_figure_24h(df, selected_zone, selected_hour=1):
    tmin, tmax = 0,24
    fig = go.Figure()
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    for column in selected_zone:
        color = color_dict[column]
        fig.add_traces(go.Scatter(x=df['HOUR'], y = df[column], 
            mode = 'lines', line=dict(color=color), name=column)
        )
    fig.update_xaxes(range = [tmin, tmax])
    fig.update_yaxes(range = [0,1])
    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext = ['0', '20', '40', '60', '80', '100']
        )
    )
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
            title='Forecast Power Output over 24 hours',
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19,20,21,22,23,24],
                ticktext = ['1:00', '2:00','3:00', '4:00','5:00','6:00', '7:00','8:00', '9:00','10:00' ,'11:00','12:00', '13:00','14:00' ,'15:00','16:00', '17:00','18:00', '19:00',
                           '20:00', '21:00','22:00', '23:00', '24.00']
            ),

        
            yaxis = dict(
                title ='Windpower in %',
            )
    )
    return fig

## get figure for the energy per hour graph 
def get_figure_energy_per_hour(df, selected_zone, selected_hour):
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    df_hour = pd.DataFrame(df[df['HOUR']==selected_hour])
    df_hour = df_hour[selected_zone]
    fig = go.Figure()
    for zone in selected_zone:
        fig.add_traces(
            go.Bar(
                x=[zone], y=df_hour[zone], 
                marker={'color': color_dict[zone]}, 
                showlegend=False,
                text=str(int(100*round(df_hour[zone],2)))+'%',
                textposition='auto',)
    )
    fig.update_yaxes(range = [0,1])
    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext = ['0', '20', '40', '60', '80', '100']
        )
    )
    fig.layout.template = 'plotly_white'
    
    fig.update_layout( 
            title='Forecast Power Output at hour '+str(selected_hour))
    return fig

## wind rose
def get_figure_windrose(df, selected_zone='Wind Farm 1', show_legend=False, show_title=True):
    dfz = df[df['ZONEID']==selected_zone]
    bins = np.linspace(0,24,13)
    labels = range(0,23,2)
    dfz['Windspeed'] = pd.cut(dfz['WS100'], bins=bins, labels = labels)
    dfz_grouped = dfz.groupby(['WD100CARD','Windspeed']).count().reset_index()
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    wd_zeros = np.zeros(len(wd_card))
    df_all_wd_card = pd.DataFrame([wd_card, wd_zeros, wd_zeros])
    df_all_wd_card = df_all_wd_card.T
    df_all_wd_card.columns = ['WD100CARD', 'Windspeed', 'FREQUENCY']
    
    data_wind = dfz_grouped[['WD100CARD', 'Windspeed', 'TIMESTAMP']]
    data_wind.columns = df_all_wd_card.columns

    ws_ls = np.arange(0,24,2)
    wind_all_speeds = pd.DataFrame([['N']*len(ws_ls), ws_ls, np.zeros(len(ws_ls))]).T
    wind_all_speeds.columns = ['WD100CARD', 'Windspeed', 'FREQUENCY']
    wind_all_speeds.rename(columns = {"WD100CARD": "Winddirection"}, inplace=True)
    wind_all_speeds
    df_all_wd_card.rename(columns = {"WD100CARD": "Winddirection"}, inplace=True) 
    data_wind.rename(columns = {"WD100CARD": "Winddirection"}, inplace=True)
    datax = pd.concat([wind_all_speeds, df_all_wd_card, data_wind], axis = 0)

    fig = px.bar_polar(datax, 
        r="FREQUENCY", 
        #theta="WD100CARD",
        theta="Winddirection",
        color="Windspeed", 
        color_discrete_sequence= px.colors.sequential.Plasma_r,
    )                     
    fig.layout.showlegend = show_legend
    fig.update_layout(
        autosize=True,
        width=300,
        height=300,
        margin=dict(l=30, r=30, b=30, t=50, pad=4)
    )
    if show_title:
        fig.update_layout( 
            title=selected_zone
        )
    return fig

def get_figure_cumulated_energy(df_forecast, df_yesterday, selected_zone):
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    # forecast data
    df_f = pd.DataFrame(df_forecast[selected_zone].mean())
    df_f.columns = ['TARGETVAR']
    # yesterday's data
    df_y = df_yesterday.groupby('ZONEID').mean()
    
    # fig = go.Figure()
    # for zone in selected_zone:
    #     color = color_dict[zone]
    #     fig.add_traces(
    #         go.Bar(x=[zone, zone+' yesterday'], 
    #             y=[df_f.loc[zone]['TARGETVAR'], df_y.loc[zone]['TARGETVAR']],
    #             marker={'color': [color_dict[zone], color_dict_light[zone]]}, 
    #             showlegend=False
    #         )
    #     )

    data = {
        "Today": [df_f.loc[zone]['TARGETVAR'] for zone in selected_zone],
        "Yesterday" :[df_y.loc[zone]['TARGETVAR'] for zone in selected_zone],
        "labels": ['Windfarm'+str(n) for n in range(1,11)]    
        
    }

    fig = go.Figure(
    data = [
        go.Bar(
            name="Today",
            x=data["labels"],
            y=data["Today"],
            #offsetgroup=0,
        ),

        go.Bar(
            name="Yesterday",
            x=data["labels"],
            y=data["Yesterday"],
            #offsetgroup=0.1,
        ),
    ],
    layout=go.Layout(
        title="Issue Types - Original and Models",
        yaxis_title="Cumulative Energy output in %",
        #barmode = 'overlay'
    )
)

    fig.update_yaxes(range = [0,1])
    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext = ['0', '20', '40', '60', '80', '100']
        )
    )
    fig.layout.template = 'plotly_white'
    fig.update_layout( 
            title='Cumulative Energy Output over the whole day')
    return fig



################################################################################
# LAYOUT
################################################################################


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
                        initial_visible_month=date(2013, 7, 1),
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
                dbc.Label("Wind Farm selection:"),
                dbc.Checklist(
                    options=[
                        {"label": "Wind Farm 1", "value": "Wind Farm 1"},
                        {"label": "Wind Farm 2", "value": "Wind Farm 2"},
                        {"label": "Wind Farm 3", "value": "Wind Farm 3"},
                        {"label": "Wind Farm 4", "value": "Wind Farm 4"},
                        {"label": "Wind Farm 5", "value": "Wind Farm 5"},
                        {"label": "Wind Farm 6", "value": "Wind Farm 6"},
                        {"label": "Wind Farm 7", "value": "Wind Farm 7"},
                        {"label": "Wind Farm 8", "value": "Wind Farm 8"},
                        {"label": "Wind Farm 9", "value": "Wind Farm 9"},
                        {"label": "Wind Farm 10", "value": "Wind Farm 10"},
                    ],
                    value=["Wind Farm 1", "Wind Farm 2", "Wind Farm 3", "Wind Farm 4", "Wind Farm 5",
                        "Wind Farm 6", "Wind Farm 7", "Wind Farm 8", "Wind Farm 9", "Wind Farm 10", ],
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
        #html.H2("Parameter", className="display-4"),
        #html.Hr(),
        html.P(
            "Choose date and zone for forecast", className="lead"
        ),
        # html.H2('Parameter', style=TEXT_STYLE),
        # html.Hr(),
        controls,
    ],
    style=SIDEBAR_STYLE,
)

title_tab_1 = html.Div('Projected Energy Output', id="title-tab-1")
figure_energy_24h = dcc.Graph(id='figure-energy-24h')
slider_hour = dcc.Slider( 
    id='slider-hour',
    min=1,
    max=24,
    value=1,
    #marks={str(hh): str(hh) for hh in range(1,25)},
    # marks={
    #     1: {'label': '1:00'},
    #     2: {'label': '2:00'},
    #     3: {'label': '3:00'},
    #     4: {'label': '4:00'}
    # },
    
    marks = {n : str(n)+':00' for n in range(1,25)},
    step=None
)

figure_energy_per_hour = dcc.Graph(id="figure-energy-per-hour")
figure_windspeed = dcc.Graph(id="figure-windspeed")
maincontent_tab_1 = html.Div( 
    [
        figure_energy_24h,
        slider_hour,
        figure_energy_per_hour,
        figure_windspeed
    ], 
)



maincontent_tab_3 = dbc.Container( 
    [   
        html.Div(""),
        dbc.Row( 
            [ 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 1"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-1", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 4"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-4", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 7"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-7", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 2"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-2", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 5"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-5", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 8"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-8", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 10"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-10", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
                dbc.Col( 
                    [ 
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 3"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-3", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 6"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-6", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                        dbc.Card( 
                            [ 
                                dbc.CardHeader("Wind Farm 9"),
                                dbc.CardBody(dcc.Graph(id="wind-rose-9", style={'marginLeft': '1em', 'marginRight': '1em'})),
                            ], style={'marginBottom': '1em', 'marginTop': '1em','marginLeft': '1em', 'marginRight': '1em'}
                        ),
                    ]
                ), 
            ]
        )
    ]
)

maincontent_tab_2 = html.Div( 
    [
        dcc.Graph(id='energy-day-cumulative')
    ]
)

title_and_tabs = html.Div( 
    [
        Header("Energy Output Forecast for the Next 24 Hours", app),
        dbc.Tabs( 
            [ 
                dbc.Tab(maincontent_tab_1, 
                    label="Forecast Energy Output", tab_id="tab-energy-forecast"),
                dbc.Tab(maincontent_tab_2, 
                    label="Cumulative Energy Output", tab_id="tab-energy-cumulated"),
                dbc.Tab(maincontent_tab_3, 
                    label="Wind Strength (in m/s) and Direction", tab_id="tab-wind-roses"),
            ],
            id="tabs",
            active_tab="tab-energy-forecast"
        )
    ]
)

app.layout = dbc.Container( 
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
                        title_and_tabs,
                    ], md=9,
                )
            ]
        ),
        # dcc.Store stores the intermediate value
        dcc.Store(id='data_wind_forecast'),
        dcc.Store(id='data_power_forecast'),
        dcc.Store(id='data_power_yesterday'),
    ], fluid=True,
)
################################################################################
# INTERACTION CALLBACKS
################################################################################

@app.callback(
    Output('figure-energy-24h', "figure"),
    Input('zone-selector', 'value'),
    Input('slider-hour', 'value'),
    Input('data_power_forecast', 'data'))
def update_graphs(selected_zone, selected_hour, data_power_forecast):
    df = pd.read_json(data_power_forecast)
    return get_figure_24h(df, selected_zone, selected_hour)


@app.callback(
    Output('figure-windspeed', "figure"),
    Input('zone-selector', 'value'),
    Input('slider-hour', 'value'),
    Input('data_wind_forecast', 'data'))   
def update_windspeed(selected_zone, selected_hour, data_wind_forecast):
    df = pd.read_json(data_wind_forecast)
    return get_figure_windspeed(df, selected_zone, selected_hour)
  

@app.callback(
    Output('figure-energy-per-hour', 'figure'),
    Input('zone-selector', 'value'),
    Input('slider-hour', 'value'),
    Input('data_power_forecast', 'data'))
def update_figure(selected_zone, selected_hour, data_power_forecast):
    df = pd.read_json(data_power_forecast)
    return get_figure_energy_per_hour(df, selected_zone, selected_hour)



@app.callback(
    Output('energy-day-cumulative', "figure"),
    Input('zone-selector', 'value'),
    Input('data_power_forecast', 'data'),
    Input('data_power_yesterday', 'data'))
def update_figure_cumulated_energy(selected_zone, data_power_forecast, data_power_yesterday):
    df_forecast = pd.read_json(data_power_forecast)
    df_yesterday = pd.read_json(data_power_yesterday)
    return get_figure_cumulated_energy(df_forecast, df_yesterday, selected_zone)

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
    Input('data_wind_forecast', 'data'))
def update_figure_windrose(data_wind):
    df_wind = pd.read_json(data_wind)
    return get_figure_windrose(df_wind, 'Wind Farm 1', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 2', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 3', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 4', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 5', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 6', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 7', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 8', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 9', show_title=False),\
           get_figure_windrose(df_wind, 'Wind Farm 10', show_title=False)

@app.callback(
    Output('data_wind_forecast', 'data'),
    Output('data_power_forecast', 'data'),
    Output('data_power_yesterday', 'data'),
    Input('date-picker-single', 'date'))
def update_data(date):
    data_wind_forecast = get_data(
        date, 
        data=data_all,
        features=features_forecast,
        delta_days=None)
    data_wind_forecast['WD100CARD'] = data_wind_forecast.WD100.apply(
        lambda x: degrees_to_cardinal(x))
    data_power_forecast = get_data(
        date,
        data=data_forecast,
        features=None)
    data_power_yesterday = get_data( 
        date,
        data=data_all,
        features=features_retro, 
        delta_days=-1
    )
    return data_wind_forecast.to_json(),\
           data_power_forecast.to_json(),\
           data_power_yesterday.to_json()
    #### HOVER DATA
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update



################################################################################
# VALUES FOR TESTING THE LAYOUT AND GRAPHS
################################################################################

# df, df_wind = make_prediction(day)
# df['HOUR'] = df['TIMESTAMP'].dt.hour
# df_wind['WD100CARD'] = df_wind.WD100.apply(lambda x: degrees_to_cardinal(x))
# print(df_wind.head())
# df.to_csv(filename,index=False)
# df_wind.to_csv(filename_wind,index=False)
data_forecast = pd.read_csv(PATH_PREDICTIONS, parse_dates=['TIMESTAMP'])
data_forecast.rename(
    columns={x: 'Wind Farm '+x.split()[-1] 
    for x in data_forecast.columns if x.startswith('Zone')}, 
    inplace=True)
add_HOUR_column(data_forecast)
data_all = pd.read_csv(PATH_DATA_ALL, parse_dates=['TIMESTAMP'])
data_all.interpolate(method='bfill', inplace=True)
data_all['ZONEID'] = data_all['ZONEID'].apply(lambda x: 'Wind Farm '+str(x))
data_all['HOUR'].replace(0,24, inplace=True)
data_all.replace('Zone', 'Wind Farm', inplace=True)
data_all.reset_index(inplace=True)


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
    pass
