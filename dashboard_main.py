from io import SEEK_CUR
import pandas as pd
import numpy as np
import base64
import datetime
import io

import dash
from datetime import date
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px

from PIL import ImageColor

from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table

from process_and_predict import create_features

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


## parse the contents of an uploaded file
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df_raw = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df_feat = create_features(df_raw)
            
        # elif 'xls' in filename:
        #     # Assume that the user uploaded an excel file
        #     df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ]), None, None

    return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns]
            ),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the web browser
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all'
            })
        ]), df_raw, df_feat

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
color_dict_light = { k:ImageColor.getcolor(v, "RGB") for k,v in color_dict.items()}
color_dict_light = { k:"rgba({}, {}, {}, {})".format(v[0], v[1], v[2], 0.4) for k,v in color_dict_light.items()}


def get_figure_24h(df, selected_zone, selected_hour=1):
    tmin, tmax = 1,24
    fig = go.Figure()
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    for column in selected_zone:
        color = color_dict[column]
        fig.add_traces(go.Scatter(x=df['HOUR'], y = df[column], 
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
def get_figure_windrose(df, selected_zone='Zone 1', show_legend=False, show_title=True):
    dfz = df[df['ZONEID']==selected_zone]
    bins = np.linspace(0,24,13)
    labels = range(0,23,2)
    dfz['WS100BIN'] = pd.cut(dfz['WS100'], bins=bins, labels = labels)
    dfz_grouped = dfz.groupby(['WD100CARD','WS100BIN']).count().reset_index()
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    wd_zeros = np.zeros(len(wd_card))
    df_all_wd_card = pd.DataFrame([wd_card, wd_zeros, wd_zeros])
    df_all_wd_card = df_all_wd_card.T
    df_all_wd_card.columns = ['WD100CARD', 'WS100BIN', 'FREQUENCY']
    
    data_wind = dfz_grouped[['WD100CARD', 'WS100BIN', 'TIMESTAMP']]
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
    
    fig = go.Figure()
    for zone in selected_zone:
        color = color_dict[zone]
        fig.add_traces(
            go.Bar(x=[zone, zone+' yesterday'], 
                y=[df_f.loc[zone]['TARGETVAR'], df_y.loc[zone]['TARGETVAR']],
                marker={'color': [color_dict[zone], color_dict_light[zone]]}, 
                showlegend=False
            )
        )
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    fig.update_layout( 
            title='Accumulated Energy Output over the whole day')
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
        html.H2("Parameter", className="display-4"),
        html.Hr(),
        html.P(
            "Choose date and zone for forecast", className="lead"
        ),
        # html.H2('Parameter', style=TEXT_STYLE),
        # html.Hr(),
        controls,
        html.Br(),
        dcc.Link('Use your own weather forecast"', href='/own-forecast')
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
    marks={str(hh): str(hh) for hh in range(1,25)},
    step=None
)

figure_energy_per_hour = dcc.Graph(id="figure-energy-per-hour")
maincontent_tab_1 = html.Div( 
    [
        figure_energy_24h,
        slider_hour,
        figure_energy_per_hour,
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
                    label="Cumulated_Energy_Output", tab_id="tab-energy-cumulated"),
                dbc.Tab(maincontent_tab_3, 
                    label="Wind Strength and Direction", tab_id="tab-wind-roses"),
            ],
            id="tabs",
            active_tab="tab-energy-forecast"
        )
    ]
)



forecasts_original_layout = dbc.Container( 
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
# LAYOUT, FOR OWN FORECASTS
################################################################################


forecasts_own_layout = dbc.Container( 
    [
        dbc.Row( 
            [
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                    html.Div(id='output-data-upload'),
                ])
            ]
        ),
        # dcc.Store stores the intermediate value
        dcc.Store(id='own_wind_forecast'),
        dcc.Store(id='own_power_forecast'),
    ], fluid=True,
)



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
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
    return get_figure_windrose(df_wind, 'Zone 1', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 2', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 3', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 4', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 5', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 6', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 7', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 8', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 9', show_title=False),\
           get_figure_windrose(df_wind, 'Zone 10', show_title=False)

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



@app.callback(Output('output-data-upload', 'children'),
              Output('own_wind_forecast', 'data'),
              Output('own_power_yesterday', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children, 

# Update the index
@app.callback(Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return forecasts_original_layout
    elif pathname == '/own-forecast':
        return forecasts_own_layout
    else:
        return forecasts_original_layout



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
add_HOUR_column(data_forecast)
data_all = pd.read_csv(PATH_DATA_ALL, parse_dates=['TIMESTAMP'])
data_all.interpolate(method='bfill', inplace=True)
data_all['ZONEID'] = data_all['ZONEID'].apply(lambda x: 'Zone '+str(x))
data_all.reset_index(inplace=True)


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
    pass
