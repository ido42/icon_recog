# -*- coding: utf-8 -*-
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import os
from PIL import Image
import numpy as np
import plotly.express as px
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.io as pio


app = dash.Dash(__name__)


app.layout = html.Div([

    # first row
    html.Div(children=[

        # first column of first row
        html.Div(children=[
        dcc.Markdown(" ## Choose Your Display ## "),
        dcc.Dropdown(
        id='main_dropdown',
        options=[
                {'label': 'beyond-better', 'value': 'beyond_better_display.PNG'},
                {'label': 'beyond-door', 'value': 'beyond_door_display.PNG'},
                {'label': 'BG14', 'value': 'BG14_display_ff_frz.PNG'}
            ],
        value = 'beyond_better_display.PNG',
        multi=False,
        clearable=False
    ),],style={"width": "300px", "height": "50px", 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '0', 'margin-top': '3vw'}),
    html.Div(children=[
    dcc.Graph(
    id='img-output',
    ),], style={'layer': 'below', "width": "500px", "height": "150px", 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '200', 'margin-top': '0'}),

    
    dcc.Markdown(" ## Add Steps ## "),
    
    html.Div("Button 1"),
    dcc.Dropdown(
        id='dropdown1',
        options=[
                {'label': 'Wi-Fi', 'value': 'Wi-Fi'},
                {'label': 'Fast Freezer', 'value': 'Fast Freeze'},
                {'label': 'Stuff', 'value': 'Stuff'},
                {'label': "None", 'value': 'None'}
            ],
        style={'width': '300px', 'height': "100%"},
        value = 'Fast Freeze',
        multi=False
    ),
    
    dcc.RadioItems(
        id="press1",
        options=[
            {'label': 'Short', 'value': 'Short'},
            {'label': 'Long', 'value': 'Long'},
        ],
        value='Short',
        labelStyle={'display': 'inline-block'}
    ),
 
    html.Div("Button 2"),
    dcc.Dropdown(
        id='dropdown2',
        options=[
                {'label': 'Wi-Fi', 'value': 'Wi-Fi'},
                {'label': 'Fast Freezer', 'value': 'Fast Freeze'},
                {'label': 'Things', 'value': 'Stuffs'},
                {'label': 'None', 'value': 'None'}
            ],
        style={'width': '300px', 'height': "100%", 'margin-bottom': '0'},
        value = 'None',
        multi=False
    ),
    
    dcc.RadioItems(
        id="press2",
        options=[
            {'label': 'Short', 'value': 'Short'},
            {'label': 'Long', 'value': 'Long'},
        ],
        value='Short',
        labelStyle={'display': 'inline-block'}
    ),

    html.Button('Add Step', id='button', n_clicks=0),
    dcc.Markdown("## Test Procedure ##"),
    html.Div("Step 1:", id="step1-output"),
    html.Div("Step 2:", id="step2-output"),
    html.Div("Step 3:", id="step3-output"),
    html.Div("Step 4:", id="step4-output"),
    
    html.Button('RUN', id='run', n_clicks=0),




#dcc.Textarea(
#        id='textarea-example',
#        value= [id="count"],
#        style={'width': '100%', 'height': 300},
#    ),
])
    ])
@app.callback(
    Output(component_id='img-output', component_property='figure'),
    Input(component_id='main_dropdown', component_property='value'))
def update_output(input_value):
    img = np.array(Image.open(f"assets/{input_value}"))
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

#@app.callback(
#    Output('count','children'),
#    Input('button','n_clicks'))
#def update_output2(n_clicks):
#    return n_clicks
#
#@app.callback(
#    Output('step1-','children'),
#    State('dropdown2', 'value'),
#    State("checklist2","value1"))
#def update_output3(value):
#    return [value1,value]

@app.callback([Output('step1-output', 'children')],
              [Input('button', 'n_clicks')],
              [State('dropdown1', 'value'),
              State('press1', 'value'),
              State('dropdown2', 'value'),
              State('press2', 'value')])
def update_output2(n_clicks, input1, input2, input3, input4):
#    return [n_clicks, input1, input2]
    if n_clicks==1:
        if input3 == "None":
            return ["""Step {}:
                Only button to check is: "{}",
                with press type: "{}""
            """.format(n_clicks, input1, input2)]
        else: ["""Step {}:
                Button 1 to check is: "{}",
                with press type: "{}" --
                Button 2 to check is: "{}",
                with press type: "{}"
            """.format(n_clicks, input1, input2, input3, input4)]
    else:
        raise PreventUpdate


        
@app.callback([Output('step2-output', 'children')],
              [Input('button', 'n_clicks')],
              [State('dropdown1', 'value'),
              State('press1', 'value'),
              State('dropdown2', 'value'),
              State('press2', 'value')])
def update_output3(n_clicks, input1, input2, input3, input4):
#    return [n_clicks, input1, input2]
    if n_clicks==2:
        return ["""Step {}:
            Button 1 to check is: "{}",
            press type is: "{}" --
            Button 2 to check is: "{}",
            press type is: "{}"
        """.format(n_clicks, input1, input2, input3, input4)]
    else:
        raise PreventUpdate
        
@app.callback([Output('step3-output', 'children')],
              [Input('button', 'n_clicks')],
              [State('dropdown1', 'value'),
              State('press1', 'value'),
              State('dropdown2', 'value'),
              State('press2', 'value')])
def update_output4(n_clicks, input1, input2, input3, input4):
#    return [n_clicks, input1, input2]
    if n_clicks==3:
        return ["""Step {}:
            Button 1 to check is: "{}",
            press type is: "{}" --
            Button 2 to check is: "{}",
            press type is: "{}"
        """.format(n_clicks, input1, input2, input3, input4)]
    else:
        raise PreventUpdate
    
@app.callback([Output('step4-output', 'children')],
              [Input('button', 'n_clicks')],
              [State('dropdown1', 'value'),
              State('press1', 'value'),
              State('dropdown2', 'value'),
              State('press2', 'value')])
def update_output5(n_clicks, input1, input2, input3, input4):
#    return [n_clicks, input1, input2]
    if n_clicks==4:
        return ["""Step {}:
            Button 1 to check is: "{}",
            press type is: "{}" --
            Button 2 to check is: "{}",
            press type is: "{}"
        """.format(n_clicks, input1, input2, input3, input4)]
    else:
        raise PreventUpdate



#@app.callback(Output('count', 'children'),
#              Input('button', 'n_clicks'))
#def update_output3(n_clicks):
#    return [n_clicks]
    
#@app.callback(
#    Output(component_id='my-output', component_property='children'),
#    Input(component_id='dropdown', component_property='value'))
#def update_output_div(input_value):
#    return 'Output: {}'.format(input_value)
    


if __name__ == "__main__":
    app.run_server(debug=True)