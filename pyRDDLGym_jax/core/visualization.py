import ast
import os
from datetime import datetime
import math
import numpy as np
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import warnings
import webbrowser

# prevent endless console prints
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import dash
from dash.dcc import Interval, Graph, Store
from dash.dependencies import Input, Output, State, ALL
from dash.html import Div, B, H4, P, Img, Hr
import dash_bootstrap_components as dbc

import plotly.colors as pc 
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyRDDLGym.core.debug.decompiler import RDDLDecompiler

if TYPE_CHECKING:
    from pyRDDLGym_jax.core.planner import JaxBackpropPlanner
    
POLICY_DIST_HEIGHT = 400
POLICY_DIST_PLOTS_PER_ROW = 6
ACTION_HEATMAP_HEIGHT = 400
PROGRESS_FOR_NEXT_RETURN_DIST = 2
PROGRESS_FOR_NEXT_POLICY_DIST = 10
REWARD_ERROR_DIST_SUBPLOTS = 20
MODEL_STATE_ERROR_HEIGHT = 300
POLICY_STATE_VIZ_MAX_HEIGHT = 800
GP_POSTERIOR_MAX_HEIGHT = 800

PLOT_AXES_FONT_SIZE = 11
EXPERIMENT_ENTRY_FONT_SIZE = 14

    
class JaxPlannerDashboard:
    '''A dashboard app for monitoring the jax planner progress.'''

    def __init__(self, theme: str=dbc.themes.CERULEAN) -> None:
        
        self.timestamps = {}
        self.duration = {}
        self.seeds = {}
        self.status = {}
        self.warnings = []
        self.progress = {}
        self.checked = {}
        self.rddl = {}
        self.planner_info = {}
        
        self.xticks = {}
        self.test_return = {}
        self.train_return = {}
        self.return_dist = {}
        self.return_dist_ticks = {}
        self.return_dist_last_progress = {}
        self.action_output = {}
        self.policy_params = {}
        self.policy_params_ticks = {}
        self.policy_params_last_progress = {}
        self.policy_viz = {}
        
        self.relaxed_exprs = {}
        self.relaxed_exprs_values = {}
        self.train_reward_dist = {}
        self.test_reward_dist = {}
        self.train_state_fluents = {}
        self.test_state_fluents = {}
        
        self.tuning_gp_heatmaps = None
        self.tuning_gp_targets = None
        self.tuning_gp_predicted = None
        self.tuning_gp_params = None
        self.tuning_gp_update = False
        
        # ======================================================================
        # CREATE PAGE LAYOUT
        # ======================================================================
        
        def create_experiment_table(active_page, page_size):
            start = (active_page - 1) * page_size
            end = start + page_size
            rows = []
            
            # header
            row = dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Display'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=1),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Experiment ID'), style={"padding": "0"}
                    ), className="border-0 bg-transparent"),
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Seed'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=1),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Timestamp'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Duration'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=1),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Best Return'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Status'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Progress'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=1)
            ])
            rows.append(row)
            
            # experiments
            for (i, id) in enumerate(self.checked):
                if i >= end:
                    break
                if i >= start and i < end:
                    progress = self.progress[id]
                    row = dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Checkbox(
                                        id={'type': 'experiment-checkbox', 'index': id},
                                        value=self.checked[id]
                                    )],
                                    style={"padding": "0"}
                                ),
                                className="border-0 bg-transparent"
                            )
                        ], width=1),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(id, style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(self.seeds[id], style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            )
                        ], width=1),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(self.timestamps[id], style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(f'{self.duration[id]:.3f}s', style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            ),
                        ], width=1),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(f'{(self.test_return[id] or [np.nan])[-1]:.3f}',
                                             style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(self.status[id], style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(
                                    [dbc.Progress(label=f"{progress}%", value=progress)],
                                    style={"padding": "0"}
                                ),
                                className="border-0 bg-transparent"
                            ),
                        ], width=1),
                    ])
                    rows.append(row)
            return rows
        
        app = dash.Dash(__name__, external_stylesheets=[theme])
        app.title = 'JaxPlan Dashboard'
        
        app.layout = dbc.Container([
            Store(id='refresh-interval', data=2000),
            Store(id='experiment-num-per-page', data=10),
            Store(id='model-params-dropdown-expr', data=''),
            Store(id='model-errors-state-dropdown-selected', data=''),
            Store(id='viz-skip-frequency', data=5),
            Store(id='viz-num-trajectories', data=3),
            Div(id='viewport-sizer', style={'display': 'none'}),
            
            # navbar
            dbc.Navbar(
                dbc.Container([
                    # Img(src=LOGO_FILE, height="30px", style={'margin-right': '10px'}),
                    dbc.NavbarBrand(f"JaxPlan Dashboard"),
                    dbc.Nav([
                        dbc.NavItem(
                            dbc.NavLink(
                                "Docs",
                                href="https://pyrddlgym.readthedocs.io/en/latest/jax.html"
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                "GitHub",
                                href="https://github.com/pyrddlgym-project/pyRDDLGym-jax"
                            )
                        ),
                        dbc.NavItem(
                            dbc.NavLink(
                                "Submit an Issue",
                                href="https://github.com/pyrddlgym-project/pyRDDLGym-jax/issues"
                            )
                        )                                
                    ], navbar=True, className="me-auto"),
                    dbc.Nav([                        
                        dbc.DropdownMenu(
                            [dbc.DropdownMenuItem("500ms", id='05sec'),
                             dbc.DropdownMenuItem("1s", id='1sec'),
                             dbc.DropdownMenuItem("2s", id='2sec'),
                             dbc.DropdownMenuItem("5s", id='5sec'),
                             dbc.DropdownMenuItem("10s", id='10sec'),
                             dbc.DropdownMenuItem("30s", id='30sec'),
                             dbc.DropdownMenuItem("1m", id='1min'),
                             dbc.DropdownMenuItem("5m", id='5min'),
                             dbc.DropdownMenuItem("1d", id='1day')],
                            label="Refresh: 2s",
                            id='refresh-rate-dropdown',
                            nav=True
                        )
                    ], navbar=True),
                    dbc.Nav([                        
                        dbc.DropdownMenu(
                            [dbc.DropdownMenuItem("5", id='5pp'),
                             dbc.DropdownMenuItem("10", id='10pp'),
                             dbc.DropdownMenuItem("25", id='25pp'),
                             dbc.DropdownMenuItem("50", id='50pp')],
                            label="Exp. Per Page: 10",
                            id='experiment-num-per-page-dropdown',
                            nav=True
                        )
                    ], navbar=True)
                ], fluid=True)
            ),
            
            # experiments
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            Div(create_experiment_table(0, 10), id='experiment-table',
                                style={'fontSize': f'{EXPERIMENT_ENTRY_FONT_SIZE}px'}),
                            dbc.Pagination(id='experiment-pagination',
                                           active_page=1, max_value=1, size="sm")
                        ], style={'padding': '10px'})
                    ], className="border-0 bg-transparent")
                ])
            ]),
            
            # empirical results tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        
                        # returns
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(Graph(id='train-return-graph'), width=6),
                                    dbc.Col(Graph(id='test-return-graph'), width=6),
                                ]),
                                dbc.Row([                           
                                    Graph(id='dist-return-graph')
                                ])
                            ]), className="border-0 bg-transparent"
                        ), label="Performance", tab_id='tab-performance'
                        ),
                        
                        # policy
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    Graph(id='action-output'),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        Hr(className='my-4')
                                    ])
                                ]),
                                dbc.Row([
                                    Graph(id='policy-params'),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        Hr(className='my-4')
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.DropdownMenu(
                                                    [dbc.DropdownMenuItem('Every Frame', id='viz-skip-1'),
                                                     dbc.DropdownMenuItem('Every 2 Frames', id='viz-skip-2'),
                                                     dbc.DropdownMenuItem('Every 3 Frames', id='viz-skip-3'),
                                                     dbc.DropdownMenuItem('Every 4 Frames', id='viz-skip-4'),
                                                     dbc.DropdownMenuItem('Every 5 Frames', id='viz-skip-5'),
                                                     dbc.DropdownMenuItem('Every 10 Frames', id='viz-skip-10')],
                                                    label="Render: Every 5 Frames",
                                                    id='viz-skip-dropdown'
                                                ),
                                            ], width='auto'),
                                            dbc.Col([
                                                dbc.DropdownMenu(
                                                    [dbc.DropdownMenuItem('1', id='viz-num-1'),
                                                     dbc.DropdownMenuItem('2', id='viz-num-2'),
                                                     dbc.DropdownMenuItem('3', id='viz-num-3'),
                                                     dbc.DropdownMenuItem('4', id='viz-num-4'),
                                                     dbc.DropdownMenuItem('5', id='viz-num-5')],
                                                    label="Max. Trajectories: 3",
                                                    id='viz-num-dropdown'
                                                ),
                                            ], width='auto'),
                                            dbc.Col([
                                                dbc.Button('Run Policy Visualization',
                                                           id='policy-viz-button'),
                                            ], width='auto')
                                        ]),
                                        dbc.Row([
                                            Graph(id='policy-viz')
                                        ])
                                    ])
                                ]),
                            ]), className="border-0 bg-transparent"
                        ), label="Policy", tab_id='tab-policy'
                        ),
                        
                        # model
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.DropdownMenu(
                                                [],
                                                label="RDDL Expression",
                                                id='model-params-dropdown'
                                            ),
                                            Graph(id='model-params-graph')
                                        ], className="border-0 bg-transparent"
                                        ),
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        Hr(className='my-4')
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card(
                                            dbc.CardBody(
                                                Graph(id='model-errors-reward-graph')
                                            ),
                                            className="border-0 bg-transparent"
                                        ),
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.DropdownMenu(
                                                [],
                                                label="State-Fluent",
                                                id='model-errors-state-dropdown'
                                            ),
                                            Graph(id='model-errors-state-graph')
                                        ], className="border-0 bg-transparent"
                                        )
                                    ])
                                ]),
                            ]), className="border-0 bg-transparent"
                        ), label="Model", tab_id='tab-model'
                        ),
                        
                        # information
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Alert(id="planner-info", color="light", dismissable=False)
                                ]),
                            ]), className="border-0 bg-transparent"
                        ), label="Debug", tab_id='tab-debug'
                        ),
                        
                        # tuning
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(Graph(id='tuning-target-graph'), width=6),
                                    dbc.Col(Graph(id='tuning-scatter-graph'), width=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        Hr(className='my-4')
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col(Graph(id='tuning-gp-mean-graph'))
                                ]),
                                dbc.Row([
                                    dbc.Col(Graph(id='tuning-gp-unc-graph'))
                                ])
                            ]), className="border-0 bg-transparent"
                        ), label="Tuning", tab_id='tab-tuning'
                        ),
                    ], id='tabs-main')
                ], width=12)
            ]),
            
            # refresh interval
            Interval(
                id='interval',
                interval=2000,
                n_intervals=0
            ),
            Div(id='trigger-experiment-check', style={'display': 'none'})
        ], fluid=True, className="dbc")
        
        # JavaScript to retrieve the viewport dimensions 
        app.clientside_callback( 
            """
            function(n_intervals) { 
                return { 
                    'height': window.innerHeight, 
                    'width': window.innerWidth 
                }; 
            } 
            """, 
            Output('viewport-sizer', 'children'), 
            Input('interval', 'n_intervals')
        )
        
        # ======================================================================
        # CREATE EVENTS
        # ======================================================================
        
        # modify refresh rate
        @app.callback(
            Output("refresh-interval", "data"),
            [Input("05sec", "n_clicks"),
             Input("1sec", "n_clicks"),
             Input("2sec", "n_clicks"),
             Input("5sec", "n_clicks"),
             Input("10sec", "n_clicks"),
             Input("30sec", "n_clicks"),
             Input("1min", "n_clicks"),
             Input("5min", "n_clicks"),
             Input("1day", "n_clicks")],
            [State('refresh-interval', 'data')]
        )
        def click_refresh_rate(n05, n1, n2, n5, n10, n30, n1m, n5m, nd, data):
            ctx = dash.callback_context 
            if not ctx.triggered: 
                return data 
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == '05sec':
                return 500
            elif button_id == '1sec':
                return 1000
            elif button_id == '2sec':
                return 2000
            elif button_id == '5sec':
                return 5000
            elif button_id == '10sec':
                return 10000
            elif button_id == '30sec':
                return 30000
            elif button_id == '1min':
                return 60000
            elif button_id == '5min':
                return 300000
            elif button_id == '1day':
                return 86400000
            return data            
            
        @app.callback(
            Output('interval', 'interval'),
            [Input('refresh-interval', 'data')]
        )
        def update_refresh_rate(selected_interval):
            return selected_interval if selected_interval else 2000
        
        @app.callback(
            Output('refresh-rate-dropdown', 'label'),
            [Input('refresh-interval', 'data')]
        )
        def update_refresh_rate(selected_interval):
            if selected_interval == 500:
                return 'Refresh: 500ms'
            elif selected_interval == 1000:
                return 'Refresh: 1s'
            elif selected_interval == 2000:
                return 'Refresh: 2s'
            elif selected_interval == 5000:
                return 'Refresh: 5s'
            elif selected_interval == 10000:
                return 'Refresh: 10s'
            elif selected_interval == 30000:
                return 'Refresh: 30s'
            elif selected_interval == 60000:
                return 'Refresh: 1m'
            elif selected_interval == 300000:
                return 'Refresh: 5m'
            else:
                return 'Refresh: 2s'
        
        # update the experiments per page
        @app.callback(
            Output("experiment-num-per-page", "data"),
            [Input("5pp", "n_clicks"),
             Input("10pp", "n_clicks"),
             Input("25pp", "n_clicks"),
             Input("50pp", "n_clicks")],
            [State('experiment-num-per-page', 'data')]
        )
        def click_experiments_per_page(n5, n10, n25, n50, data):
            ctx = dash.callback_context 
            if not ctx.triggered: 
                return data 
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == '5pp':
                return 5
            elif button_id == '10pp':
                return 10
            elif button_id == '25pp':
                return 25
            elif button_id == '50pp':
                return 50
            return data
        
        @app.callback(
            Output('experiment-num-per-page-dropdown', 'label'),
            [Input('experiment-num-per-page', 'data')]
        )
        def update_experiments_per_page(selected_num):
            return f'Exp. Per Page: {selected_num}'
        
        # update the experiment table
        @app.callback(
            Output('experiment-table', 'children'),
            [Input('interval', 'n_intervals'),
             Input('experiment-pagination', 'active_page')],
            State('experiment-num-per-page', 'data')
        ) 
        def update_experiment_table(n, active_page, npp): 
            return create_experiment_table(active_page, npp)
        
        @app.callback(
            Output('experiment-pagination', 'max_value'),
            Input('interval', 'n_intervals'),
            State('experiment-num-per-page', 'data')
        ) 
        def update_experiment_max_pages(n, npp): 
            return (len(self.checked) + npp - 1) // npp            
        
        @app.callback(
            Output('trigger-experiment-check', 'children'),
            Input({'type': 'experiment-checkbox', 'index': ALL}, 'value'),
            State({'type': 'experiment-checkbox', 'index': ALL}, 'id')
        )
        def update_checked_experiment_status(checked, ids):
            for (i, chk) in enumerate(checked):
                row = ids[i]['index']
                self.checked[row] = chk
            return time.time()
        
        # update the return information        
        @app.callback(
            Output('train-return-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_train_return_graph(n, trigger, active_tab):
            if active_tab != 'tab-performance': return dash.no_update
            fig = go.Figure()
            for (row, checked) in self.checked.copy().items():
                if checked:
                    fig.add_trace(go.Scatter(
                        x=self.xticks[row], y=self.train_return[row],
                        name=f'id={row}',
                        mode='lines+markers',
                        marker=dict(size=3), line=dict(width=2)
                    ))
            fig.update_layout(
                title=dict(text="Train Return"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Cumulative Reward")),
                font=dict(size=PLOT_AXES_FONT_SIZE),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                template="plotly_white"
            )
            return fig
        
        @app.callback(
            Output('test-return-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_test_return_graph(n, trigger, active_tab):
            if active_tab != 'tab-performance': return dash.no_update
            fig = go.Figure()
            for (row, checked) in self.checked.copy().items():
                if checked:
                    fig.add_trace(go.Scatter(
                        x=self.xticks[row], y=self.test_return[row],
                        name=f'id={row}',
                        mode='lines+markers',
                        marker=dict(size=3), line=dict(width=2)
                    ))
            fig.update_layout(
                title=dict(text="Test Return"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Cumulative Reward")),
                font=dict(size=PLOT_AXES_FONT_SIZE),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                template="plotly_white"
            )
            return fig
        
        @app.callback(
            Output('dist-return-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_dist_return_graph(n, trigger, active_tab):
            if active_tab != 'tab-performance': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            for (row, checked) in self.checked.copy().items():
                if checked:
                    return_dists = self.return_dist[row]
                    ticks = self.return_dist_ticks[row]
                    colors = pc.sample_colorscale(
                        pc.get_colorscale('Blues'),
                        np.linspace(0.1, 1, len(return_dists))
                    )
                    for (ic, (tick, dist)) in enumerate(zip(ticks, return_dists)):
                        fig.add_trace(go.Violin(
                            y=dist, line_color=colors[ic], name=f'{tick}'
                        ))
                    fig.update_traces(
                        orientation='v', side='positive', width=3, points=False)
                    fig.update_layout(
                        title=dict(text="Distribution of Return"),
                        xaxis=dict(title=dict(text="Training Iteration")),
                        yaxis=dict(title=dict(text="Cumulative Reward")),
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        showlegend=False,
                        yaxis_showgrid=False, yaxis_zeroline=False,
                        template="plotly_white"
                    )
                    break
            return fig
        
        # update the action heatmap
        @app.callback(
            Output('action-output', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_action_heatmap(n, trigger, active_tab):
            if active_tab != 'tab-policy': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            for (row, checked) in self.checked.copy().items():
                if checked and self.action_output[row] is not None:
                    num_plots = len(self.action_output[row])
                    titles = []
                    for (_, act, _) in self.action_output[row]:
                        titles.append(f'Values of Action-Fluents {act}')
                        titles.append(f'Std. Dev. of Action-Fluents {act}')
                    fig = make_subplots(
                        rows=num_plots, cols=2,
                        shared_xaxes=True, horizontal_spacing=0.15,
                        subplot_titles=titles
                    )
                    for (i, (action_output, action, action_labels)) \
                    in enumerate(self.action_output[row]):
                        action_values = np.mean(1. * action_output, axis=0).T
                        action_errors = np.std(1. * action_output, axis=0).T
                        fig.add_trace(go.Heatmap(
                            z=action_values,
                            x=np.arange(action_values.shape[1]),
                            y=np.arange(action_values.shape[0]),
                            colorscale='Blues', colorbar_x=0.45,
                            colorbar_len=0.8 / num_plots,
                            colorbar_y=1 - (i + 0.5) / num_plots
                        ), row=i + 1, col=1)
                        fig.add_trace(go.Heatmap(
                            z=action_errors,
                            x=np.arange(action_errors.shape[1]),
                            y=np.arange(action_errors.shape[0]),
                            colorscale='Reds', colorbar_len=0.8 / num_plots,
                            colorbar_y=1 - (i + 0.5) / num_plots
                        ), row=i + 1, col=2)
                    fig.update_layout(
                        title="Values of Action-Fluents",
                        xaxis=dict(title=dict(text="Decision Epoch")),
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        height=ACTION_HEATMAP_HEIGHT * num_plots,
                        showlegend=False,
                        template="plotly_white"
                    )          
                    break
            return fig
        
        # update the weight histograms
        @app.callback(
            Output('policy-params', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_policy_params(n, trigger, active_tab):
            if active_tab != 'tab-policy': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            for (row, checked) in self.checked.copy().items():
                policy_params = self.policy_params[row]
                policy_params_ticks = self.policy_params_ticks[row]
                if checked and policy_params is not None and policy_params:
                    titles = []
                    for (layer_name, layer_params) in policy_params[0].items():
                        if isinstance(layer_params, dict):
                            for weight_name in layer_params:
                                titles.append(f'{layer_name}/{weight_name}')
                    
                    n_rows = math.ceil(len(policy_params) / POLICY_DIST_PLOTS_PER_ROW)
                    fig = make_subplots(
                        rows=n_rows, cols=POLICY_DIST_PLOTS_PER_ROW,
                        shared_xaxes=True,
                        subplot_titles=titles
                    )
                    colors = pc.sample_colorscale(
                        pc.get_colorscale('Blues'),
                        np.linspace(0.1, 1, len(policy_params))
                    )[::-1]
                    
                    for (it, (tick, policy_params_t)) in enumerate(
                        zip(policy_params_ticks[::-1], policy_params[::-1])):
                        r, c = 1, 1
                        for (layer_name, layer_params) in policy_params_t.items():
                            if isinstance(layer_params, dict):
                                for (weight_name, weight_values) in layer_params.items():
                                    if r <= n_rows:
                                        fig.add_trace(go.Violin(
                                            x=np.ravel(weight_values),
                                            line_color=colors[it], name=f'{tick}'
                                        ), row=r, col=c)
                                    c += 1
                                    if c > POLICY_DIST_PLOTS_PER_ROW:
                                        r += 1
                                        c = 1
                    fig.update_traces(
                        orientation='h', side='positive', width=3, points=False)
                    fig.update_layout(
                        title="Distribution of Network Weight Parameters",
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        showlegend=False,
                        xaxis_showgrid=False, xaxis_zeroline=False,
                        height=POLICY_DIST_HEIGHT * n_rows,
                        template="plotly_white"
                    )
                    break
            return fig
        
        # modify viz skip rate
        @app.callback(
            Output("viz-skip-frequency", "data"),
            [Input("viz-skip-1", "n_clicks"),
             Input("viz-skip-2", "n_clicks"),
             Input("viz-skip-3", "n_clicks"),
             Input("viz-skip-4", "n_clicks"),
             Input("viz-skip-5", "n_clicks"),
             Input("viz-skip-10", "n_clicks")],
            [State('viz-skip-frequency', 'data')]
        )
        def click_viz_skip_rate(v1, v2, v3, v4, v5, v10, data):
            ctx = dash.callback_context 
            if not ctx.triggered: 
                return data 
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'viz-skip-1':
                return 1
            elif button_id == 'viz-skip-2':
                return 2
            elif button_id == 'viz-skip-3':
                return 3
            elif button_id == 'viz-skip-4':
                return 4
            elif button_id == 'viz-skip-5':
                return 5
            elif button_id == 'viz-skip-10':
                return 10
            return data            
        
        @app.callback(
            Output('viz-skip-dropdown', 'label'),
            [Input('viz-skip-frequency', 'data')]
        )
        def update_viz_skip_dropdown_text(viz_skip):
            if viz_skip == 1:
                return 'Render: Every Frame'
            else:
                return f'Render: Every {viz_skip} Frames'
        
        # modify viz count
        @app.callback(
            Output("viz-num-trajectories", "data"),
            [Input("viz-num-1", "n_clicks"),
             Input("viz-num-2", "n_clicks"),
             Input("viz-num-3", "n_clicks"),
             Input("viz-num-4", "n_clicks"),
             Input("viz-num-5", "n_clicks")],
            [State('viz-num-trajectories', 'data')]
        )
        def click_viz_num_render(v1, v2, v3, v4, v5, data):
            ctx = dash.callback_context 
            if not ctx.triggered: 
                return data 
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'viz-num-1':
                return 1
            elif button_id == 'viz-num-2':
                return 2
            elif button_id == 'viz-num-3':
                return 3
            elif button_id == 'viz-num-4':
                return 4
            elif button_id == 'viz-num-5':
                return 5
            return data            
        
        @app.callback(
            Output('viz-num-dropdown', 'label'),
            [Input('viz-num-trajectories', 'data')]
        )
        def update_viz_num_dropdown_text(viz_num):
            return f'Max. Trajectories: {viz_num}'
            
        # update the policy viz
        @app.callback(
            Output('policy-viz', 'figure'),
            Input("policy-viz-button", "n_clicks"),
            [State('viewport-sizer', 'children'),
             State("viz-skip-frequency", "data"),
             State("viz-num-trajectories", "data")]
        )
        def update_policy_viz(n_clicks, viewport_size, skip_freq, viz_num):
            if not viewport_size: return dash.no_update
            if not n_clicks: return dash.no_update
            
            for (row, checked) in self.checked.copy().items():
                viz = self.policy_viz[row]
                if checked and viz is not None:
                    states = self.train_state_fluents[row]
                    lookahead = next(iter(states.values())).shape[1]
                    batch_idx = self.representative_trajectories(states, k=viz_num)
                    policy_viz_frames = []
                    for idx in batch_idx:
                        avg_image = 0.
                        num_image = 0
                        viz.__init__(self.rddl[row])
                        for t in range(0, lookahead, skip_freq):
                            state_t = {name: values[idx, t]
                                       for (name, values) in states.items()}
                            state_t = self.rddl[row].ground_vars_with_values(state_t)
                            avg_image += np.asarray(viz.render(state_t))
                            num_image += 1
                        avg_image /= num_image
                        policy_viz_frames.append(avg_image)
                        
                    subplot_width = min(
                        viewport_size['width'] // len(policy_viz_frames), 
                        POLICY_STATE_VIZ_MAX_HEIGHT)
                    fig = make_subplots(
                        rows=1, cols=len(policy_viz_frames)
                    )
                    for (col, frame) in enumerate(policy_viz_frames):
                        fig.add_trace(go.Image(z=frame, hoverinfo='skip'),
                                      row=1, col=1 + col)
                    fig.update_layout( 
                        title="Representative Trajectories",
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        xaxis=dict(showticklabels=False), 
                        yaxis=dict(showticklabels=False),
                        width=subplot_width * len(policy_viz_frames),
                        height=subplot_width * 1,
                        showlegend=False,
                        template="plotly_white"
                    )
                    return fig
            return dash.no_update
        
        # update the model parameter information
        @app.callback(
            Output('model-params-dropdown', 'children'),
            [Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_model_params_dropdown_create(trigger, active_tab):
            if active_tab != 'tab-model': return dash.no_update
            items = []
            for (row, checked) in self.checked.copy().items():
                if checked:
                    items = []
                    for (expr_id, expr) in self.relaxed_exprs[row].items():
                        items.append(dbc.DropdownMenuItem([
                            B(f'{expr_id}: '),
                            expr.replace('\n', ' ')[:120]
                        ], id={'type': 'expr-dropdown-item', 'index': expr_id}))
                    break
            return items
        
        @app.callback(
            Output('model-params-dropdown-expr', 'data'),
            Input({'type': 'expr-dropdown-item', 'index': ALL}, 'n_clicks')
        )
        def update_model_params_dropdown_select(n_clicks):
            ctx = dash.callback_context
            if not ctx.triggered: 
                return dash.no_update
            if not next((item for item in n_clicks if item is not None), False):
                return dash.no_update
            return ast.literal_eval(
                ctx.triggered[0]['prop_id'].split('.n_clicks')[0])['index']
                
        @app.callback(
            Output('model-params-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('tabs-main', 'active_tab')],
            [State('model-params-dropdown-expr', 'data')]
        )
        def update_model_params_graph(n, active_tab, expr_id):
            if active_tab != 'tab-model': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            if expr_id == '': return fig            
            for (row, checked) in self.checked.copy().items():
                if checked:
                    fig.add_trace(go.Scatter(
                        x=self.xticks[row],
                        y=self.relaxed_exprs_values[row][expr_id],
                        mode='lines+markers',
                        marker=dict(size=3), line=dict(width=2)
                    ))
                    fig.update_layout(
                        title=dict(text=f"Model Parameters for Expression {expr_id}"),
                        xaxis=dict(title=dict(text="Training Iteration")),
                        yaxis=dict(title=dict(text="Parameter Value")),
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        template="plotly_white"
                    )
                    break
            return fig
        
        # update the model errors information for reward
        @app.callback(
            Output('model-errors-reward-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_model_error_reward_graph(n, trigger, active_tab):
            if active_tab != 'tab-model': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            for (row, checked) in self.checked.copy().items():
                if checked and row in self.train_reward_dist:
                    data = self.train_reward_dist[row]
                    num_epochs = data.shape[1]
                    step = 1
                    if num_epochs > REWARD_ERROR_DIST_SUBPLOTS:
                        step = num_epochs // REWARD_ERROR_DIST_SUBPLOTS
                    for epoch in range(0, num_epochs, step):
                        fig.add_trace(go.Violin(
                            y=self.train_reward_dist[row][:, epoch], x0=epoch,
                            side='negative', line_color='red',
                            name=f'Train Epoch {epoch + 1}'
                        ))
                        fig.add_trace(go.Violin(
                            y=self.test_reward_dist[row][:, epoch], x0=epoch,
                            side='positive', line_color='blue',
                            name=f'Test Epoch {epoch + 1}'
                        ))
                    fig.update_traces(meanline_visible=True)
                    fig.update_layout(
                        title=dict(text="Distribution of Reward in Relaxed Model vs True Model"),
                        xaxis=dict(title=dict(text="Decision Epoch")),
                        yaxis=dict(title=dict(text="Reward")),
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        violingap=0, violinmode='overlay', showlegend=False,
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        template="plotly_white"
                    )
                    break
            return fig
        
        # update the model errors information for state
        @app.callback(
            Output('model-errors-state-dropdown', 'children'),
            [Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_model_errors_state_dropdown_create(trigger, active_tab):
            if active_tab != 'tab-model': return dash.no_update
            items = []
            for (row, checked) in self.checked.copy().items():
                if checked:
                    items = []
                    for name in self.train_state_fluents[row]:
                        items.append(dbc.DropdownMenuItem(
                            [name],
                            id={'type': 'state-fluent-dropdown-item', 'index': name}
                        ))
                    break
            return items
        
        @app.callback(
            Output('model-errors-state-dropdown-selected', 'data'),
            Input({'type': 'state-fluent-dropdown-item', 'index': ALL}, 'n_clicks')
        )
        def update_model_errors_state_dropdown_select(n_clicks):
            ctx = dash.callback_context
            if not ctx.triggered: 
                return dash.no_update
            if not next((item for item in n_clicks if item is not None), False):
                return dash.no_update
            return ast.literal_eval(
                ctx.triggered[0]['prop_id'].split('.n_clicks')[0])['index']
        
        @app.callback(
            Output('model-errors-state-graph', 'figure'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')],
            [State('model-errors-state-dropdown-selected', 'data')]
        )
        def update_model_errors_state_graph(n, trigger, active_tab, state):
            if active_tab != 'tab-model': return dash.no_update
            fig = go.Figure()
            fig.update_layout(template='plotly_white')
            if not state: return fig
            for (row, checked) in self.checked.copy().items():
                if checked and row in self.train_state_fluents:
                    train_values = self.train_state_fluents[row][state]
                    test_values = self.test_state_fluents[row][state]
                    train_values = 1 * train_values.reshape(train_values.shape[:2] + (-1,))
                    test_values = 1 * test_values.reshape(test_values.shape[:2] + (-1,))
                    num_epochs, num_states = train_values.shape[1:]
                    step = 1
                    if num_epochs > REWARD_ERROR_DIST_SUBPLOTS:
                        step = num_epochs // REWARD_ERROR_DIST_SUBPLOTS
                    fig = make_subplots(
                        rows=num_states, cols=1, shared_xaxes=True,
                        subplot_titles=self.rddl[row].variable_groundings[state]
                    )
                    for istate in range(num_states):
                        for epoch in range(0, num_epochs, step):
                            fig.add_trace(go.Violin(
                                y=train_values[:, epoch, istate], x0=epoch,
                                side='negative', line_color='red',
                                name=f'Train Epoch {epoch + 1}'
                            ), row=istate + 1, col=1)
                            fig.add_trace(go.Violin(
                                y=test_values[:, epoch, istate], x0=epoch,
                                side='positive', line_color='blue',
                                name=f'Test Epoch {epoch + 1}'
                            ), row=istate + 1, col=1)
                    fig.update_traces(meanline_visible=True)
                    fig.update_layout(
                        title=dict(text=(f"Distribution of State-Fluent {state} "
                                         f"in Relaxed Model vs True Model")),
                        xaxis=dict(title=dict(text="Decision Epoch")),
                        yaxis=dict(title=dict(text="State-Fluent Value")),
                        font=dict(size=PLOT_AXES_FONT_SIZE),
                        height=MODEL_STATE_ERROR_HEIGHT * num_states,
                        violingap=0, violinmode='overlay', showlegend=False,
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        template="plotly_white"
                    )
                    break
            return fig
                
        # update the run information
        @app.callback(
            Output('planner-info', 'children'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_planner_info(n, trigger, active_tab): 
            if active_tab != 'tab-debug': return dash.no_update
            result = []
            for (row, checked) in self.checked.copy().items():
                if checked:
                    result = [
                        H4(f'Hyper-Parameters [id={row}]', className="alert-heading"),
                        P(self.planner_info[row], style={"whiteSpace": "pre-wrap"})
                    ]
                    break            
            return result
        
        # update the tuning result
        @app.callback(
            [Output('tuning-target-graph', 'figure'),
             Output('tuning-scatter-graph', 'figure'),
             Output('tuning-gp-mean-graph', 'figure'),
             Output('tuning-gp-unc-graph', 'figure')],
            [Input('interval', 'n_intervals'),
             Input('tabs-main', 'active_tab'),
             Input('viewport-sizer', 'children')]
        )
        def update_tuning_gp_graph(n, active_tab, viewport_size):
            if not self.tuning_gp_update: return dash.no_update
            if not viewport_size: return dash.no_update
            
            # tuning target trend
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=np.arange(len(self.tuning_gp_targets)), y=self.tuning_gp_targets,
                mode='lines+markers',
                marker=dict(size=3), line=dict(width=2)
            ))
            fig1.update_layout(
                title=dict(text="Target Values of Trial Points"),
                xaxis=dict(title=dict(text="Trial Point")),
                yaxis=dict(title=dict(text="Target Value")),
                font=dict(size=PLOT_AXES_FONT_SIZE),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                template="plotly_white"
            )
            
            # tuning scatter actual and predicted
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=self.tuning_gp_targets, y=self.tuning_gp_predicted,
                mode='markers', marker=dict(size=6)
            ))
            fig2.add_shape(
                type="line", 
                x0=np.min(self.tuning_gp_targets), 
                y0=np.min(self.tuning_gp_targets), 
                x1=np.max(self.tuning_gp_targets), 
                y1=np.max(self.tuning_gp_targets),
                line=dict(dash="dot", color='gray')
            )
            fig2.update_layout(
                title=dict(text="Gaussian Process Goodness-of-Fit Plot"),
                xaxis=dict(title=dict(text="Actual Target Value")),
                yaxis=dict(title=dict(text="Predicted Target Value")),
                font=dict(size=PLOT_AXES_FONT_SIZE),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                template="plotly_white"
            )
            
            # tuning posterior plot
            num_cols = len(self.tuning_gp_heatmaps)
            num_rows = len(self.tuning_gp_heatmaps[0])            
            fig3 = make_subplots(rows=num_rows, cols=num_cols)
            fig4 = make_subplots(rows=num_rows, cols=num_cols)            
            for col, data_col in enumerate(self.tuning_gp_heatmaps):
                for row, data in enumerate(data_col):
                    p1, p2, p1v, p2v, mean, std = data
                    fig3.add_trace(go.Heatmap(
                        z=mean, x=p1v, y=p2v, colorscale='Blues', showscale=False
                    ), row=row + 1, col=col + 1)
                    fig4.add_trace(go.Heatmap(
                        z=std, x=p1v, y=p2v, colorscale='Reds', showscale=False
                    ), row=row + 1, col=col + 1)
                    fig3.add_trace(go.Scatter(
                        x=self.tuning_gp_params[p1], 
                        y=self.tuning_gp_params[p2],
                        opacity=1,
                        mode='markers', 
                        marker=dict(size=5, color='green', symbol='x')
                    ), row=row + 1, col=col + 1)     
                    fig4.add_trace(go.Scatter(
                        x=self.tuning_gp_params[p1], 
                        y=self.tuning_gp_params[p2],
                        opacity=1,
                        mode='markers', 
                        marker=dict(size=5, color='green', symbol='x')
                    ), row=row + 1, col=col + 1)    
                    fig3.update_xaxes(title_text=p1, row=row + 1, col=col + 1)
                    fig4.update_xaxes(title_text=p1, row=row + 1, col=col + 1)
                    fig3.update_yaxes(title_text=p2, row=row + 1, col=col + 1)
                    fig4.update_yaxes(title_text=p2, row=row + 1, col=col + 1)
            subplot_width = min(
                GP_POSTERIOR_MAX_HEIGHT, viewport_size['width'] // num_cols)                         
            fig3.update_layout(
                title="Posterior Mean of Gaussian Process",
                font=dict(size=PLOT_AXES_FONT_SIZE),
                height=subplot_width * num_rows,
                width=subplot_width * num_cols,
                autosize=False,
                showlegend=False,
                template="plotly_white"
            )       
            fig4.update_layout(
                title="Posterior Uncertainty of Gaussian Process",
                font=dict(size=PLOT_AXES_FONT_SIZE),
                height=subplot_width * num_rows,
                width=subplot_width * num_cols,
                autosize=False,
                showlegend=False,
                template="plotly_white"
            )
                     
            self.tuning_gp_update = False
            return (fig1, fig2, fig3, fig4)
            
        self.app = app
    
    # ==========================================================================
    # DASHBOARD EXECUTION
    # ==========================================================================
    
    def launch(self, port: int=1222, daemon: bool=True) -> None:
        '''Launches the dashboard in a browser window.'''
        
        # open the browser to the required port
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://127.0.0.1:{port}/')
        
        # run the app in a new thread at the specified port
        def run_dash():
            self.app.run(port=port)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.daemon = daemon
        dash_thread.start()
    
    @staticmethod
    def get_planner_info(planner: 'JaxBackpropPlanner') -> Dict[str, Any]:
        '''Some additional info directly from the planner that is required by
        the dashboard.'''        
        return {
            'rddl': planner.rddl,
            'string': planner.summarize_system() + str(planner),
            'model_parameter_info': planner.compiled.model_parameter_info(),
            'trace_info': planner.compiled.traced
        }
        
    def register_experiment(self, experiment_id: str,
                            planner_info: Dict[str, Any],
                            key: Optional[int]=None,
                            viz: Optional[Any]=None) -> str:
        '''Starts monitoring a new experiment.'''
        
        # make sure experiment id does not exist
        if experiment_id is None: 
            experiment_id = len(self.xticks) + 1
        if experiment_id in self.xticks:
            raise ValueError(f'An experiment with id {experiment_id} '
                             'was already created.')
            
        self.timestamps[experiment_id] = datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d %H:%M:%S')
        self.duration[experiment_id] = 0
        self.seeds[experiment_id] = key    
        self.status[experiment_id] = 'N/A'  
        self.progress[experiment_id] = 0
        self.warnings = []
        self.rddl[experiment_id] = planner_info['rddl']
        self.planner_info[experiment_id] = planner_info['string']        
        self.checked[experiment_id] = False
        
        self.xticks[experiment_id] = []
        self.train_return[experiment_id] = []
        self.test_return[experiment_id] = []
        self.return_dist_ticks[experiment_id] = []
        self.return_dist_last_progress[experiment_id] = 0
        self.return_dist[experiment_id] = []
        self.action_output[experiment_id] = None
        self.policy_params[experiment_id] = []
        self.policy_params_ticks[experiment_id] = []
        self.policy_params_last_progress[experiment_id] = 0
        self.policy_viz[experiment_id] = viz
        
        decompiler = RDDLDecompiler()
        self.relaxed_exprs[experiment_id] = {}
        self.relaxed_exprs_values[experiment_id] = {}
        for info in planner_info['model_parameter_info'].values():
            expr = planner_info['trace_info'].lookup(info['id'])
            compiled_expr = decompiler.decompile_expr(expr)
            self.relaxed_exprs[experiment_id][info['id']] = compiled_expr
            self.relaxed_exprs_values[experiment_id][info['id']] = []
        
        return experiment_id
    
    @staticmethod
    def representative_trajectories(trajectories, k, max_iter=300):
        n = next(iter(trajectories.values())).shape[0]
        points = np.concatenate([
            np.reshape(1. * values, (n, -1)) 
            for values in trajectories.values()
        ], axis=1)
        
        k = min(k, n)
        centroids = points[np.random.choice(n, k, replace=False)]    
        for _ in range(max_iter):
            distances = np.linalg.norm(
                points[:, None, :] - centroids[None, :, :], axis=-1)
            cluster_assignment = np.argmin(distances, axis=1)
            new_centroids = np.stack([
                np.mean(points[cluster_assignment == i], axis=0)
                for i in range(k)
            ], axis=0)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return np.unique(np.argmin(distances, axis=0))

    def update_experiment(self, experiment_id: str, callback: Dict[str, Any]) -> None:
        '''Pass new information and update the dashboard for a given experiment.'''
        
        # data for return curves
        iteration = callback['iteration']
        self.xticks[experiment_id].append(iteration)
        self.train_return[experiment_id].append(callback['train_return'])    
        self.test_return[experiment_id].append(callback['best_return'])
        
        # data for return distributions
        progress = callback['progress']
        if progress - self.return_dist_last_progress[experiment_id] \
            >= PROGRESS_FOR_NEXT_RETURN_DIST:
            self.return_dist_ticks[experiment_id].append(iteration)
            self.return_dist[experiment_id].append(
                np.sum(np.asarray(callback['reward']), axis=1))
            self.return_dist_last_progress[experiment_id] = progress
        
        # data for action heatmaps
        action_output = []
        rddl = self.rddl[experiment_id]
        for action in rddl.action_fluents:
            action_values = np.asarray(callback['fluents'][action])
            action_output.append(
                (action_values.reshape(action_values.shape[:2] + (-1,)),
                 action,
                 rddl.variable_groundings[action])
            )
        self.action_output[experiment_id] = action_output
        
        # data for policy weight distributions
        if progress - self.policy_params_last_progress[experiment_id] \
            >= PROGRESS_FOR_NEXT_POLICY_DIST:
            self.policy_params_ticks[experiment_id].append(iteration)
            self.policy_params[experiment_id].append(callback['best_params'])
            self.policy_params_last_progress[experiment_id] = progress
        
        # data for model relaxations
        model_params = callback['model_params']
        for (key, values) in model_params.items():
            expr_id = int(str(key).split('_')[0])
            self.relaxed_exprs_values[experiment_id][expr_id].append(values.item())
        self.train_reward_dist[experiment_id] = callback['train_log']['reward']
        self.test_reward_dist[experiment_id] = callback['reward']
        self.train_state_fluents[experiment_id] = {
            name: np.asarray(callback['train_log']['fluents'][name])
            for name in rddl.state_fluents or name in rddl.observ_fluents
        }
        self.test_state_fluents[experiment_id] = {
            name: np.asarray(callback['fluents'][name])
            for name in self.train_state_fluents[experiment_id]
        }
        
        # update experiment table info
        self.status[experiment_id] = str(callback['status']).split('.')[1]
        self.duration[experiment_id] = callback["elapsed_time"]
        self.progress[experiment_id] = progress
        self.warnings = None
    
    def update_tuning(self, optimizer: Any,
                      bounds: Dict[str, Tuple[float, float]]) -> None:
        '''Updates the hyper-parameter tuning plots.'''
        
        self.tuning_gp_heatmaps = []
        self.tuning_gp_update = False
        if not optimizer.res: return
        
        self.tuning_gp_targets = optimizer.space.target.reshape((-1,))
        self.tuning_gp_predicted = \
            optimizer._gp.predict(optimizer.space.params).reshape((-1,))
        self.tuning_gp_params = {name: optimizer.space.params[:, i] 
                                 for (i, name) in enumerate(optimizer.space.keys)}
        
        for (i1, param1) in enumerate(optimizer.space.keys):
            self.tuning_gp_heatmaps.append([])
            for (i2, param2) in enumerate(optimizer.space.keys):
                if i2 > i1:
                    
                    # Generate a grid for visualization
                    p1_values = np.linspace(*bounds[param1], 100)
                    p2_values = np.linspace(*bounds[param2], 100)
                    P1, P2 = np.meshgrid(p1_values, p2_values)
                    
                    # Predict the mean and deviation of the surrogate model
                    fixed_params = max(
                        optimizer.res,
                        key=lambda x: x['target'])['params'].copy()
                    fixed_params.pop(param1)
                    fixed_params.pop(param2)
                    param_grid = []
                    for p1, p2 in zip(np.ravel(P1), np.ravel(P2)):
                        params = {param1: p1, param2: p2}
                        params.update(fixed_params)
                        param_grid.append(
                            [params[key] for key in optimizer.space.keys])
                    param_grid = np.asarray(param_grid)
                    mean, std = optimizer._gp.predict(param_grid, return_std=True)
                    mean = mean.reshape(P1.shape)
                    std = std.reshape(P1.shape)
                    self.tuning_gp_heatmaps[-1].append(
                        (param1, param2, p1_values, p2_values, mean, std))
        self.tuning_gp_update = True
