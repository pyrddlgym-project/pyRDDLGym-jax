import os
from datetime import datetime
import math
import numpy as np
import time
import threading
from typing import Any, Dict, Optional, TYPE_CHECKING
import warnings
import webbrowser

import dash
from dash.dcc import Interval, Graph, Store
from dash.dependencies import Input, Output, State, ALL
from dash.html import Div, B, H4, P
import dash_bootstrap_components as dbc

import plotly.colors as pc 
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyRDDLGym.core.debug.decompiler import RDDLDecompiler

from pyRDDLGym_jax import __version__
if TYPE_CHECKING:
    from pyRDDLGym_jax.core.planner import JaxBackpropPlanner
    
POLICY_DIST_HEIGHT = 400
POLICY_DIST_PLOTS_PER_ROW = 6
ACTION_HEATMAP_HEIGHT = 400
EXPERIMENT_PER_PAGE = 10
PROGRESS_FOR_NEXT_RETURN_DIST = 2
PROGRESS_FOR_NEXT_POLICY_DIST = 10
        
    
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
        self.action_output = {}
        self.policy_params = {}
        self.policy_params_ticks = {}
        
        self.relaxed_exprs = {}
        self.relaxed_exprs_values = {}
        
        # ======================================================================
        # CREATE PAGE LAYOUT
        # ======================================================================
        
        def create_experiment_table(active_page, page_size=EXPERIMENT_PER_PAGE):
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
                ], width=1),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Seed'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Timestamp'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=2),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Duration'), style={"padding": "0"}
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
                ], width=2)
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
                                        id={'type': 'checkbox', 'index': id},
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
                        ], width=1),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody(self.seeds[id], style={"padding": "0"}),
                                className="border-0 bg-transparent"
                            )
                        ], width=2),
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
                        ], width=2),
                    ])
                    rows.append(row)
            return rows
        
        def create_model_relaxation_graph(xticks, values, expr_id):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xticks, y=values,
                mode='lines+markers',
                marker=dict(size=2), line=dict(width=2)
            ))
            fig.update_layout(
                title=dict(text=f"Model Parameters for Expression {expr_id}"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Parameter Value")),
                font=dict(size=11),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                template="plotly_white"
            )
            return fig
            
        def create_model_relaxation_table(experiment_id):
            rows = []
            
            # create header
            row = dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Relaxed Expression'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=4),
                dbc.Col([
                    dbc.Card(dbc.CardBody(
                        B('Model Parameter(s)'), style={"padding": "0"}
                    ), className="border-0 bg-transparent")
                ], width=8)
            ])
            rows.append(row)
            if experiment_id is None: return rows
            
            # create content
            for (expr_id, expr) in self.relaxed_exprs[experiment_id].items():
                xvalues = self.xticks[experiment_id]
                yvalues = self.relaxed_exprs_values[experiment_id][expr_id]
                fig = create_model_relaxation_graph(xvalues, yvalues, expr_id)
                row = dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody(expr, style={"padding": "0"}),
                            className="border-0 bg-transparent"
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody(
                                Graph(figure=fig), style={"padding": "0"}
                            ),
                            className="border-0 bg-transparent"
                        ),
                    ], width=8)
                ])
                rows.append(row)
            return rows
            
        app = dash.Dash(__name__, external_stylesheets=[theme])
        
        app.layout = dbc.Container([
            Store(id='refresh-interval'),
            
            # navbar
            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand(f"JaxPlan Version {__version__}"),
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
                             dbc.DropdownMenuItem("2s", id='2sec', active=True),
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
                    ], navbar=True)
                ], fluid=True)
            ),
            
            # experiments
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            Div(create_experiment_table(0), id='experiment-table'),
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
                                    Graph(id='policy-params'),
                                ])
                            ]), className="border-0 bg-transparent"
                        ), label="Policy", tab_id='tab-policy'
                        ),
                        
                        # model
                        dbc.Tab(dbc.Card(
                            dbc.CardBody([
                                Div(create_model_relaxation_table(None), id='model-relaxation-table')
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
                        )
                    ], id='tabs-main')
                ], width=12)
            ]),
            
            # refresh interval
            Interval(
                id='interval',
                interval=1000,
                n_intervals=0
            ),
            Div(id='trigger-experiment-check', style={'display': 'none'})
        ], fluid=True, className="dbc")
        
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
            else: 
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
                return 'Refresh: 1d'
        
        # update the experiment table
        @app.callback(
            Output('experiment-table', 'children'),
            [Input('interval', 'n_intervals'),
             Input('experiment-pagination', 'active_page')]
        ) 
        def update_experiment_table(n, active_page): 
            return create_experiment_table(active_page)
        
        @app.callback(
            Output('experiment-pagination', 'max_value'),
            Input('interval', 'n_intervals')
        ) 
        def update_experiment_max_pages(n): 
            return (len(self.checked) + EXPERIMENT_PER_PAGE - 1) // EXPERIMENT_PER_PAGE            
        
        @app.callback(
            Output('trigger-experiment-check', 'children'),
            Input({'type': 'checkbox', 'index': ALL}, 'value'),
            State({'type': 'checkbox', 'index': ALL}, 'id')
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
                        marker=dict(size=2), line=dict(width=2)
                    ))
            fig.update_layout(
                title=dict(text="Train Return"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Cumulative Reward")),
                font=dict(size=11),
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
                        marker=dict(size=2), line=dict(width=2)
                    ))
            fig.update_layout(
                title=dict(text="Test Return"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Cumulative Reward")),
                font=dict(size=11),
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
                    break
            fig.update_traces(
                orientation='v', side='positive', width=3, points=False)
            fig.update_layout(
                title=dict(text="Distribution of Return"),
                xaxis=dict(title=dict(text="Training Iteration")),
                yaxis=dict(title=dict(text="Cumulative Reward")),
                font=dict(size=11),
                showlegend=False,
                yaxis_showgrid=False, yaxis_zeroline=False,
                template="plotly_white"
            )
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
                        ), row=i + 1, col=1)
                        fig.add_trace(go.Heatmap(
                            z=action_errors,
                            x=np.arange(action_errors.shape[1]),
                            y=np.arange(action_errors.shape[0]),
                            colorscale='Reds'
                        ), row=i + 1, col=2)
                    fig.update_layout(
                        title="Values of Action-Fluents",
                        xaxis=dict(title=dict(text="Training Iteration")),
                        font=dict(size=11),
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
                        font=dict(size=11),
                        showlegend=False,
                        xaxis_showgrid=False, xaxis_zeroline=False,
                        height=POLICY_DIST_HEIGHT * n_rows,
                        template="plotly_white"
                    )
                    break
            return fig
        
        # update the model relaxation information
        @app.callback(
            Output('model-relaxation-table', 'children'),
            [Input('interval', 'n_intervals'),
             Input('trigger-experiment-check', 'children'),
             Input('tabs-main', 'active_tab')]
        )
        def update_model_relaxation_table(n, trigger, active_tab):
            if active_tab != 'tab-model': return dash.no_update
            result = []
            for (row, checked) in self.checked.copy().items():
                if checked:
                    result = create_model_relaxation_table(row)
                    break
            return result
        
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
        
    def register_experiment(self, experiment_id: str, 
                            planner: 'JaxBackpropPlanner',
                            key: Optional[int]=None) -> str:
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
        self.rddl[experiment_id] = planner.rddl
        self.planner_info[experiment_id] = str(planner)        
        self.checked[experiment_id] = False
        
        self.xticks[experiment_id] = []
        self.train_return[experiment_id] = []
        self.test_return[experiment_id] = []
        self.return_dist_ticks[experiment_id] = []
        self.return_dist[experiment_id] = []
        self.action_output[experiment_id] = None
        self.policy_params[experiment_id] = []
        self.policy_params_ticks[experiment_id] = []
        
        decompiler = RDDLDecompiler()
        self.relaxed_exprs[experiment_id] = {}
        self.relaxed_exprs_values[experiment_id] = {}
        for info in planner.compiled.model_parameter_info().values():
            expr = planner.compiled.traced.lookup(info['id'])
            compiled_expr = decompiler.decompile_expr(expr)
            self.relaxed_exprs[experiment_id][info['id']] = compiled_expr
            self.relaxed_exprs_values[experiment_id][info['id']] = []
            
        return experiment_id
    
    def update_experiment(self, experiment_id: str, callback: Dict[str, Any]) -> None:
        '''Pass new information and update the dashboard for a given experiment.'''
        
        # data for return curves
        iteration = callback['iteration']
        self.xticks[experiment_id].append(iteration)
        self.train_return[experiment_id].append(callback['train_return'])    
        self.test_return[experiment_id].append(callback['best_return'])
        
        # data for return distributions
        progress = callback['progress']
        if progress % PROGRESS_FOR_NEXT_RETURN_DIST == 0 \
        and progress != self.progress[experiment_id]:
            self.return_dist_ticks[experiment_id].append(iteration)
            self.return_dist[experiment_id].append(
                np.sum(np.asarray(callback['reward']), axis=1))
        
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
        if progress % PROGRESS_FOR_NEXT_POLICY_DIST == 0 \
        and progress != self.progress[experiment_id]:
            self.policy_params_ticks[experiment_id].append(iteration)
            self.policy_params[experiment_id].append(callback['best_params'])
        
        # data for model relaxations
        model_params = callback['model_params']
        for (key, values) in model_params.items():
            expr_id = int(str(key).split('_')[0])
            self.relaxed_exprs_values[experiment_id][expr_id].append(values.item())
        
        # update experiment table info
        self.status[experiment_id] = str(callback['status']).split('.')[1]
        self.duration[experiment_id] = callback["elapsed_time"]
        self.progress[experiment_id] = progress
        self.warnings = None
    