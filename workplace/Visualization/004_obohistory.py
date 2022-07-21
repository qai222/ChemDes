import glob
import math
import os.path

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

from lsal.utils import pkl_load, get_basename

AvailableHistoryData = sorted(glob.glob("../SingleLigandCampaign/al_workflow/visdata_obo/*.pkl"))
AvailableHistoryData = {get_basename(f): pkl_load(f) for f in AvailableHistoryData}

SWF_names = sorted(AvailableHistoryData.keys())

uncertainty_types = ["uncertainty", "uncertainty_top2"]

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div(
            [
                html.Div(children="""SWF name: """),
                dcc.Dropdown(
                    options=SWF_names,
                    value=SWF_names[0],
                    id="swf_name",
                    multi=False,
                    clearable=True,
                    disabled=False,
                ),
            ],
            style={"width": "15%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.Div(children="""learning session: """),
                dcc.Dropdown(
                    id="learning_session",
                    multi=False,
                    clearable=True,
                    disabled=False,
                ),
            ],
            style={"width": "15%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.Div(children="""overall uncertainty: """),
                dcc.Dropdown(
                    id="uncertainty_type",
                    options=uncertainty_types,
                    value=uncertainty_types[0],
                    multi=False,
                    clearable=True,
                    disabled=False,
                ),
            ],
            style={"width": "15%", "display": "inline-block"},
        ),
    ]),

    dcc.Graph(id='graph', clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),

])


@app.callback(
    Output("learning_session", "options"),
    [Input("swf_name", "value"), ],
)
def session_options(swf_name: str):
    return sorted(AvailableHistoryData[swf_name].keys())


@app.callback(
    Output('graph', 'figure'),
    [Input('swf_name', 'value'), Input("learning_session", "value"), Input("uncertainty_type", "value")],
)
def update_graph(swf_name: str, status_key: str, uncertainty_type: str):
    visdata = AvailableHistoryData[swf_name][status_key]
    ligands = sorted(visdata.keys())

    # subplot titles
    titles = []
    for ligand in ligands:
        title = "{}:{}<br>uncertainty: {:.4f}".format(ligand.int_label, ligand.name, visdata[ligand][uncertainty_type])
        titles.append(title)

    # subplot layout
    ncols = 5
    nrows = (len(ligands) // ncols) + 1
    fig = make_subplots(rows=nrows + 1, cols=ncols, subplot_titles=titles)

    ys = []
    xs = []
    for i, Ligand in enumerate(visdata):
        if i == 0:
            showlegend = True
        else:
            showlegend = False
        is_learned = visdata[Ligand]["is_learned"]
        x = visdata[Ligand]["real_xs"]
        y = visdata[Ligand]["real_ys"]
        fx = visdata[Ligand]["fake_xs"].tolist()
        fy = visdata[Ligand]["fake_ys"].tolist()
        fyerr = 3 * visdata[Ligand]["fake_ys_err"]
        fyerr = fyerr.tolist()
        xs += fx
        xs += x
        ys += fy
        ys += y
        irow = i // ncols
        icol = i % ncols
        # add real xy trace
        fig.add_trace(
            go.Scatter(
                x=x, y=y, name="real", mode="markers",
                marker=dict(color="black"),
                showlegend=showlegend,
            ),
            row=irow + 1, col=icol + 1
        )
        # add fake xy trace
        if is_learned:
            trace_color = 'rgba(0, 255, 255, 0.1)'
        else:
            trace_color = 'rgba(255, 0, 0, 0.1)'
        fig.add_trace(
            go.Scatter(
                x=fx, y=fy, name="pred +/- 3*sigma", mode="markers",
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=fyerr,
                    color=trace_color,
                    symmetric=True,
                    visible=True),
                marker=dict(color=trace_color),
                showlegend=showlegend,
            ),
            row=irow + 1, col=icol + 1
        )

    fig.update_layout(
        height=300 * (nrows + 1), width=300 * ncols,
        legend_title_text='Experimental Data',
    )
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    xaxis_min = math.log10(minx) - 0.1
    xaxis_max = math.log10(maxx) + 0.1
    fig.update_xaxes(type="log", range=[xaxis_min, xaxis_max])
    fig.update_yaxes(range=[miny * 0.9, maxy * 1.1])  # linear range
    return fig


if __name__ == "__main__":
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
