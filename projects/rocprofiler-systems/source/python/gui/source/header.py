#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os

import dash_daq as daq
import dash_bootstrap_components as dbc

from dash import html, dcc


def file_path():
    return html.Div(
        className="workload",
        children=[
            dcc.Input(
                id="file-path",
                placeholder="Insert Workload directory",
                type="text",
                debounce=True,
            )
        ],
    )


def function_filter(_id, _placeholder):
    return html.Div(
        className="regex",
        children=[
            dcc.Input(
                id=_id,
                placeholder=_placeholder,
                type="text",
                debounce=True,
            )
        ],
    )


def upload_file():
    return html.Div(
        className="upload",
        children=[
            # drag and drop
            dcc.Upload(
                id="upload-drag",
                children=[html.A("Drag and Drop or Select a File")],
            )
        ],
    )


def minPoints(name, values):
    return html.Div(
        className="filter",
        id="min-points",
        children=[
            # html.Div(
            #     children=[
            html.A(children=["Min Points:"]),
            daq.Slider(
                min=0,
                max=values,
                step=1,
                value=1,
                id="points-filt",
                handleLabel={"showCurrentValue": True, "label": " "},
                size=120,
            ),
            #     ],
            # ),
        ],
    )


def span(name):
    total_length = 30
    base = os.path.basename(name)
    dir_len = total_length - len(base)
    if len(name) > total_length:
        return {
            "label": html.Span(name[0:dir_len] + "..." + base, title=name),
            "value": name,
        }
    else:
        return {"label": html.Span(name, title=name), "value": name}


def sortBy(name, values, default, multi_):
    values = list(map(span, values))
    return html.Div(
        className="filter",
        children=[
            html.A(children=[name + ":"]),
            dcc.Dropdown(
                values,
                id=name + "-filt",
                multi=multi_,
                value=default,
                clearable=False,
            ),
        ],
    )


def refresh():
    return html.Div(
        className="nav-right",
        children=[
            html.Button(
                className="refresh",
                children=["Refresh Data"],
                id="refresh",
            )
        ],
    )


def get_header(dropDownMenuItems, input_filters):
    children_ = [
        html.Nav(
            id="nav-wrap",
            children=[
                html.Ul(
                    id="nav",
                    children=[
                        html.Div(
                            className="nav-left",
                            children=[
                                dbc.DropdownMenu(
                                    dropDownMenuItems, label="Menu", menu_variant="dark"
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    ]
    filter_children = []
    for filter in input_filters:
        header_nav = children_[0].children[0].children

        if filter["type"] == "int":
            filter_children.append(minPoints(filter["Name"], filter["values"]))
        elif filter["type"] == "Name":
            filter_children.append(
                sortBy(
                    filter["Name"],
                    filter["values"],
                    filter["default"],
                    filter["multi"],
                    # {},
                )
            )
        else:
            print("type not supported")
            # sys.exit(1)

    header_nav = children_[0].children[0].children
    filter_children.append(function_filter("experiment_regex", "Experiment regex"))
    filter_children.append(function_filter("progpt_regex", "Progress Point regex"))
    filter_children.append(file_path())
    filter_children.append(upload_file())
    # filter_children.append(refresh())
    ul = html.Div(
        id="nav-center",
        className="nav-center",
        children=filter_children,
        # [
        # html.Li(className="filter", children=filter_children),
        # refresh(),
        # html.Li(
        #     className="regex",
        #     children=[
        #         function_filter("experiment_regex", "Funtion/line regex"),
        #         function_filter("progpt_regex", "Experiment regex"),
        #         file_path(),
        #         upload_file(),
        #     ],
        # ),
        # ],
    )
    header_nav.append(ul)
    header_nav.append(refresh())
    return html.Header(id="home", children=children_)
