import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import dash_cytoscape as cyto
import json
import uuid
import base64
from datetime import datetime

# Load extra layouts
cyto.load_extra_layouts()

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "CyEditor - Dash/Cytoscape"

# Enhanced node types with colors and shapes - inspired by diagrams.net General section
node_types = [
    # Basic Flow Chart Shapes
    {"label": "Start", "value": "start", "color": "#3174ad", "shape": "ellipse"},
    {"label": "Process", "value": "process", "color": "#27ae60", "shape": "rectangle"},
    {"label": "Decision", "value": "decision", "color": "#f39c12", "shape": "diamond"},
    {"label": "End", "value": "end", "color": "#e74c3c", "shape": "ellipse"},
    {
        "label": "Document",
        "value": "document",
        "color": "#9b59b6",
        "shape": "rectangle",
    },
    {"label": "Data", "value": "data", "color": "#1abc9c", "shape": "parallelogram"},
    # General Geometric Shapes (diagrams.net style)
    {"label": "Circle", "value": "circle", "color": "#ff6b6b", "shape": "ellipse"},
    {"label": "Square", "value": "square", "color": "#4ecdc4", "shape": "rectangle"},
    {"label": "Triangle", "value": "triangle", "color": "#ffe66d", "shape": "triangle"},
    {"label": "Hexagon", "value": "hexagon", "color": "#a8e6cf", "shape": "hexagon"},
    {"label": "Octagon", "value": "octagon", "color": "#ff8b94", "shape": "octagon"},
    {"label": "Star", "value": "star", "color": "#ffd93d", "shape": "star"},
    # Additional Diagram Elements
    {"label": "Note", "value": "note", "color": "#ffcc5c", "shape": "round-rectangle"},
    {
        "label": "Cloud",
        "value": "cloud",
        "color": "#96ceb4",
        "shape": "round-rectangle",
    },
    {"label": "Database", "value": "database", "color": "#74b9ff", "shape": "barrel"},
    {
        "label": "Actor",
        "value": "actor",
        "color": "#fd79a8",
        "shape": "round-rectangle",
    },
]

# Default stylesheet for nodes and edges
default_stylesheet = [
    {
        "selector": "node",
        "style": {
            "content": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "data(color)",
            "color": "white",
            "border-width": 2,
            "border-color": "#2c5aa0",
            "width": "data(size)",
            "height": "data(size)",
            "font-size": "12px",
            "text-wrap": "wrap",
            "text-max-width": "80px",
            "shape": "data(shape)",
            "font-weight": "bold",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "data(curve_style)",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "data(color)",
            "line-color": "data(color)",
            "width": "data(width)",
            "content": "data(label)",
            "font-size": "10px",
            "text-rotation": "autorotate",
            "line-style": "data(line_style)",
        },
    },
    {
        "selector": ":selected",
        "style": {
            "background-color": "#ff6b6b",
            "border-color": "#ff5252",
            "line-color": "#ff6b6b",
            "target-arrow-color": "#ff6b6b",
            "border-width": 3,
        },
    },
    {
        "selector": ".highlighted",
        "style": {"background-color": "#ffd700", "border-color": "#ffcc00"},
    },
]

# Initial graph data with enhanced properties
initial_elements = [
    {
        "data": {
            "id": "start1",
            "label": "Start",
            "color": "#3174ad",
            "size": 60,
            "type": "start",
            "shape": "ellipse",
        },
        "position": {"x": 100, "y": 100},
    },
    {
        "data": {
            "id": "process1",
            "label": "Process Data",
            "color": "#27ae60",
            "size": 80,
            "type": "process",
            "shape": "rectangle",
        },
        "position": {"x": 300, "y": 100},
    },
    {
        "data": {
            "id": "decision1",
            "label": "Valid?",
            "color": "#f39c12",
            "size": 70,
            "type": "decision",
            "shape": "diamond",
        },
        "position": {"x": 500, "y": 100},
    },
    {
        "data": {
            "id": "end1",
            "label": "End",
            "color": "#e74c3c",
            "size": 60,
            "type": "end",
            "shape": "ellipse",
        },
        "position": {"x": 700, "y": 100},
    },
    {
        "data": {
            "id": "edge1",
            "source": "start1",
            "target": "process1",
            "label": "begin",
            "color": "#666",
            "width": 2,
            "curve_style": "bezier",
            "line_style": "solid",
        }
    },
    {
        "data": {
            "id": "edge2",
            "source": "process1",
            "target": "decision1",
            "label": "check",
            "color": "#666",
            "width": 2,
            "curve_style": "bezier",
            "line_style": "solid",
        }
    },
    {
        "data": {
            "id": "edge3",
            "source": "decision1",
            "target": "end1",
            "label": "yes",
            "color": "#27ae60",
            "width": 2,
            "curve_style": "bezier",
            "line_style": "solid",
        }
    },
]

# App layout
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1(
                    "CyEditor - Dash/Cytoscape Flow Chart Editor",
                    className="header-title",
                ),
                # Main toolbar
                html.Div(
                    [
                        html.Button(
                            "Add Node", id="add-node-btn", className="btn btn-primary"
                        ),
                        html.Button(
                            "Add Edge", id="add-edge-btn", className="btn btn-secondary"
                        ),
                        html.Button(
                            "Delete Selected",
                            id="delete-btn",
                            className="btn btn-danger",
                        ),
                        html.Button(
                            "Clear All", id="clear-btn", className="btn btn-warning"
                        ),
                        html.Button(
                            "Fit Graph", id="fit-btn", className="btn btn-info"
                        ),
                        html.Button(
                            "Undo",
                            id="undo-btn",
                            className="btn btn-secondary",
                            disabled=True,
                        ),
                        html.Button(
                            "Redo",
                            id="redo-btn",
                            className="btn btn-secondary",
                            disabled=True,
                        ),
                    ],
                    className="toolbar",
                ),
                # File operations
                html.Div(
                    [
                        html.Button(
                            "Export JSON", id="export-btn", className="btn btn-success"
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Import JSON"]),
                            className="upload-btn",
                            multiple=False,
                        ),
                        html.Button(
                            "Export PNG",
                            id="export-png-btn",
                            className="btn btn-secondary",
                        ),
                    ],
                    className="file-toolbar",
                ),
                # Layout and view controls
                html.Div(
                    [
                        html.Label("Layout:", className="control-label"),
                        dcc.Dropdown(
                            id="layout-dropdown",
                            options=[
                                {"label": "Grid", "value": "grid"},
                                {"label": "Random", "value": "random"},
                                {"label": "Circle", "value": "circle"},
                                {"label": "Concentric", "value": "concentric"},
                                {"label": "Breadthfirst", "value": "breadthfirst"},
                                {"label": "Cose", "value": "cose"},
                                {"label": "Cola", "value": "cola"},
                                {"label": "Dagre", "value": "dagre"},
                            ],
                            value="cose",
                            className="layout-dropdown",
                        ),
                        html.Label("Edge Style:", className="control-label"),
                        dcc.Dropdown(
                            id="edge-style-dropdown",
                            options=[
                                {"label": "Bezier", "value": "bezier"},
                                {"label": "Straight", "value": "straight"},
                                {"label": "Taxi", "value": "taxi"},
                                {"label": "Segments", "value": "segments"},
                            ],
                            value="bezier",
                            className="edge-style-dropdown",
                        ),
                    ],
                    className="controls",
                ),
            ],
            className="header",
        ),
        # Main content area
        html.Div(
            [
                # Left sidebar - Node types panel
                html.Div(
                    [
                        html.H3("Node Types", className="panel-title"),
                        html.Div(
                            [
                                html.Div([
                                    html.Div(
                                        node_type["label"],
                                        className="node-type-item",
                                        id={
                                            "type": "node-type",
                                            "index": node_type["value"],
                                        },
                                        style={
                                            "background-color": node_type["color"],
                                            "color": "white",
                                            "padding": "10px",
                                            "margin": "5px",
                                            "border-radius": "5px",
                                            "cursor": "pointer",
                                            "text-align": "center",
                                        },
                                    )
                                    for node_type in node_types
                                ])
                            ],
                            className="node-types-container",
                        ),
                        html.H3("Properties", className="panel-title"),
                        html.Div(
                            id="properties-panel",
                            children=[html.P("Select an element to edit properties")],
                        ),
                    ],
                    className="left-sidebar",
                ),
                # Main graph area
                html.Div(
                    [
                        cyto.Cytoscape(
                            id="cytoscape-graph",
                            elements=initial_elements,
                            stylesheet=default_stylesheet,
                            style={"width": "100%", "height": "600px"},
                            layout={"name": "cose"},
                            boxSelectionEnabled=True,
                            userPanningEnabled=True,
                            userZoomingEnabled=True,
                            autoungrabify=False,
                            autounselectify=False,
                        )
                    ],
                    className="graph-container",
                ),
                # Right sidebar - Navigator and info
                html.Div(
                    [
                        html.H3("Navigator", className="panel-title"),
                        html.Div(
                            id="navigator-info",
                            children=[html.P("Graph overview and navigation")],
                        ),
                        html.H3("Element Info", className="panel-title"),
                        html.Div(id="selected-info", children="No element selected"),
                    ],
                    className="right-sidebar",
                ),
            ],
            className="main-content",
        ),
        # Hidden components for state management
        dcc.Store(id="undo-stack", data=[]),
        dcc.Store(id="redo-stack", data=[]),
        dcc.Store(id="current-state", data=initial_elements),
        dcc.Store(id="selected-node-type", data="process"),
        dcc.Store(id="edge-mode", data=False),
        dcc.Store(id="edge-source", data=None),
        # Download component for exporting
        dcc.Download(id="download-json"),
        dcc.Download(id="download-png"),
    ],
    className="app-container",
)


# Callback to handle node type selection
@app.callback(
    Output("selected-node-type", "data"),
    [Input({"type": "node-type", "index": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def select_node_type(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        return "process"

    # Find which node type was clicked
    for i, clicks in enumerate(n_clicks_list):
        if clicks:
            return node_types[i]["value"]
    return "process"


# Callback to handle edge mode toggle
@app.callback(
    [
        Output("edge-mode", "data"),
        Output("edge-source", "data"),
        Output("add-edge-btn", "children"),
    ],
    [Input("add-edge-btn", "n_clicks")],
    [State("edge-mode", "data")],
    prevent_initial_call=True,
)
def toggle_edge_mode(n_clicks, current_edge_mode):
    if n_clicks:
        new_mode = not current_edge_mode
        button_text = "Cancel Edge" if new_mode else "Add Edge"
        return new_mode, None, button_text
    return current_edge_mode, None, "Add Edge"


# Main callback to update graph elements
@app.callback(
    [
        Output("cytoscape-graph", "elements"),
        Output("current-state", "data"),
        Output("undo-stack", "data"),
        Output("redo-stack", "data"),
        Output("edge-source", "data", allow_duplicate=True),
    ],
    [
        Input("add-node-btn", "n_clicks"),
        Input("delete-btn", "n_clicks"),
        Input("clear-btn", "n_clicks"),
        Input("upload-data", "contents"),
        Input("cytoscape-graph", "tapNodeData"),
        Input("edge-mode", "data"),
    ],
    [
        State("cytoscape-graph", "elements"),
        State("cytoscape-graph", "selectedNodeData"),
        State("cytoscape-graph", "selectedEdgeData"),
        State("selected-node-type", "data"),
        State("current-state", "data"),
        State("undo-stack", "data"),
        State("redo-stack", "data"),
        State("edge-source", "data"),
    ],
    prevent_initial_call=True,
)
def update_elements(
    add_clicks,
    delete_clicks,
    clear_clicks,
    upload_contents,
    tap_node_data,
    edge_mode,
    current_elements,
    selected_nodes,
    selected_edges,
    selected_node_type,
    current_state,
    undo_stack,
    redo_stack,
    edge_source,
):
    ctx = callback_context

    if not ctx.triggered:
        return current_elements, current_state, undo_stack, redo_stack, edge_source

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    new_undo_stack = undo_stack.copy()
    new_redo_stack = []  # Clear redo stack on new action

    # Save current state to undo stack
    if current_elements != current_state:
        new_undo_stack.append(current_state)
        if len(new_undo_stack) > 20:  # Limit undo stack size
            new_undo_stack.pop(0)

    if trigger_id == "cytoscape-graph" and tap_node_data and edge_mode:
        # Handle edge creation in edge mode
        tapped_node_id = tap_node_data["id"]

        if edge_source is None:
            # First click - set source node
            return (
                current_elements,
                current_state,
                undo_stack,
                redo_stack,
                tapped_node_id,
            )
        else:
            # Second click - create edge from source to target
            if edge_source != tapped_node_id:  # Don't create self-loops
                new_edge_id = f"edge_{uuid.uuid4().hex[:8]}"
                new_edge = {
                    "data": {
                        "id": new_edge_id,
                        "source": edge_source,
                        "target": tapped_node_id,
                        "label": "",
                        "color": "#666",
                        "width": 2,
                        "curve_style": "bezier",
                        "line_style": "solid",
                    }
                }
                new_elements = current_elements + [new_edge]
                return new_elements, new_elements, new_undo_stack, new_redo_stack, None
            else:
                # Reset if trying to create self-loop
                return current_elements, current_state, undo_stack, redo_stack, None

    elif trigger_id == "add-node-btn" and add_clicks:
        # Add a new node
        node_type_info = next(
            (nt for nt in node_types if nt["value"] == selected_node_type),
            node_types[1],
        )
        new_id = f"node_{uuid.uuid4().hex[:8]}"
        new_node = {
            "data": {
                "id": new_id,
                "label": f"{node_type_info['label']} {len([e for e in current_elements if 'source' not in e.get('data', {})])}",
                "color": node_type_info["color"],
                "size": 70,
                "type": selected_node_type,
                "shape": node_type_info["shape"],
            },
            "position": {
                "x": 200 + (add_clicks * 50) % 400,
                "y": 200 + (add_clicks * 50) % 400,
            },
        }
        new_elements = current_elements + [new_node]
        return new_elements, new_elements, new_undo_stack, new_redo_stack, edge_source

    elif trigger_id == "delete-btn" and delete_clicks:
        # Delete selected elements
        selected_ids = set()
        if selected_nodes:
            selected_ids.update([node["id"] for node in selected_nodes])
        if selected_edges:
            selected_ids.update([edge["id"] for edge in selected_edges])

        # Remove selected elements and edges connected to selected nodes
        filtered_elements = []
        for element in current_elements:
            element_id = element["data"]["id"]
            # Skip if element is selected
            if element_id in selected_ids:
                continue
            # Skip edges connected to selected nodes
            if "source" in element["data"] and (
                element["data"]["source"] in selected_ids
                or element["data"]["target"] in selected_ids
            ):
                continue
            filtered_elements.append(element)

        return (
            filtered_elements,
            filtered_elements,
            new_undo_stack,
            new_redo_stack,
            edge_source,
        )

    elif trigger_id == "clear-btn" and clear_clicks:
        # Clear all elements
        return [], [], new_undo_stack, new_redo_stack, None

    elif trigger_id == "upload-data" and upload_contents:
        # Import JSON data
        try:
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded.decode("utf-8"))
            if isinstance(data, list):
                return data, data, new_undo_stack, new_redo_stack, None
        except Exception as e:
            print(f"Error importing data: {e}")
            return current_elements, current_state, undo_stack, redo_stack, edge_source

    elif trigger_id == "edge-mode":
        # Reset edge source when edge mode changes
        return current_elements, current_state, undo_stack, redo_stack, None

    return current_elements, current_state, undo_stack, redo_stack, edge_source


# Callback to update layout
@app.callback(Output("cytoscape-graph", "layout"), [Input("layout-dropdown", "value")])
def update_layout(layout_name):
    return {"name": layout_name}


# Callback to update edge styles
@app.callback(
    Output("cytoscape-graph", "stylesheet"),
    [Input("edge-style-dropdown", "value")],
    [State("cytoscape-graph", "elements")],
)
def update_edge_style(edge_style, elements):
    # Update the stylesheet to change edge curve style
    new_stylesheet = default_stylesheet.copy()
    for style in new_stylesheet:
        if style["selector"] == "edge":
            style["style"]["curve-style"] = edge_style
    return new_stylesheet


# Callback to display selected element info and properties
@app.callback(
    [Output("selected-info", "children"), Output("properties-panel", "children")],
    [
        Input("cytoscape-graph", "selectedNodeData"),
        Input("cytoscape-graph", "selectedEdgeData"),
    ],
)
def display_selected_info(selected_nodes, selected_edges):
    info = []
    properties = []

    if selected_nodes:
        info.append(html.H4("Selected Nodes:"))
        for node in selected_nodes:
            info.append(
                html.P(
                    f"ID: {node['id']}, Label: {node.get('label', 'No label')}, Type: {node.get('type', 'unknown')}"
                )
            )

            # Add properties panel for node editing
            properties.extend([
                html.H4("Node Properties"),
                html.Label("Label:"),
                dcc.Input(
                    id={"type": "node-label", "index": node["id"]},
                    value=node.get("label", ""),
                    type="text",
                    className="property-input",
                ),
                html.Label("Size:"),
                dcc.Slider(
                    id={"type": "node-size", "index": node["id"]},
                    min=30,
                    max=150,
                    step=10,
                    value=node.get("size", 60),
                    marks={i: str(i) for i in range(30, 151, 30)},
                    className="property-slider",
                ),
                html.Label("Color:"),
                dcc.Input(
                    id={"type": "node-color", "index": node["id"]},
                    value=node.get("color", "#3174ad"),
                    type="text",
                    className="property-input",
                ),
            ])

    if selected_edges:
        info.append(html.H4("Selected Edges:"))
        for edge in selected_edges:
            info.append(
                html.P(f"ID: {edge['id']}, From: {edge['source']} To: {edge['target']}")
            )

            # Add properties panel for edge editing
            properties.extend([
                html.H4("Edge Properties"),
                html.Label("Label:"),
                dcc.Input(
                    id={"type": "edge-label", "index": edge["id"]},
                    value=edge.get("label", ""),
                    type="text",
                    className="property-input",
                ),
                html.Label("Width:"),
                dcc.Slider(
                    id={"type": "edge-width", "index": edge["id"]},
                    min=1,
                    max=10,
                    step=1,
                    value=edge.get("width", 2),
                    marks={i: str(i) for i in range(1, 11)},
                    className="property-slider",
                ),
                html.Label("Color:"),
                dcc.Input(
                    id={"type": "edge-color", "index": edge["id"]},
                    value=edge.get("color", "#666"),
                    type="text",
                    className="property-input",
                ),
            ])

    if not info:
        info = ["No element selected"]
        properties = [html.P("Select an element to edit properties")]

    return info, properties


# Callback for exporting data
@app.callback(
    Output("download-json", "data"),
    [Input("export-btn", "n_clicks")],
    [State("cytoscape-graph", "elements")],
)
def export_graph_data(n_clicks, elements):
    if n_clicks:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return dict(
            content=json.dumps(elements, indent=2),
            filename=f"flowchart_{timestamp}.json",
        )


# Callback for undo/redo button states
@app.callback(
    [Output("undo-btn", "disabled"), Output("redo-btn", "disabled")],
    [Input("undo-stack", "data"), Input("redo-stack", "data")],
)
def update_undo_redo_buttons(undo_stack, redo_stack):
    return len(undo_stack) == 0, len(redo_stack) == 0


# Separate callback for undo/redo functionality
@app.callback(
    [
        Output("cytoscape-graph", "elements", allow_duplicate=True),
        Output("undo-stack", "data", allow_duplicate=True),
        Output("redo-stack", "data", allow_duplicate=True),
        Output("current-state", "data", allow_duplicate=True),
    ],
    [Input("undo-btn", "n_clicks"), Input("redo-btn", "n_clicks")],
    [
        State("cytoscape-graph", "elements"),
        State("undo-stack", "data"),
        State("redo-stack", "data"),
    ],
    prevent_initial_call=True,
)
def handle_undo_redo(
    undo_clicks, redo_clicks, current_elements, undo_stack, redo_stack
):
    ctx = callback_context
    if not ctx.triggered:
        return current_elements, undo_stack, redo_stack, current_elements

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "undo-btn" and undo_clicks and undo_stack:
        # Undo operation
        previous_state = undo_stack[-1]
        new_undo_stack = undo_stack[:-1]
        new_redo_stack = redo_stack + [current_elements]
        return previous_state, new_undo_stack, new_redo_stack, previous_state

    elif trigger_id == "redo-btn" and redo_clicks and redo_stack:
        # Redo operation
        next_state = redo_stack[-1]
        new_redo_stack = redo_stack[:-1]
        new_undo_stack = undo_stack + [current_elements]
        return next_state, new_undo_stack, new_redo_stack, next_state

    return current_elements, undo_stack, redo_stack, current_elements


# Add CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            
            .app-container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                min-height: 100vh;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
            }
            
            .header-title {
                margin: 0 0 20px 0;
                font-size: 28px;
                font-weight: 300;
            }
            
            .toolbar, .file-toolbar {
                margin-bottom: 15px;
            }
            
            .btn {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                margin: 5px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            .btn-primary { background-color: #007bff; }
            .btn-secondary { background-color: #6c757d; }
            .btn-danger { background-color: #dc3545; }
            .btn-warning { background-color: #ffc107; color: #212529; }
            .btn-info { background-color: #17a2b8; }
            .btn-success { background-color: #28a745; }
            
            .btn:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .controls {
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .control-label {
                font-weight: bold;
                margin-right: 5px;
            }
            
            .layout-dropdown, .edge-style-dropdown {
                min-width: 120px;
            }
            
            .upload-btn {
                background-color: #17a2b8;
                color: white;
                padding: 10px 15px;
                border-radius: 6px;
                cursor: pointer;
                display: inline-block;
                margin: 5px;
                transition: all 0.3s ease;
            }
            
            .upload-btn:hover {
                background-color: #138496;
                transform: translateY(-2px);
            }
            
            .main-content {
                display: flex;
                min-height: 600px;
            }
            
            .left-sidebar, .right-sidebar {
                width: 280px;
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
                padding: 20px;
                overflow-y: auto;
                max-height: 600px;
            }
            
            .right-sidebar {
                border-right: none;
                border-left: 1px solid #dee2e6;
            }
            
            .graph-container {
                flex: 1;
                border: 1px solid #ddd;
                margin: 0;
                position: relative;
            }
            
            .panel-title {
                margin: 0 0 15px 0;
                font-size: 18px;
                font-weight: 600;
                color: #495057;
                border-bottom: 2px solid #007bff;
                padding-bottom: 5px;
            }
            
            .node-types-container {
                margin-bottom: 30px;
            }
            
            .node-type-item {
                transition: all 0.3s ease;
                border: 2px solid transparent;
            }
            
            .node-type-item:hover {
                transform: scale(1.05);
                border-color: white;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            .property-input {
                width: 100%;
                padding: 8px;
                margin: 5px 0 15px 0;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .property-slider {
                margin: 10px 0 20px 0;
            }
            
            #selected-info {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                margin-top: 10px;
            }
            
            #navigator-info {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                margin-bottom: 20px;
                text-align: center;
                color: #6c757d;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=8051)
