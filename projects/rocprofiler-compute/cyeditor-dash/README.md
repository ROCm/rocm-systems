# CyEditor - Dash/Cytoscape Flow Chart Editor

A modern, web-based flow chart editor built with Dash and Cytoscape, converted from the original [cyeditor](https://github.com/demonray/cyeditor) Vue.js project.

## Features

### ✅ Core Features
- **Interactive Flow Chart Editing**: Create, edit, and manipulate flow charts with ease
- **Enhanced Node Types**: 16 node types including basic flow chart shapes (Start, Process, Decision, End, Document, Data) and general geometric shapes (Circle, Square, Triangle, Hexagon, Octagon, Star, Note, Cloud, Database, Actor) inspired by diagrams.net
- **Advanced Edge Management**: Multiple edge styles (Bezier, Straight, Taxi, Segments) with customizable properties
- **Undo/Redo Functionality**: Full undo/redo support for all operations
- **Properties Panel**: Real-time editing of node and edge properties (label, size, color, width)
- **Multiple Layout Algorithms**: Grid, Random, Circle, Concentric, Breadthfirst, Cose, Cola, Dagre
- **Import/Export**: JSON import/export for saving and loading flow charts
- **Professional UI**: Modern interface with sidebars, toolbars, and responsive design

### 🎨 Visual Features
- **Node Type Selection**: Click-to-select node types with visual feedback
- **Real-time Property Editing**: Instant updates when modifying element properties
- **Selection Highlighting**: Visual feedback for selected elements
- **Responsive Design**: Adapts to different screen sizes
- **Professional Styling**: Modern gradient headers and smooth animations

### 🔧 Technical Features
- **State Management**: Comprehensive state management with undo/redo stacks
- **Element Management**: Add, delete, and modify nodes and edges
- **Layout Control**: Dynamic layout switching
- **File Operations**: Export to JSON with timestamps
- **Error Handling**: Robust error handling for file operations

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd cyeditor-dash
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python3 app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8050
   ```

## Usage Guide

### Getting Started

1. **Adding Nodes**:
   - Select a node type from the left sidebar (Start, Process, Decision, etc.)
   - Click the "Add Node" button in the toolbar
   - The new node will appear on the canvas

2. **Selecting Elements**:
   - Click on any node or edge to select it
   - Selected elements will be highlighted in red
   - Multiple elements can be selected using box selection

3. **Editing Properties**:
   - Select a node or edge
   - Use the Properties panel in the left sidebar to modify:
     - Label text
     - Size (for nodes)
     - Color
     - Width (for edges)

4. **Creating Connections**:
   - Click "Add Edge" button (manual edge creation)
   - Or use Cytoscape's built-in edge creation by dragging from node to node

### Toolbar Functions

| Button | Function |
|--------|----------|
| Add Node | Creates a new node of the selected type |
| Add Edge | Enables edge creation mode |
| Delete Selected | Removes selected elements |
| Clear All | Removes all elements from the canvas |
| Fit Graph | Fits the entire graph in the viewport |
| Undo | Undoes the last action |
| Redo | Redoes the last undone action |
| Export JSON | Downloads the current graph as JSON |
| Import JSON | Uploads a JSON file to load a graph |

### Layout Options

- **Grid**: Arranges nodes in a grid pattern
- **Random**: Places nodes randomly
- **Circle**: Arranges nodes in a circle
- **Concentric**: Creates concentric circles based on node hierarchy
- **Breadthfirst**: Tree-like layout based on connections
- **Cose**: Force-directed layout (default)
- **Cola**: Constraint-based layout
- **Dagre**: Directed graph layout

### Edge Styles

- **Bezier**: Smooth curved edges (default)
- **Straight**: Direct straight lines
- **Taxi**: Right-angled connections
- **Segments**: Multi-segment paths

## File Format

The application uses JSON format for import/export. Here's the structure:

```json
[
  {
    "data": {
      "id": "node1",
      "label": "Start",
      "color": "#3174ad",
      "size": 60,
      "type": "start",
      "shape": "ellipse"
    },
    "position": {"x": 100, "y": 100}
  },
  {
    "data": {
      "id": "edge1",
      "source": "node1",
      "target": "node2",
      "label": "connects",
      "color": "#666",
      "width": 2,
      "curve_style": "bezier",
      "line_style": "solid"
    }
  }
]
```

## Comparison with Original CyEditor

| Feature | Original CyEditor | Dash/Cytoscape Version |
|---------|-------------------|------------------------|
| Framework | Vue.js | Dash (Python) |
| Graph Library | Cytoscape.js | Dash Cytoscape |
| Node Types | ✅ | ✅ |
| Properties Editing | ✅ | ✅ |
| Undo/Redo | ✅ | ✅ |
| Import/Export | ✅ | ✅ |
| Multiple Layouts | ✅ | ✅ |
| Navigator | ✅ | 🔄 (Planned) |
| Context Menu | ✅ | 🔄 (Planned) |
| Copy/Paste | ✅ | 🔄 (Planned) |
| Grid Lines | ✅ | 🔄 (Planned) |
| Drag & Drop Nodes | ✅ | ✅ (Click-to-add) |

## Architecture

### Components Structure

```
cyeditor-dash/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components

1. **Main Layout**: Header with toolbars, three-panel layout (sidebar-graph-sidebar)
2. **Node Types Panel**: Interactive node type selection
3. **Properties Panel**: Dynamic property editing based on selection
4. **Graph Container**: Main Cytoscape component
5. **State Management**: Undo/redo stacks and current state tracking

### Callbacks

- **Element Management**: Add, delete, modify graph elements
- **Property Updates**: Real-time property editing
- **Layout Control**: Dynamic layout switching
- **File Operations**: Import/export functionality
- **Undo/Redo**: State management for operations

## Customization

### Adding New Node Types

To add new node types, modify the `node_types` list in `app.py`:

```python
node_types = [
    # ... existing types
    {'label': 'Custom', 'value': 'custom', 'color': '#ff6b6b', 'shape': 'hexagon'}
]
```

### Styling

The application uses embedded CSS in the `app.index_string`. Modify the styles to customize:
- Colors and themes
- Layout dimensions
- Button styles
- Panel appearances

### Adding Features

The modular callback structure makes it easy to add new features:
1. Add UI components to the layout
2. Create corresponding callbacks
3. Update state management as needed

## Dependencies

- **dash**: Web application framework
- **dash-cytoscape**: Cytoscape.js integration for Dash
- **pandas**: Data manipulation (optional)
- **plotly**: Plotting library (comes with Dash)

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Performance

- Optimized for graphs with up to 1000 nodes
- Efficient state management with limited undo stack
- Responsive UI with smooth animations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is inspired by the original [cyeditor](https://github.com/demonray/cyeditor) project. Please refer to the original project's license for any licensing considerations.

## Roadmap

### Planned Features
- [ ] Context menu functionality
- [ ] Copy/paste operations
- [ ] Grid lines and snap-to-grid
- [ ] Navigator panel with minimap
- [ ] Export to PNG/SVG
- [ ] Keyboard shortcuts
- [ ] Node resizing handles
- [ ] Edge editing with control points
- [ ] Custom node shapes
- [ ] Themes and styling options

### Future Enhancements
- [ ] Real-time collaboration
- [ ] Plugin system
- [ ] Advanced layout algorithms
- [ ] Animation and transitions
- [ ] Touch/mobile support
- [ ] Accessibility improvements

## Support

For issues, questions, or contributions, please create an issue in the repository or contact the maintainers.

---

**Note**: This is a conversion of the original Vue.js cyeditor to Dash/Cytoscape. While it maintains most of the core functionality, some features are implemented differently due to the framework differences.
