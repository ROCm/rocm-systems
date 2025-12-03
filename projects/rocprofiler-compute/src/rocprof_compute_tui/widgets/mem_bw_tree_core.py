from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Static


class TreeNode:
    """Represents a node in the decision tree."""

    def __init__(
        self, label: str, metadata: str = "", children: list["TreeNode"] | None = None
    ) -> None:
        self.label = label
        self.metadata = metadata
        self.children: list[TreeNode] = children or []
        self.expanded: bool = True
        self.x: int = 0
        self.y: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        node = cls(data.get("label", ""), data.get("metadata", ""))
        for child_data in data.get("children", []):
            node.children.append(cls.from_dict(child_data))
        return node

    # --- unified width helpers (match drawing exactly) ---
    def _label_len(self) -> int:
        # what we put between the spaces inside the box
        return (
            len(self.label)
            + (1 if self.label and self.metadata else 0)
            + len(self.metadata)
        )

    def _box_total_width(self) -> int:
        return self._label_len() + 4

    def calculate_positions(self, x: int = 0, y: int = 0) -> int:
        """Place this node and children. Return subtree total width."""
        self.y = y
        node_width = self._box_total_width()

        if not self.children or not self.expanded:
            self.x = x
            return node_width

        HSPACE = 4
        VSPACE = 7

        # position children
        child_x = x
        total_width = 0
        for child in self.children:
            w = child.calculate_positions(child_x, y + VSPACE)
            total_width += w
            child_x += w + HSPACE

        if len(self.children) > 1:
            total_width += HSPACE * (len(self.children) - 1)

        total_width = max(total_width, node_width)
        self.x = x + (total_width - node_width) // 2
        return total_width


class TreeCanvas(Static):
    """Canvas for rendering the tree (refactored; behavior unchanged)."""

    def __init__(self, root: TreeNode) -> None:
        super().__init__()
        self.root = root
        self.selected = root
        self._cached_width = 100
        self._cached_height = 40

    # ---- helpers that match TreeNode width math ----
    def _label_len(self, n: TreeNode) -> int:
        return len(n.label) + (1 if n.label and n.metadata else 0) + len(n.metadata)

    def _inner_width(self, n: TreeNode) -> int:
        return self._label_len(n) + 2  # the inside of the box (spaces + text)

    # ---------- small helpers ----------

    @staticmethod
    def _box(label: str) -> tuple[str, str, str, int]:
        """Return (top, mid, bot, inner_width) for the label box."""
        w = len(label) + 2
        top = "┌" + "─" * w + "┐"
        mid = "│" + f" {label} " + "│"
        bot = "└" + "─" * w + "┘"
        return top, mid, bot, w

    def _center_x(self, n: TreeNode, ox: int) -> int:
        return n.x + ox + (n._box_total_width() // 2)

    # ---------- sizing ----------

    def _get_depth(self, node: TreeNode) -> int:
        if not node.children or not node.expanded:
            return 0
        return 1 + max(self._get_depth(c) for c in node.children)

    # ---- measuring via the text we render (no content_* needed) ----
    def _max_bottom_y(self, n: TreeNode) -> int:
        bottom = n.y + 2
        if n.children and n.expanded:
            for c in n.children:
                bottom = max(bottom, self._max_bottom_y(c))
        return bottom

    def _measure_tree(self) -> tuple[int, int]:
        total_width = self.root.calculate_positions(0, 0)
        height = self._max_bottom_y(self.root) + 5
        width = total_width + 8
        self._cached_width, self._cached_height = width, height
        return width, height

    # ---------- rendering ----------

    def render(self) -> Text:
        # compute tree layout and the centering offset WITHOUT mutating node.x
        total_width = self.root.calculate_positions(0, 0)
        pref_w, pref_h = self._measure_tree()
        viewport_w = getattr(self.size, "width", 0) or 0
        x_offset = max(0, (viewport_w - total_width) // 2)

        # build canvas
        canvas: list[list[tuple[str, str]]] = [
            [(" ", "") for _ in range(pref_w)] for _ in range(pref_h)
        ]

        # draw
        self._draw_node(canvas, self.root, x_offset)

        # to Rich Text
        out = Text()
        for row in canvas:
            for ch, style in row:
                out.append(ch, style or None)
            out.append("\n")
        return out

    def _box_parts(self, node: TreeNode) -> tuple[str, str, str, int]:
        """Return (top, mid, bot, inner_w) for node; rounded for leaves."""
        is_leaf = not node.children  # leaf = has no children
        label_text = f"{node.label} {node.metadata}".strip()
        inner_w = (
            len(node.label)
            + (1 if node.label and node.metadata else 0)
            + len(node.metadata)
            + 2
        )
        if is_leaf:
            # rounded corners for leaves
            top = "╭" + "─" * inner_w + "╮"
            mid = "│" + f" {label_text} " + "│"
            bot = "╰" + "─" * inner_w + "╯"
        else:
            # square corners for non-leaves
            top = "┌" + "─" * inner_w + "┐"
            mid = "│" + f" {label_text} " + "│"
            bot = "└" + "─" * inner_w + "┘"
        return top, mid, bot, inner_w

    def _draw_node(
        self, canvas: list[list[tuple[str, str]]], node: TreeNode, ox: int
    ) -> None:
        label_text = f"{node.label} {node.metadata}".strip()
        inner_w = self._inner_width(node)
        top = "┌" + "─" * inner_w + "┐"
        mid = "│" + f" {label_text} " + "│"
        bot = "└" + "─" * inner_w + "┘"
        style = (
            "bold cyan"
            if node is self.selected
            else ("dim" if (not node.expanded and node.children) else "")
        )

        top, mid, bot, inner_w = self._box_parts(node)
        bx = node.x + ox
        by = node.y
        self._put(canvas, bx, by, top, style)
        self._put(canvas, bx, by + 1, mid, style)
        self._put(canvas, bx, by + 2, bot, style)

        if node is self.selected:
            self._put(canvas, bx - 1, by + 1, "►", "bold yellow")

        if node.children and node.expanded:
            self._draw_connectors(canvas, node, ox, inner_w)
            for child in node.children:
                self._draw_node(canvas, child, ox)

    def _draw_connectors(
        self, canvas: list[list[tuple[str, str]]], node: TreeNode, ox: int, inner_w: int
    ) -> None:
        if not node.children:
            return

        node_center = self._center_x(node, ox)
        stub_y = node.y + 3
        STUB = 2

        # 2-row vertical stub below the parent box
        for i in range(STUB):
            self._put(canvas, node_center, stub_y + i, "│", "")

        if not node.expanded:
            return

        bar_y = stub_y + STUB
        child_centers = [self._center_x(c, ox) for c in node.children]

        # ---------- Single child ----------
        if len(child_centers) == 1:
            c = child_centers[0]
            if c == node_center:
                # Perfectly aligned: continue vertical with NO gap
                for y in range(bar_y, node.children[0].y):
                    self._put(canvas, c, y, "│", "")
                return

            # offset single-child: draw a short bar and use CORNERS at both ends
            left, right = (node_center, c) if node_center < c else (c, node_center)
            for x in range(left + 1, right):
                self._put(canvas, x, bar_y, "─", "")
            if c > node_center:
                self._put(canvas, node_center, bar_y, "└", "")
                self._put(canvas, c, bar_y, "┐", "")
            else:
                self._put(canvas, node_center, bar_y, "┘", "")
                self._put(canvas, c, bar_y, "┌", "")
            # vertical drop from child
            for y in range(bar_y + 1, node.children[0].y):
                self._put(canvas, c, y, "│", "")
            return

        # ---------- Multiple children ----------
        left_c, right_c = min(child_centers), max(child_centers)

        # draw the bar across the span
        for x in range(left_c, right_c + 1):
            self._put(canvas, x, bar_y, "─", "")

        # parent junction connects UP to stub; keep a tee at center
        self._put(canvas, node_center, bar_y, "┴", "")

        # children junctions: corners at the ENDS, tees for interior drops,
        # continuous vertical if child is exactly at parent center
        for idx, (c, child) in enumerate(zip(child_centers, node.children)):
            if c == node_center:
                # centered child: continue the vertical through the bar row
                self._put(canvas, c, bar_y, "│", "")
            elif c == left_c:
                self._put(canvas, c, bar_y, "┌", "")  # right+down
            elif c == right_c:
                self._put(canvas, c, bar_y, "┐", "")  # left+down
            else:
                self._put(canvas, c, bar_y, "┬", "")  # interior tee-down

            # drop to child box
            for y in range(bar_y + 1, child.y):
                self._put(canvas, c, y, "│", "")

    # ---------- canvas utilities ----------

    def _put(
        self,
        canvas: list[list[tuple[str, str]]],
        x: int,
        y: int,
        text: str,
        style: str = "",
    ) -> None:
        """Put text with line-merge semantics so crossings form proper tees/corners."""
        if not (0 <= y < len(canvas)):
            return
        row = canvas[y]
        width = len(row)

        # Map glyphs to direction sets
        char2dirs = {
            " ": set(),
            "─": {"l", "r"},
            "│": {"u", "d"},
            "┬": {"l", "r", "d"},
            "┴": {"l", "r", "u"},
            # rounded corners for leaves
            "╭": {"r", "d"},
            "╮": {"l", "d"},
            "╰": {"r", "u"},
            "╯": {"l", "u"},
            # vertical/horizontal arrows occasionally used by fonts—treat as vertical
            "↑": {"u", "d"},
            "↓": {"u", "d"},
        }
        # Reverse map
        dirs2char = {
            frozenset(): " ",
            frozenset({"l", "r"}): "─",
            frozenset({"u", "d"}): "│",
            frozenset({"l", "r", "d"}): "┬",
            frozenset({"l", "r", "u"}): "┴",
        }
        rounded = {"╭", "╮", "╰", "╯"}

        def merge(old: str, new: str) -> str:
            if new == " ":
                return old
            if old == " ":
                return new

            # Prefer a tee over a cross when a bar meets a single-direction vertical.
            if (old, new) in {("─", "│"), ("│", "─")}:
                return "┬"

            # keep box corners if a line brushes them
            if old in rounded | {"┌", "┐", "└", "┘"} and new in {"─", "│"}:
                return old
            if new in rounded | {"┌", "┐", "└", "┘"} and old in {"─", "│"}:
                return new

            # general union of directions
            d_old = char2dirs.get(old, set())
            d_new = char2dirs.get(new, set())
            combo = frozenset(d_old | d_new)
            return dirs2char.get(combo, new)

        for i, ch in enumerate(text):
            xi = x + i
            if 0 <= xi < width:
                prev_ch, prev_style = row[xi]
                row[xi] = (merge(prev_ch, ch), style or prev_style)

    def _apply_offset(self, node: TreeNode, dx: int) -> None:
        node.x += dx
        for child in node.children:
            self._apply_offset(child, dx)

    # ---------- interaction ----------

    def toggle(self) -> None:
        """Toggle expansion of selected node safely."""
        if self.selected.children:
            self.selected.expanded = not self.selected.expanded
            # Recalculate layout fully and refresh safely
            self.root.calculate_positions(0, 0)
            self.refresh(layout=True)

    def _find_parent(
        self, target: TreeNode, root: TreeNode | None = None
    ) -> TreeNode | None:
        root = self.root if root is None else root
        if target is root:
            return None
        for child in root.children:
            if child is target:
                return root
            found = self._find_parent(target, child)
            if found:
                return found
        return None

    def navigate(self, direction: str) -> None:
        parent = self._find_parent(self.selected)
        if direction == "down" and self.selected.children and self.selected.expanded:
            self.selected = self.selected.children[0]
        elif direction == "up" and parent:
            self.selected = parent
        elif direction in ("left", "right") and parent:
            sibs = parent.children
            idx = sibs.index(self.selected)
            step = -1 if direction == "left" else 1
            nxt = idx + step
            if 0 <= nxt < len(sibs):
                self.selected = sibs[nxt]
        self.refresh()

    async def on_resize(self) -> None:
        self.refresh()

    async def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse click to select a node."""
        x, y = event.x, event.y
        # account for any scroll offset
        scroll_x, scroll_y = self.scroll_offset
        x += scroll_x
        y += scroll_y

        # find the clicked node
        clicked = self._find_node_at(self.root, x, y)
        if clicked:
            self.selected = clicked
            self.refresh()

    def _find_node_at(self, node: TreeNode, x: int, y: int) -> TreeNode | None:
        """Recursively find which node box was clicked."""
        # box boundaries (3 rows high)
        box_left = node.x
        box_top = node.y
        box_right = node.x + node._box_total_width()
        box_bottom = node.y + 2

        if box_left <= x <= box_right and box_top <= y <= box_bottom:
            return node

        if node.children and node.expanded:
            for c in node.children:
                found = self._find_node_at(c, x, y)
                if found:
                    return found
        return None


class DecisionTreeApp(App):
    """A decision tree viewer using Textual."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle", "Expand/Collapse"),
        Binding("w", "nav('up')", "Up"),
        Binding("s", "nav('down')", "Down"),
        Binding("a", "nav('left')", "Left"),
        Binding("d", "nav('right')", "Right"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }

    #tree-scroll {
        width: 1fr;
        height: 1fr;
        overflow-x: auto;
        overflow-y: auto;
        border: solid $primary;
    }

    #tree-scroll > * {
        width: auto;   /* allows real content width */
    }
    """

    def __init__(self, tree_data: dict[str, Any]) -> None:
        super().__init__()
        self.root = TreeNode.from_dict(tree_data)

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="tree-scroll"):
            yield TreeCanvas(self.root)
        yield Footer()

    def action_toggle(self) -> None:
        self.query_one(TreeCanvas).toggle()

    def action_nav(self, direction: str) -> None:
        self.query_one(TreeCanvas).navigate(direction)


mem_bw_decision_tree_dict = {
    "label": "Workload BW Measurement",
    "children": [
        {
            "label": "High EA -> L2 Backpressure",
            "children": [
                {"label": "SoC Tunning"},
                {
                    "label": "High TCR -> TCP Backpressure",
                    "children": [
                        {
                            "label": "vL1d Hit Low",
                            "children": [
                                {
                                    "label": "Non-Temporal set",
                                    "children": [
                                        {"label": "Try non-temporal (NT)"},
                                        {
                                            "label": "High TC -> TA Backpressure",
                                            "children": [
                                                {
                                                    "label": "High TA -> VMEM Backpressure",
                                                    "children": [
                                                        {"label": "SW optimization"},
                                                        {
                                                            "label": "Bottleneck: Workload"
                                                        },
                                                    ],
                                                },
                                                {
                                                    "label": "High vL1d set full stall",
                                                    "children": [
                                                        {
                                                            "label": "Tagram hotspotting",
                                                            "children": [
                                                                {
                                                                    "label": "Bottleneck: TCP capacity"
                                                                },
                                                                {
                                                                    "label": "SW optimization"
                                                                },
                                                                {
                                                                    "label": "UTCL1 Stall",
                                                                    "children": [
                                                                        {
                                                                            "label": "UTCL1 Latency",
                                                                            "children": [
                                                                                {
                                                                                    "label": "Bottleneck: UTCL2"
                                                                                }
                                                                            ],
                                                                        }
                                                                    ],
                                                                },
                                                            ],
                                                        }
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                }
                            ],
                        },
                        {
                            "label": "L2 Channel hotspotting",
                            "children": [
                                {"label": "SW optimization"},
                                {
                                    "label": "L2 Hit Low",
                                    "children": [
                                        {"label": "Try non-temporal (NT)"},
                                        {
                                            "label": "Write Traffic",
                                            "children": [
                                                {"label": "Bottleneck: L2 Arb"},
                                                {"label": "Unlikely"},
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ],
}

# Example usage
if __name__ == "__main__":
    # with open("src/rocprof_compute_tui/utils/mem_bw_decision_tree.yaml") as f:
    #    mem_bw_decision_tree_dict = yaml.safe_load(f)

    app = DecisionTreeApp(mem_bw_decision_tree_dict)
    app.run()
