from __future__ import annotations

from typing import Any, Optional

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static


def _slugify(label: str) -> str:
    """Best-effort stable id when node_id isn't provided."""
    out = []
    last_us = False
    for ch in label.strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_us = False
        else:
            if not last_us:
                out.append("_")
                last_us = True
    s = "".join(out).strip("_")
    return s or "node"


class TreeNode:
    """Represents a node in the decision tree."""

    def __init__(
        self,
        label: str,
        metadata: str = "",
        children: Optional[list["TreeNode"]] = None,
        node_id: Optional[str] = None,
    ) -> None:
        self.label = label
        self.metadata = metadata
        self.children: list[TreeNode] = children or []
        self.expanded: bool = True
        self.x: int = 0
        self.y: int = 0

        # Stable identity + scalable styling flags.
        self.node_id: str = node_id or _slugify(label)
        self.tags: set[str] = set()  # e.g. {"active", "dim", "warn"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        node = cls(
            data.get("label", ""),
            data.get("metadata", ""),
            node_id=data.get("node_id") or data.get("id"),
        )
        for child_data in data.get("children", []):
            node.children.append(cls.from_dict(child_data))
        return node

    def find_by_id(self, node_id: str) -> Optional["TreeNode"]:
        if self.node_id == node_id:
            return self
        for c in self.children:
            found = c.find_by_id(node_id)
            if found:
                return found
        return None

    def walk(self) -> list["TreeNode"]:
        out = [self]
        for c in self.children:
            out.extend(c.walk())
        return out

    # --- unified width helpers (match drawing exactly) ---
    def _label_len(self) -> int:
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


class NodeSelected(Message):
    """Emitted when a node becomes selected in the TreeCanvas."""

    def __init__(self, sender: "TreeCanvas", node_id: str) -> None:
        super().__init__()
        self.sender = sender
        self.node_id = node_id


class TreeCanvas(Static):
    """Canvas for rendering the tree (refactored; behavior unchanged)."""

    def __init__(self, root: TreeNode, *, id: str = "mem-bw-tree-canvas") -> None:
        super().__init__(id=id)
        self.root = root
        self.selected = root
        self._cached_width = 100
        self._cached_height = 40

    # ---- helpers that match TreeNode width math ----
    def _label_len(self, n: TreeNode) -> int:
        return len(n.label) + (1 if n.label and n.metadata else 0) + len(n.metadata)

    def _inner_width(self, n: TreeNode) -> int:
        return self._label_len(n) + 2

    @staticmethod
    def _box(label: str) -> tuple[str, str, str, int]:
        w = len(label) + 2
        top = "┌" + "─" * w + "┐"
        mid = "│" + f" {label} " + "│"
        bot = "└" + "─" * w + "┘"
        return top, mid, bot, w

    def _center_x(self, n: TreeNode, ox: int) -> int:
        return n.x + ox + (n._box_total_width() // 2)

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

    def render(self) -> Text:
        total_width = self.root.calculate_positions(0, 0)
        pref_w, pref_h = self._measure_tree()
        viewport_w = getattr(self.size, "width", 0) or 0
        x_offset = max(0, (viewport_w - total_width) // 2)

        canvas: list[list[tuple[str, str]]] = [
            [(" ", "") for _ in range(pref_w)] for _ in range(pref_h)
        ]

        self._draw_node(canvas, self.root, x_offset)

        out = Text(no_wrap=True)
        for row in canvas:
            for ch, style in row:
                out.append(ch, style or None)
            out.append("\n")
        return out

    def _box_parts(self, node: TreeNode) -> tuple[str, str, str, int]:
        is_leaf = not node.children
        label_text = f"{node.label} {node.metadata}".strip()
        inner_w = (
            len(node.label)
            + (1 if node.label and node.metadata else 0)
            + len(node.metadata)
            + 2
        )
        if is_leaf:
            top = "╭" + "─" * inner_w + "╮"
            mid = "│" + f" {label_text} " + "│"
            bot = "╰" + "─" * inner_w + "╯"
        else:
            top = "┌" + "─" * inner_w + "┐"
            mid = "│" + f" {label_text} " + "│"
            bot = "└" + "─" * inner_w + "┘"
        return top, mid, bot, inner_w

    def _node_style(self, node: TreeNode) -> str:
        # Selected has top priority.
        if node is self.selected:
            return "bold cyan"

        # Decision-engine tags.
        if "warn" in node.tags:
            return "bold yellow"
        if "active" in node.tags:
            return "bold green"
        if "dim" in node.tags:
            return "dim"

        # Existing collapsed style.
        if (not node.expanded and node.children):
            return "dim"

        return ""

    def _draw_node(
        self, canvas: list[list[tuple[str, str]]], node: TreeNode, ox: int
    ) -> None:
        top, mid, bot, inner_w = self._box_parts(node)
        bx = node.x + ox
        by = node.y
        style = self._node_style(node)

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

        for i in range(STUB):
            self._put(canvas, node_center, stub_y + i, "│", "")

        if not node.expanded:
            return

        bar_y = stub_y + STUB
        child_centers = [self._center_x(c, ox) for c in node.children]

        if len(child_centers) == 1:
            c = child_centers[0]
            if c == node_center:
                for y in range(bar_y, node.children[0].y):
                    self._put(canvas, c, y, "│", "")
                return

            left, right = (node_center, c) if node_center < c else (c, node_center)
            for x in range(left + 1, right):
                self._put(canvas, x, bar_y, "─", "")
            if c > node_center:
                self._put(canvas, node_center, bar_y, "└", "")
                self._put(canvas, c, bar_y, "┐", "")
            else:
                self._put(canvas, node_center, bar_y, "┘", "")
                self._put(canvas, c, bar_y, "┌", "")
            for y in range(bar_y + 1, node.children[0].y):
                self._put(canvas, c, y, "│", "")
            return

        left_c, right_c = min(child_centers), max(child_centers)

        for x in range(left_c, right_c + 1):
            self._put(canvas, x, bar_y, "─", "")

        self._put(canvas, node_center, bar_y, "┴", "")

        for idx, (c, child) in enumerate(zip(child_centers, node.children)):
            if c == node_center:
                self._put(canvas, c, bar_y, "│", "")
            elif c == left_c:
                self._put(canvas, c, bar_y, "┌", "")
            elif c == right_c:
                self._put(canvas, c, bar_y, "┐", "")
            else:
                self._put(canvas, c, bar_y, "┬", "")

            for y in range(bar_y + 1, child.y):
                self._put(canvas, c, y, "│", "")

    def _put(
        self,
        canvas: list[list[tuple[str, str]]],
        x: int,
        y: int,
        text: str,
        style: str = "",
    ) -> None:
        if not (0 <= y < len(canvas)):
            return
        row = canvas[y]
        width = len(row)

        char2dirs = {
            " ": set(),
            "─": {"l", "r"},
            "│": {"u", "d"},
            "┬": {"l", "r", "d"},
            "┴": {"l", "r", "u"},
            "╭": {"r", "d"},
            "╮": {"l", "d"},
            "╰": {"r", "u"},
            "╯": {"l", "u"},
            "↑": {"u", "d"},
            "↓": {"u", "d"},
        }
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

            if (old, new) in {("─", "│"), ("│", "─")}:
                return "┬"

            if old in rounded | {"┌", "┐", "└", "┘"} and new in {"─", "│"}:
                return old
            if new in rounded | {"┌", "┐", "└", "┘"} and old in {"─", "│"}:
                return new

            d_old = char2dirs.get(old, set())
            d_new = char2dirs.get(new, set())
            combo = frozenset(d_old | d_new)
            return dirs2char.get(combo, new)

        for i, ch in enumerate(text):
            xi = x + i
            if 0 <= xi < width:
                prev_ch, prev_style = row[xi]
                row[xi] = (merge(prev_ch, ch), style or prev_style)

    # ---------- interaction ----------

    def _set_selected(self, node: TreeNode) -> None:
        if node is self.selected:
            return
        self.selected = node
        self.post_message(NodeSelected(self, node.node_id))
        self.refresh()

    def toggle(self) -> None:
        if self.selected.children:
            self.selected.expanded = not self.selected.expanded
            self.root.calculate_positions(0, 0)
            self.refresh(layout=True)

    def _find_parent(
        self, target: TreeNode, root: Optional[TreeNode] = None
    ) -> Optional[TreeNode]:
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
        nxt: Optional[TreeNode] = None

        if direction == "down" and self.selected.children and self.selected.expanded:
            nxt = self.selected.children[0]
        elif direction == "up" and parent:
            nxt = parent
        elif direction in ("left", "right") and parent:
            sibs = parent.children
            idx = sibs.index(self.selected)
            step = -1 if direction == "left" else 1
            nidx = idx + step
            if 0 <= nidx < len(sibs):
                nxt = sibs[nidx]

        if nxt is not None:
            self._set_selected(nxt)
        else:
            self.refresh()

    async def on_resize(self) -> None:
        self.refresh()

    async def on_mount(self) -> None:
        # Ensure initial selection is communicated.
        self.post_message(NodeSelected(self, self.selected.node_id))

    async def on_mouse_down(self, event: events.MouseDown) -> None:
        x, y = event.x, event.y
        scroll_x, scroll_y = self.scroll_offset
        x += scroll_x
        y += scroll_y

        clicked = self._find_node_at(self.root, x, y)
        if clicked:
            self._set_selected(clicked)

    def _find_node_at(self, node: TreeNode, x: int, y: int) -> Optional[TreeNode]:
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
