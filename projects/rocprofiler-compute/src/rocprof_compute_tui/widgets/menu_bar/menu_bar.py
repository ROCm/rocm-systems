from textual import on
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Button

from rocprof_compute_tui.widgets.recent_directories import RecentDirectoriesScreen


class DropdownMenu(Container):
    def compose(self):
        """Compose the dropdown menu with menu items."""
        yield Button("Open Workload", id="menu-open-workload", classes="menu-item")
        yield Button("Open Recent", id="menu-open-recent", classes="menu-item")
        # TODO:
        # yield Button("Attach", id="menu-attach", classes="menu-item")
        yield Button("Exit", id="menu-exit", classes="menu-item")

    def on_mount(self) -> None:
        self.add_class("hidden")


class MenuButton(Button):
    is_open = reactive(False)

    def __init__(self, label, menu_id, *args, **kwargs):
        super().__init__(label, *args, **kwargs)
        self.menu_id = menu_id

    def on_click(self):
        self.is_open = not self.is_open
        dropdown = self.app.query_one(f"#{self.menu_id}", DropdownMenu)

        if self.is_open:
            dropdown.remove_class("hidden")
        else:
            dropdown.add_class("hidden")


class MenuBar(Container):
    """A menu bar that spans the width of the app."""

    def compose(self):
        yield Horizontal(
            MenuButton("File", "file-dropdown", id="menu-file"), id="menu-buttons"
        )

        with Container(id="dropdown-container"):
            yield DropdownMenu(id="file-dropdown")

    def on_mount(self):
        self.border_title = "MENU BAR"
        self.add_class("section")
        self.parent_main_view = self.screen.query_one("#main-container", Horizontal)

    @on(Button.Pressed, "#menu-open-recent")
    def show_recent(self):
        if not self.app.recent_dirs:
            self.notify("No recent directories found", severity="warning")
            return

        def on_recent_selected(selected_dir):
            if selected_dir:
                self.parent_main_view.selected_path = selected_dir
                self.query_one("#file-dropdown", DropdownMenu).add_class("hidden")
                self.parent_main_view.run_analysis()

        self.app.push_screen(
            RecentDirectoriesScreen(self.app.recent_dirs), on_recent_selected
        )

    @on(Button.Pressed, "#menu-exit")
    def exit_app(self):
        self.app.exit()
