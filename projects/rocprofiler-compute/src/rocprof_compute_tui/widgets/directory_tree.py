"""
Specialized Widget Modules
-------------------------
Contains custom widget implementations for the application.
"""

from textual.widgets import DirectoryTree


class FolderOnlyDirectory(DirectoryTree):
    """Directory tree that only shows folders."""

    def filter_paths(self, paths):
        """Filter to only show directories."""
        return [path for path in paths if path.is_dir()]
