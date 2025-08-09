# This file is part of the Soma Cube game.
# Copyright (C) 2025 Bishoy Wadea
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from gettext import gettext as _

from sugar3.activity import activity
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import ActivityToolbarButton, StopButton
from sugar3.graphics.toolbutton import ToolButton
from sugar3.graphics.palette import Palette
from sugar3.graphics import style

from gi.repository import Gtk, Gdk, GLib

from glview import GLView


class SomaCube(activity.Activity):
    def __init__(self, handle):
        activity.Activity.__init__(self, handle)

        self._create_toolbar()
        self._setup_content()

    def _create_toolbar(self):
        toolbar_box = ToolbarBox()

        # Activity button
        activity_button = ActivityToolbarButton(self)
        toolbar_box.toolbar.insert(activity_button, -1)

        # Separator
        separator = Gtk.SeparatorToolItem()
        separator.props.draw = False
        toolbar_box.toolbar.insert(separator, -1)

        # Reset button
        self.reset_button = ToolButton("emblem-busy")
        self.reset_button.set_tooltip("Reset")
        self.reset_button.connect("clicked", self._on_play_again_clicked)
        toolbar_box.toolbar.insert(self.reset_button, -1)
        self.reset_button.show()

        # Help button
        self.help_button = ToolButton("toolbar-help")
        self.help_button.set_tooltip("Help")
        self._setup_help_palette()
        toolbar_box.toolbar.insert(self.help_button, -1)
        self.help_button.show()

        # Separator
        separator = Gtk.SeparatorToolItem()
        separator.props.draw = False
        separator.set_expand(True)
        toolbar_box.toolbar.insert(separator, -1)

        # Stop button
        stop_button = StopButton(self)
        toolbar_box.toolbar.insert(stop_button, -1)
        stop_button.show()

        self.set_toolbar_box(toolbar_box)
        toolbar_box.show_all()

    def _setup_help_palette(self):
        """Create a Sugar-style help palette"""
        palette = Palette('Help')
        palette.props.primary_text = 'Soma Cube Game Help'
        palette.props.secondary_text = 'Learn how to play the Soma Cube puzzle game'

        # Create help content container
        help_content = Gtk.VBox(spacing=style.DEFAULT_SPACING)
        help_content.set_border_width(style.DEFAULT_SPACING)

        # Game description
        desc_label = Gtk.Label()
        desc_label.set_markup('<b>About Soma Cube:</b>')
        desc_label.set_alignment(0, 0.5)
        help_content.pack_start(desc_label, False, False, 0)

        desc_text = Gtk.Label()
        desc_text.set_text('The Soma cube is a 3D puzzle where you must arrange\n'
                           '7 different pieces to form a 3×3×3 cube.')
        desc_text.set_alignment(0, 0.5)
        desc_text.set_line_wrap(True)
        desc_text.set_max_width_chars(style.MENU_WIDTH_CHARS)
        help_content.pack_start(desc_text, False, False, 0)

        # Controls section
        controls_label = Gtk.Label()
        controls_label.set_markup('<b>Controls:</b>')
        controls_label.set_alignment(0, 0.5)
        help_content.pack_start(
            controls_label,
            False,
            False,
            style.DEFAULT_SPACING)

        # Create controls grid
        controls_grid = Gtk.Grid()
        controls_grid.set_column_spacing(style.DEFAULT_SPACING)
        controls_grid.set_row_spacing(2)

        controls = [
            ('Mouse:', 'Select and direct camera'),
            ('Arrow Keys:', 'Move selected piece'),
            ('W:', 'move camera forward'),
            ('S:', 'move camera backward'),
            ('A:', 'move camera left'),
            ('D:', 'move camera right'),
            ('Shift:', 'move camera down'),
            ('Space:', 'move camera up'),
        ]

        for i, (key, action) in enumerate(controls):
            key_label = Gtk.Label()
            key_label.set_text(key)
            key_label.set_alignment(0, 0.5)
            key_label.set_markup(f'<tt>{key}</tt>')

            action_label = Gtk.Label()
            action_label.set_text(action)
            action_label.set_alignment(0, 0.5)

            controls_grid.attach(key_label, 0, i, 1, 1)
            controls_grid.attach(action_label, 1, i, 1, 1)

        help_content.pack_start(controls_grid, False, False, 0)

        # Goal section
        goal_label = Gtk.Label()
        goal_label.set_markup('<b>Goal:</b>')
        goal_label.set_alignment(0, 0.5)
        help_content.pack_start(
            goal_label, False, False, style.DEFAULT_SPACING)

        goal_text = Gtk.Label()
        goal_text.set_text('Arrange all 7 pieces to completely fill the 3×3×3 cube.\n'
                           'No pieces should overlap or extend outside the cube.')
        goal_text.set_alignment(0, 0.5)
        goal_text.set_line_wrap(True)
        goal_text.set_max_width_chars(style.MENU_WIDTH_CHARS)
        help_content.pack_start(goal_text, False, False, 0)

        # Add content to palette
        palette.set_content(help_content)
        help_content.show_all()

        # Set palette on the help button
        self.help_button.set_palette(palette)

    def _setup_content(self):
        # Create main container
        self.main_box = Gtk.VBox()

        # Create an Overlay to layer the HUD on top of the GL view
        overlay = Gtk.Overlay()
        self.main_box.pack_start(overlay, True, True, 0)

        # Create our OpenGL view
        self.gl_view = GLView()
        self.gl_view.set_can_focus(True)
        self.gl_view.grab_focus()
        overlay.add(self.gl_view)

        # Create a Grid to hold the HUD labels
        controls_hud = Gtk.Grid()
        controls_hud.set_column_spacing(10)
        controls_hud.set_row_spacing(5)
        controls_hud.set_halign(Gtk.Align.START)
        controls_hud.set_valign(Gtk.Align.START)
        controls_hud.set_margin_top(10)
        controls_hud.set_margin_start(10)

        controls_hud.set_can_focus(False)
        overlay.add_overlay(controls_hud)
        overlay.set_overlay_pass_through(controls_hud, True)

        hud_title = Gtk.Label()
        hud_title.set_markup("<b><u>Piece Controls</u></b>")
        label_up = Gtk.Label(label="Up:")
        label_down = Gtk.Label(label="Down:")
        label_left = Gtk.Label(label="Left:")
        label_right = Gtk.Label(label="Right:")
        label_fwd = Gtk.Label(label="Forward:")
        label_back = Gtk.Label(label="Backward:")

        controls_hud.attach(hud_title, 0, 0, 2, 1)
        controls_hud.attach(label_up, 0, 1, 2, 1)
        controls_hud.attach(label_down, 0, 2, 2, 1)
        controls_hud.attach(label_left, 0, 3, 2, 1)
        controls_hud.attach(label_right, 0, 4, 2, 1)
        controls_hud.attach(label_fwd, 0, 5, 2, 1)
        controls_hud.attach(label_back, 0, 6, 2, 1)

        self.gl_view.hud_labels = {
            "up": label_up,
            "down": label_down,
            "left": label_left,
            "right": label_right,
            "forward": label_fwd,
            "backward": label_back,
        }

        # Victory box setup
        self.victory_box = Gtk.VBox(spacing=15)
        self.victory_box.set_halign(Gtk.Align.CENTER)
        self.victory_box.set_valign(Gtk.Align.CENTER)

        victory_label = Gtk.Label()
        victory_label.set_markup(
            "<span size='xx-large' weight='bold'>Puzzle Complete!</span>")

        play_again_button = Gtk.Button(label="Play Again")
        play_again_button.connect('clicked', self._on_play_again_clicked)

        self.victory_box.pack_start(victory_label, False, False, 0)
        self.victory_box.pack_start(play_again_button, False, False, 0)

        overlay.add_overlay(self.victory_box)

        # Connect the signal from glview
        self.gl_view.connect('puzzle-completed', self._on_puzzle_completed)

        self.set_canvas(self.main_box)
        self.main_box.show_all()

        self.victory_box.hide()

    def _on_puzzle_completed(self, gl_view):
        """Handler for the 'puzzle-completed' signal."""
        print("Activity received puzzle completion signal!")
        self.victory_box.show()

    def _on_play_again_clicked(self, button):
        """Handler for the 'Play Again' button click."""
        self.victory_box.hide()
        self.gl_view.reset_puzzle()
