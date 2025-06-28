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

from sugar3.activity import activity
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import ActivityToolbarButton, StopButton
from sugar3.graphics.toolbutton import ToolButton
from gi.repository import Gtk, GdkPixbuf, Gdk
import os
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
        reset_button = ToolButton("emblem-busy")
        reset_button.set_tooltip("Reset")
        reset_button.connect("clicked", self._on_play_again_clicked)
        toolbar_box.toolbar.insert(reset_button, -1)

        # Help button
        help_button = ToolButton("toolbar-help")
        help_button.set_tooltip("Help")
        help_button.connect("clicked", self._show_help)
        toolbar_box.toolbar.insert(help_button, -1)

        # Separator
        separator = Gtk.SeparatorToolItem()
        separator.props.draw = False
        separator.set_expand(True)
        toolbar_box.toolbar.insert(separator, -1)

        # Stop button
        stop_button = StopButton(self)
        toolbar_box.toolbar.insert(stop_button, -1)

        self.set_toolbar_box(toolbar_box)
        toolbar_box.show_all()

    def _setup_content(self):
        # Create main container
        self.main_box = Gtk.VBox()

        # Create an Overlay to layer the HUD on top of the GL view
        overlay = Gtk.Overlay()
        self.main_box.pack_start(overlay, True, True, 0)

        # Create our OpenGL view
        self.gl_view = GLView()
        overlay.add(self.gl_view)

        # Create a Grid to hold the HUD labels
        controls_hud = Gtk.Grid()
        controls_hud.set_column_spacing(10)
        controls_hud.set_row_spacing(5)
        controls_hud.set_halign(Gtk.Align.START)
        controls_hud.set_valign(Gtk.Align.START)
        controls_hud.set_margin_top(10)
        controls_hud.set_margin_start(10)

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

        # Add help overlay setup
        self.help_overlay = Gtk.EventBox()
        self.help_overlay.set_halign(Gtk.Align.FILL)
        self.help_overlay.set_valign(Gtk.Align.FILL)
        
        # Make the EventBox clickable and connect click event
        self.help_overlay.connect('button-press-event', self._on_help_clicked)
        
        # Create a box to center the help image
        help_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        help_container.set_halign(Gtk.Align.CENTER)
        help_container.set_valign(Gtk.Align.CENTER)
        
        try:
            help_image = Gtk.Image.new_from_file('help.gif')
            help_container.pack_start(help_image, False, False, 0)
        except Exception as e:
            print(f"Failed to load help image: {e}")
            # Fallback to a label
            label = Gtk.Label("Help image not found")
            help_container.pack_start(label, False, False, 0)
        
        self.help_overlay.add(help_container)
        
        # Add semi-transparent background (optional)
        self.help_overlay.override_background_color(
            Gtk.StateFlags.NORMAL, 
            Gdk.RGBA(0, 0, 0, 0.8)  # Black with 80% opacity
        )
        
        overlay.add_overlay(self.help_overlay)
    
        
        self.victory_box = Gtk.VBox(spacing=15)
        self.victory_box.set_halign(Gtk.Align.CENTER)
        self.victory_box.set_valign(Gtk.Align.CENTER)

        victory_label = Gtk.Label()
        victory_label.set_markup("<span size='xx-large' weight='bold'>Puzzle Complete!</span>")

        play_again_button = Gtk.Button(label="Play Again")
        play_again_button.connect('clicked', self._on_play_again_clicked)

        self.victory_box.pack_start(victory_label, False, False, 0)
        self.victory_box.pack_start(play_again_button, False, False, 0)

        overlay.add_overlay(self.victory_box)

        # --- CONNECT THE SIGNAL FROM GLVIEW ---
        self.gl_view.connect('puzzle-completed', self._on_puzzle_completed)

        self.main_box.pack_start(self.gl_view, True, True, 0)

        self.set_canvas(self.main_box)
        self.main_box.show_all()
        self.victory_box.hide()
        self.help_overlay.hide()

    def _show_help(self, button):
        """Show help dialog with image"""
        self.help_overlay.show_all()
    
    def _on_help_clicked(self, widget, event):
        """Hide help overlay when clicked anywhere"""
        self.help_overlay.hide()
        return True

    def _on_puzzle_completed(self, gl_view):
        """Handler for the 'puzzle-completed' signal."""
        print("Activity received puzzle completion signal!")
        self.victory_box.show()

    def _on_play_again_clicked(self, button):
        """Handler for the 'Play Again' button click."""
        self.victory_box.hide()
        self.gl_view.reset_puzzle()