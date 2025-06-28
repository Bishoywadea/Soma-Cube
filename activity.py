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

        self.animation_timeout = None
        self.animation_iter = None

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

        # Victory box setup
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

        # Help overlay setup
        self.help_overlay = Gtk.EventBox()
        self.help_overlay.set_halign(Gtk.Align.FILL)
        self.help_overlay.set_valign(Gtk.Align.FILL)
        
        # Make the EventBox clickable and connect click event
        self.help_overlay.connect('button-press-event', self._on_help_clicked)
        
        # Create a container for help content
        help_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        help_container.set_halign(Gtk.Align.CENTER)
        help_container.set_valign(Gtk.Align.CENTER)
        
        try:
            # Get screen dimensions
            screen = Gdk.Screen.get_default()
            screen_width = screen.get_width()
            screen_height = screen.get_height()
            
            # Leave some margin (90% of screen size)
            max_width = int(screen_width * 0.9)
            max_height = int(screen_height * 0.85)
            
            # Load the animated GIF
            animation = GdkPixbuf.PixbufAnimation.new_from_file('help.gif')
            
            # Get original dimensions
            orig_width = animation.get_width()
            orig_height = animation.get_height()
            
            # Calculate scaling factor to fit screen while maintaining aspect ratio
            scale_x = max_width / orig_width
            scale_y = max_height / orig_height
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            print(f"Original GIF size: {orig_width}x{orig_height}")
            print(f"Screen size: {screen_width}x{screen_height}")
            print(f"Scaled GIF size: {new_width}x{new_height}")
            
            # Create image widget
            self.help_image = Gtk.Image()
            
            # Scale the static preview first
            static_pixbuf = animation.get_static_image()
            if static_pixbuf:
                scaled_pixbuf = static_pixbuf.scale_simple(
                    new_width, new_height, 
                    GdkPixbuf.InterpType.BILINEAR
                )
                self.help_image.set_from_pixbuf(scaled_pixbuf)
            
            # Store the animation parameters for scaled playback
            self.animation = animation
            self.scaled_width = new_width
            self.scaled_height = new_height
            
            help_container.pack_start(self.help_image, False, False, 0)
            
        except Exception as e:
            print(f"Failed to load help image: {e}")
            import traceback
            traceback.print_exc()
            # Fallback label
            label = Gtk.Label("Help image not found")
            label.override_color(Gtk.StateFlags.NORMAL, Gdk.RGBA(1, 1, 1, 1))
            help_container.pack_start(label, False, False, 0)
        
        # Add instruction label
        instruction_label = Gtk.Label()
        instruction_label.set_markup("<span foreground='white' size='large'>Click anywhere to close</span>")
        help_container.pack_start(instruction_label, False, False, 0)
        
        self.help_overlay.add(help_container)
        
        # Add semi-transparent background
        self.help_overlay.override_background_color(
            Gtk.StateFlags.NORMAL, 
            Gdk.RGBA(0, 0, 0, 0.8)
        )
        
        overlay.add_overlay(self.help_overlay)
        
        # Connect the signal from glview
        self.gl_view.connect('puzzle-completed', self._on_puzzle_completed)

        self.set_canvas(self.main_box)
        self.main_box.show_all()
        
        # Initially hide these overlays
        self.victory_box.hide()
        self.help_overlay.hide()

    def _start_scaled_animation(self):
        """Start the scaled GIF animation"""
        if hasattr(self, 'animation') and self.animation_timeout is None:
            from gi.repository import GLib
            
            self.animation_iter = self.animation.get_iter(None)
            self._update_scaled_animation()

    def _stop_animation(self):
        """Stop the GIF animation"""
        if self.animation_timeout is not None:
            from gi.repository import GLib
            GLib.source_remove(self.animation_timeout)
            self.animation_timeout = None

    def _update_scaled_animation(self):
        """Update animation frame with scaling"""
        if self.animation_iter:
            # Get current frame
            pixbuf = self.animation_iter.get_pixbuf()
            
            # Scale the frame
            scaled_pixbuf = pixbuf.scale_simple(
                self.scaled_width, 
                self.scaled_height,
                GdkPixbuf.InterpType.BILINEAR
            )
            
            # Update the image
            self.help_image.set_from_pixbuf(scaled_pixbuf)
            
            # Advance to next frame
            self.animation_iter.advance(None)
            
            # Schedule next frame
            from gi.repository import GLib
            delay = self.animation_iter.get_delay_time()
            if delay > 0:
                self.animation_timeout = GLib.timeout_add(
                    delay,
                    self._update_scaled_animation
                )
            else:
                # Default delay if not specified
                self.animation_timeout = GLib.timeout_add(
                    100,
                    self._update_scaled_animation
                )
        
        return False

    def _on_help_clicked(self, widget, event):
        """Hide help overlay when clicked anywhere"""
        if hasattr(self, '_stop_animation'):
            self._stop_animation()
        self.help_overlay.hide()
        return True
    
    def _show_help(self, button):
        """Show help overlay"""
        self.help_overlay.show_all()
        if hasattr(self, '_start_scaled_animation'):
            self._start_scaled_animation()

    def _on_puzzle_completed(self, gl_view):
        """Handler for the 'puzzle-completed' signal."""
        print("Activity received puzzle completion signal!")
        self.victory_box.show()

    def _on_play_again_clicked(self, button):
        """Handler for the 'Play Again' button click."""
        self.victory_box.hide()
        self.gl_view.reset_puzzle()