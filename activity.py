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
        self.help_button.connect("clicked", self._on_help_clicked)
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

    def _on_help_clicked(self, button):
        """Show the help dialog when help button is clicked."""
        help_message = """About Soma Cube:
The Soma cube is a classic 3D puzzle where you must arrange 7 different pieces to form a perfect 3×3×3 cube. Each piece is made up of 3 or 4 unit cubes joined face-to-face.

Game Controls:
• Mouse: Select pieces and control camera view
• Arrow Keys: Move the selected piece in the grid
• W: Move camera forward
• S: Move camera backward  
• A: Move camera left
• D: Move camera right
• Shift: Move camera down
• Space: Move camera up

Piece Movement:
• Up Arrow: Move piece up one level
• Down Arrow: Move piece down one level
• Left Arrow: Move piece left
• Right Arrow: Move piece right
• Page Up: Move piece forward (into the screen)
• Page Down: Move piece backward (out of the screen)

Goal:
Arrange all 7 pieces to completely fill the 3×3×3 cube. No pieces should overlap or extend outside the cube boundaries. 

Tips:
• Start with corner and edge pieces - they have fewer placement options
• Rotate the camera to see all sides of your progress
• Each piece can be rotated and flipped in multiple orientations
• There are 240 different solutions to discover!

The Soma cube was invented by Danish mathematician Piet Hein in 1933. It's not just a puzzle - it's a fascinating exploration of 3D geometry and spatial reasoning!"""
        
        self._show_dialog("Soma Cube Help", help_message)

    def _show_dialog(self, title, message):
        """Show custom help dialog with Sugar styling"""
        try:
            from sugar3.graphics import style
            parent_window = self.get_toplevel()
            
            dialog = Gtk.Window()
            dialog.set_title(title)
            dialog.set_modal(True)
            dialog.set_decorated(False)
            dialog.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
            dialog.set_border_width(style.LINE_WIDTH)
            dialog.set_transient_for(parent_window)
            
            dialog_width = min(700, max(500, self.get_allocated_width() * 3 // 4))
            dialog_height = min(600, max(400, self.get_allocated_height() * 3 // 4))
            dialog.set_size_request(dialog_width, dialog_height)
            
            main_vbox = Gtk.VBox()
            main_vbox.set_border_width(style.DEFAULT_SPACING)
            dialog.add(main_vbox)
            
            header_box = Gtk.HBox()
            header_box.set_spacing(style.DEFAULT_SPACING)
            
            title_label = Gtk.Label()
            title_label.set_markup(f'<span size="large" weight="bold">🧩 {title}</span>')
            header_box.pack_start(title_label, True, True, 0)
            
            close_button = Gtk.Button()
            close_button.set_relief(Gtk.ReliefStyle.NONE)
            close_button.set_size_request(40, 40)
            
            try:
                from sugar3.graphics.icon import Icon
                close_icon = Icon(icon_name='dialog-cancel', pixel_size=24)
                close_button.add(close_icon)
            except:
                close_label = Gtk.Label()
                close_label.set_markup('<span size="x-large" weight="bold">✕</span>')
                close_button.add(close_label)
            
            close_button.connect('clicked', lambda b: dialog.destroy())
            header_box.pack_end(close_button, False, False, 0)
            
            main_vbox.pack_start(header_box, False, False, 0)
            
            separator = Gtk.HSeparator()
            main_vbox.pack_start(separator, False, False, style.DEFAULT_SPACING)
            
            scrolled = Gtk.ScrolledWindow()
            scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            scrolled.set_hexpand(True)
            scrolled.set_vexpand(True)
            
            content_label = Gtk.Label()
            content_label.set_text(message)
            content_label.set_halign(Gtk.Align.START)
            content_label.set_valign(Gtk.Align.START)
            content_label.set_line_wrap(True)
            content_label.set_max_width_chars(90)
            content_label.set_selectable(True)
            content_label.set_margin_left(15)
            content_label.set_margin_right(15)
            content_label.set_margin_top(15)
            content_label.set_margin_bottom(15)
            
            scrolled.add(content_label)
            main_vbox.pack_start(scrolled, True, True, 0)
            
            try:
                css_provider = Gtk.CssProvider()
                css_data = """
                window {
                    background-color: #ffffff;
                    border: 3px solid #4A90E2;
                    border-radius: 12px;
                }
                label {
                    color: #333333;
                }
                button {
                    border-radius: 20px;
                }
                button:hover {
                    background-color: rgba(74, 144, 226, 0.1);
                }
                scrolledwindow {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                }
                """.encode('utf-8')
                
                css_provider.load_from_data(css_data)
                style_context = dialog.get_style_context()
                style_context.add_provider(css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            except Exception as css_error:
                print(f"CSS styling failed: {css_error}")
            
            dialog.show_all()
            
            dialog.connect('key-press-event', 
                        lambda d, e: d.destroy() if Gdk.keyval_name(e.keyval) == 'Escape' else False)
            
        except Exception as e:
            print(f"Error showing help dialog: {e}")
            self._show_simple_help_fallback()

    def _show_simple_help_fallback(self):
        """Simple fallback help dialog if custom dialog fails"""
        dialog = Gtk.MessageDialog(
            parent=self.get_toplevel(),
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=_("Soma Cube Help"),
        )
        dialog.format_secondary_text(
            _("Arrange 7 pieces to form a 3×3×3 cube. Use mouse to select pieces "
              "and arrow keys to move them. Goal: fill the cube completely!")
        )
        dialog.run()
        dialog.destroy()

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
