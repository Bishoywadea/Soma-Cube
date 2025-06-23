from sugar3.activity import activity
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import ActivityToolbarButton, StopButton
from sugar3.graphics.toolbutton import ToolButton
from gi.repository import Gtk, Gdk, Pango
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
        
        # Help button
        help_button = ToolButton('toolbar-help')
        help_button.set_tooltip('Help')
        help_button.connect('clicked', self._show_help)
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
        
        # Create our OpenGL view
        self.gl_view = GLView()
        self.gl_view.set_size_request(600, 400)
        
        # Add some instructions
        self.instructions = Gtk.Label()
        self.instructions.set_markup(
            "<span size='large'>Rotate view with mouse</span>")
        self.instructions.set_halign(Gtk.Align.CENTER)
        self.instructions.set_margin_bottom(10)
        
        self.main_box.pack_start(self.gl_view, True, True, 0)
        self.main_box.pack_start(self.instructions, False, False, 0)
        
        self.set_canvas(self.main_box)
        self.main_box.show_all()
    
    def _show_help(self, button):
        """Show help dialog"""
        dialog = Gtk.Dialog(
            title="Soma Cube Help",
            parent=self,
            flags=Gtk.DialogFlags.MODAL,
            buttons=("_Close", Gtk.ResponseType.CLOSE)
        )
        
        # Set dialog size
        dialog.set_default_size(400, 300)
        
        # Create text view for help content
        textview = Gtk.TextView()
        textview.set_editable(False)
        textview.set_cursor_visible(False)
        textview.set_wrap_mode(Gtk.WrapMode.WORD)
        
        # Set font
        font_desc = Pango.FontDescription("Sans 12")
        textview.modify_font(font_desc)
        
        # Add help text
        help_text = """Soma Cube Help:

1. Left-click and drag to rotate the cube
2. Right-click and drag to pan the view
3. Scroll to zoom in/out
4. Press R to reset the view"""
        
        buffer = textview.get_buffer()
        buffer.set_text(help_text)
        
        # Add to dialog
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(
            Gtk.PolicyType.AUTOMATIC, 
            Gtk.PolicyType.AUTOMATIC)
        scrolled.add(textview)
        dialog.get_content_area().pack_start(scrolled, True, True, 0)
        
        # Show dialog
        dialog.show_all()
        dialog.run()
        dialog.destroy()