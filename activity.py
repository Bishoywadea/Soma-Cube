from sugar3.activity import activity
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import ActivityToolbarButton, StopButton
from gi.repository import Gtk
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
        separator.set_expand(True)
        toolbar_box.toolbar.insert(separator, -1)
        
        # Stop button
        stop_button = StopButton(self)
        toolbar_box.toolbar.insert(stop_button, -1)
        
        self.set_toolbar_box(toolbar_box)
        toolbar_box.show_all()
    
    def _setup_content(self):
        # Create main container
        main_box = Gtk.VBox()
        
        # Create our OpenGL view
        self.gl_view = GLView()
        self.gl_view.set_size_request(600, 400)
        
        # Add some instructions
        instructions = Gtk.Label()
        instructions.set_markup(
            "<span size='large'>Rotate view with mouse</span>")
        instructions.set_halign(Gtk.Align.CENTER)
        instructions.set_margin_bottom(10)
        
        main_box.pack_start(self.gl_view, True, True, 0)
        main_box.pack_start(instructions, False, False, 0)
        
        self.set_canvas(main_box)
        main_box.show_all()