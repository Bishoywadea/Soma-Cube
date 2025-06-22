import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from sugar3.activity import activity
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import ActivityToolbarButton
from sugar3.activity.widgets import StopButton
from sugar3.graphics.toolbutton import ToolButton

class SomaCube(activity.Activity):
    def __init__(self, handle):
        activity.Activity.__init__(self, handle)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create a simple label
        label = Gtk.Label()
        label.set_markup("<span size='x-large' weight='bold'>Soma Cube</span>")
        label.set_halign(Gtk.Align.CENTER)
        label.set_valign(Gtk.Align.CENTER)
        
        # Create a main box to hold everything
        main_box = Gtk.VBox()
        main_box.pack_start(label, True, True, 0)
        
        # Add to the activity canvas
        self.set_canvas(main_box)
        main_box.show_all()
    
    def _create_toolbar(self):
        toolbar_box = ToolbarBox()
        
        # Activity button
        activity_button = ActivityToolbarButton(self)
        toolbar_box.toolbar.insert(activity_button, -1)
        activity_button.show()
        
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
