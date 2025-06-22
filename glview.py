import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math

# Simple vertex shader
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Simple fragment shader
fragment_shader = """
#version 330 core
out vec4 frag_color;
uniform vec3 color;

void main() {
    frag_color = vec4(color, 1.0);
}
"""

class GLView(Gtk.GLArea):
    def __init__(self):
        super().__init__()
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)
        self.connect("realize", self.on_realize)
        self.connect("render", self.on_render)
        
        # Camera control
        self.camera_rotation = [20.0, -45.0]  # Camera rotation (pitch, yaw)
        self.zoom = 5.0
        self.camera_position = [0.0, 0.0, 0.0]
        
        # Object control
        self.object_position = [0.0, 0.0, 0.0]
        self.object_rotation = [0.0, 0.0]
        
        # Mouse control state
        self.last_mouse_pos = None
        self.dragging_object = False
        self.selected_object = None
        
        # Movement
        self.keys_pressed = set()
        self.movement_speed = 0.1
        self.render_timer = None
        
        # Set up events
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                       Gdk.EventMask.BUTTON_RELEASE_MASK |
                       Gdk.EventMask.POINTER_MOTION_MASK |
                       Gdk.EventMask.SCROLL_MASK |
                       Gdk.EventMask.KEY_PRESS_MASK |
                       Gdk.EventMask.KEY_RELEASE_MASK)
        
        self.connect("button-press-event", self.on_mouse_press)
        self.connect("button-release-event", self.on_mouse_release)
        self.connect("motion-notify-event", self.on_mouse_motion)
        self.connect("scroll-event", self.on_scroll)
        self.connect("key-press-event", self.on_key_press)
        self.connect("key-release-event", self.on_key_release)
        
        self.set_can_focus(True)
        
        # OpenGL objects
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.shader = None

    def on_realize(self, area):
        self.make_current()
        
        # Initialize OpenGL
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        # Create shader
        self.shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Create cube
        self.setup_cube()
        
    def setup_cube(self):
        # Cube vertices
        vertices = np.array([
            # Front face (z = 0.5)
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            
            # Back face (z = -0.5)
            -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,
            
            # Top face (y = 0.5)
            -0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,
             0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,
            
            # Bottom face (y = -0.5)
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5,
            
            # Right face (x = 0.5)
             0.5, -0.5,  0.5,
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5,  0.5,  0.5,
            
            # Left face (x = -0.5)
            -0.5, -0.5,  0.5,
            -0.5, -0.5, -0.5,
            -0.5,  0.5, -0.5,
            -0.5,  0.5,  0.5,
        ], dtype=np.float32)
        
        # Indices
        indices = np.array([
            # Front face
            0, 1, 2,    2, 3, 0,
            # Back face
            4, 5, 6,    6, 7, 4,
            # Top face
            8, 9, 10,   10, 11, 8,
            # Bottom face
            12, 13, 14, 14, 15, 12,
            # Right face
            16, 17, 18, 18, 19, 16,
            # Left face
            20, 21, 22, 22, 23, 20
        ], dtype=np.uint32)
        
        # Create VAO, VBO, EBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        glBindVertexArray(0)
    
    def get_camera_vectors(self):
        """Calculate forward, right, and up vectors based on camera rotation"""
        # Convert angles to radians
        pitch = math.radians(self.camera_rotation[0])
        yaw = math.radians(self.camera_rotation[1])
        
        # Calculate forward vector
        forward = np.array([
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw)
        ])
        
        # Calculate right vector (cross product of forward and world up)
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        # Calculate camera up vector
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        return forward, right, up
    
    def check_object_hit(self, x, y):
        """Check if mouse click hits an object"""
        self.make_current()
        
        # Flip Y coordinate
        viewport_height = self.get_allocated_height()
        gl_y = viewport_height - y
        
        # Read depth value at cursor position
        depth = glReadPixels(int(x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
        
        return depth < 1.0
    
    def on_render(self, area, context):
        if not self.shader:
            return False
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        width = self.get_allocated_width()
        height = self.get_allocated_height()
        if width == 0 or height == 0:
            return False
            
        glViewport(0, 0, width, height)
        glUseProgram(self.shader)
        
        # Create matrices
        aspect = width / height
        projection = self.perspective(45.0, aspect, 0.1, 100.0)
        
        # View matrix with camera rotation
        view = self.create_view_matrix()
        
        # Model matrix with object transformations
        model = np.eye(4, dtype=np.float32)
        model = self.translate(model, self.object_position[0], 
                             self.object_position[1], 
                             self.object_position[2])
        model = self.rotate_x(model, self.object_rotation[0])
        model = self.rotate_y(model, self.object_rotation[1])
        
        # Set uniforms
        model_loc = glGetUniformLocation(self.shader, "model")
        view_loc = glGetUniformLocation(self.shader, "view")
        proj_loc = glGetUniformLocation(self.shader, "projection")
        color_loc = glGetUniformLocation(self.shader, "color")
        
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T.flatten())
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.T.flatten())
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.T.flatten())
        
        # Draw solid cube
        glUniform3f(color_loc, 0.7, 0.7, 0.7)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        
        # Draw wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glUniform3f(color_loc, 0.0, 0.0, 0.0)
        glLineWidth(2.0)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        return True
    
    def create_view_matrix(self):
        """Create view matrix with camera rotations"""
        view = np.eye(4, dtype=np.float32)
        
        # Apply camera transformations
        view = self.translate(view, -self.camera_position[0], 
                            -self.camera_position[1], 
                            -(self.camera_position[2] + self.zoom))
        
        view = self.rotate_x(view, self.camera_rotation[0])
        view = self.rotate_y(view, self.camera_rotation[1])
        
        return view
    
    def perspective(self, fovy, aspect, near, far):
        """Create perspective projection matrix"""
        f = 1.0 / math.tan(math.radians(fovy) / 2.0)
        nf = 1.0 / (near - far)
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) * nf, 2 * far * near * nf],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def translate(self, m, x, y, z):
        """Apply translation to matrix"""
        trans = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return np.dot(trans, m)
    
    def rotate_x(self, m, angle):
        """Rotate around X axis"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        rot = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return np.dot(rot, m)
    
    def rotate_y(self, m, angle):
        """Rotate around Y axis"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        rot = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return np.dot(rot, m)
    
    def on_mouse_press(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = (event.x, event.y)
            self.grab_focus()
            
            # Check if we clicked on an object
            self.dragging_object = self.check_object_hit(event.x, event.y)
            
            return True
    
    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = None
            self.dragging_object = False
            return True
    
    def on_mouse_motion(self, widget, event):
        if self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            
            if self.dragging_object:
                # Move object
                self.object_rotation[1] += dx * 0.5
                self.object_rotation[0] += dy * 0.5
            else:
                # Rotate camera
                self.camera_rotation[1] += dx * 0.5
                self.camera_rotation[0] += dy * 0.5
                # Clamp camera pitch
                self.camera_rotation[0] = max(-89, min(89, self.camera_rotation[0]))
            
            self.queue_render()
            self.last_mouse_pos = (event.x, event.y)
            return True
    
    def on_scroll(self, widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom = max(2.0, self.zoom - 0.5)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom = min(20.0, self.zoom + 0.5)
        
        self.queue_render()
        return True
    
    def on_key_press(self, widget, event):
        """Handle key press events"""
        self.keys_pressed.add(event.keyval)
        
        # Start continuous rendering
        if self.render_timer is None:
            self.render_timer = GLib.timeout_add(16, self.update_movement)
        
        # Handle immediate actions
        if event.keyval == Gdk.KEY_r or event.keyval == Gdk.KEY_R:
            # Reset everything
            self.camera_rotation = [20.0, -45.0]
            self.camera_position = [0.0, 0.0, 0.0]
            self.object_position = [0.0, 0.0, 0.0]
            self.object_rotation = [0.0, 0.0]
            self.zoom = 5.0
            self.queue_render()
            
        return True
    
    def on_key_release(self, widget, event):
        """Handle key release events"""
        self.keys_pressed.discard(event.keyval)
        
        # Stop continuous rendering if no keys pressed
        if not self.keys_pressed and self.render_timer:
            GLib.source_remove(self.render_timer)
            self.render_timer = None
            
        return True
    
    def update_movement(self):
        """Update movement based on pressed keys - relative to camera orientation"""
        # Get camera direction vectors
        forward, right, up = self.get_camera_vectors()
        
        # Calculate movement based on camera orientation
        movement = np.array([0.0, 0.0, 0.0])
        
        # Forward/backward (W/S) - move along camera's forward direction
        if Gdk.KEY_w in self.keys_pressed or Gdk.KEY_W in self.keys_pressed:
            movement += forward * self.movement_speed
        if Gdk.KEY_s in self.keys_pressed or Gdk.KEY_S in self.keys_pressed:
            movement -= forward * self.movement_speed
            
        # Left/right (A/D) - move along camera's right direction
        if Gdk.KEY_a in self.keys_pressed or Gdk.KEY_A in self.keys_pressed:
            movement -= right * self.movement_speed
        if Gdk.KEY_d in self.keys_pressed or Gdk.KEY_D in self.keys_pressed:
            movement += right * self.movement_speed
            
        # Up/down (Q/E or Space/Shift) - move along world up
        if Gdk.KEY_q in self.keys_pressed or Gdk.KEY_Q in self.keys_pressed:
            movement[1] -= self.movement_speed
        if Gdk.KEY_e in self.keys_pressed or Gdk.KEY_E in self.keys_pressed:
            movement[1] += self.movement_speed
            
        if Gdk.KEY_space in self.keys_pressed:
            movement[1] += self.movement_speed
        if Gdk.KEY_Shift_L in self.keys_pressed or Gdk.KEY_Shift_R in self.keys_pressed:
            movement[1] -= self.movement_speed
        
        # Apply movement to camera position
        self.camera_position[0] += movement[0]
        self.camera_position[1] += movement[1]
        self.camera_position[2] += movement[2]
            
        # Arrow keys to move object (unchanged)
        if Gdk.KEY_Left in self.keys_pressed:
            self.object_position[0] -= self.movement_speed
        if Gdk.KEY_Right in self.keys_pressed:
            self.object_position[0] += self.movement_speed
        if Gdk.KEY_Up in self.keys_pressed:
            self.object_position[1] += self.movement_speed
        if Gdk.KEY_Down in self.keys_pressed:
            self.object_position[1] -= self.movement_speed
            
        self.queue_render()
        return True  # Continue timer

class CubeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="3D Cube Viewer - View-Relative Movement")
        self.set_default_size(800, 600)
        
        # Create a box layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(box)
        
        # Add instructions
        label = Gtk.Label()
        label.set_markup(
            "<b>Mouse Controls:</b>\n" +
            "• Click + drag on <b>object</b>: Rotate object\n" +
            "• Click + drag on <b>empty space</b>: Rotate camera view\n" +
            "• Scroll: Zoom in/out\n\n" +
            "<b>Camera Movement (relative to view):</b>\n" +
            "• W/S: Move forward/backward\n" +
            "• A/D: Strafe left/right\n" +
            "• Q/E or Space/Shift: Move up/down\n" +
            "• Arrow keys: Move object\n" +
            "• R: Reset all\n\n" +
            "<i>Movement is now relative to where you're looking!</i>"
        )
        label.set_margin_top(5)
        label.set_margin_bottom(5)
        label.set_margin_left(10)
        label.set_margin_right(10)
        box.pack_start(label, False, False, 0)
        
        # Add separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        box.pack_start(separator, False, False, 0)
        
        # Add GL area
        self.gl_area = GLView()
        box.pack_start(self.gl_area, True, True, 0)
        
        # Focus the GL area for keyboard input
        self.gl_area.grab_focus()
        
        self.connect("destroy", Gtk.main_quit)

if __name__ == "__main__":
    window = CubeWindow()
    window.show_all()
    Gtk.main()