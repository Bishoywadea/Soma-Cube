import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math

vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
"""

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
        self.camera_distance = 3.0
        self.camera_rotation = [20.0, 45.0]
        self.last_mouse_pos = None
        
        # Set up mouse controls
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                       Gdk.EventMask.BUTTON_RELEASE_MASK |
                       Gdk.EventMask.POINTER_MOTION_MASK |
                       Gdk.EventMask.SCROLL_MASK)
        self.connect("button-press-event", self.on_mouse_press)
        self.connect("button-release-event", self.on_mouse_release)
        self.connect("motion-notify-event", self.on_mouse_motion)
        self.connect("scroll-event", self.on_scroll)
        
        # OpenGL objects
        self.vao = None
        self.vbo = None
        self.shader = None

    def on_realize(self, area):
        print("Realizing OpenGL context...")
        self.make_current()
        
        # Check OpenGL version
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
        
        # Initialize OpenGL
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        try:
            # Create shader program
            self.shader = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            print("Shaders compiled successfully")
        except Exception as e:
            print(f"Shader compilation error: {e}")
            return
        
        # Simple cube vertices - just 8 corners
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  # 0
             0.5, -0.5,  0.5,  # 1
             0.5,  0.5,  0.5,  # 2
            -0.5,  0.5,  0.5,  # 3
            # Back face
            -0.5, -0.5, -0.5,  # 4
             0.5, -0.5, -0.5,  # 5
             0.5,  0.5, -0.5,  # 6
            -0.5,  0.5, -0.5   # 7
        ], dtype=np.float32)
        
        # Cube indices for triangles
        indices = np.array([
            # Front face
            0, 1, 2,  2, 3, 0,
            # Back face
            4, 5, 6,  6, 7, 4,
            # Left face
            4, 0, 3,  3, 7, 4,
            # Right face
            1, 5, 6,  6, 2, 1,
            # Bottom face
            4, 5, 1,  1, 0, 4,
            # Top face
            3, 2, 6,  6, 7, 3
        ], dtype=np.uint32)
        
        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Set vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        print("OpenGL setup complete")
    
    def on_render(self, area, context):
        if not self.shader:
            return False
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader
        glUseProgram(self.shader)
        
        # Get viewport dimensions
        width = self.get_allocated_width()
        height = self.get_allocated_height()
        
        if width <= 0 or height <= 0:
            return False
            
        # Create matrices
        aspect = width / height
        
        # Projection matrix
        fov = math.radians(45.0)
        f = 1.0 / math.tan(fov / 2.0)
        proj = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1.002, -0.2002],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        # View matrix (camera)
        view = np.eye(4, dtype=np.float32)
        
        # Apply transformations in reverse order
        # 1. Move back
        view[2, 3] = -self.camera_distance
        
        # 2. Rotate around X (pitch)
        rx = math.radians(self.camera_rotation[0])
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, sin_x, cos_x, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 3. Rotate around Y (yaw)
        ry = math.radians(self.camera_rotation[1])
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        rot_y = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combine transformations
        view = np.dot(rot_x, view)
        view = np.dot(rot_y, view)
        
        # Model matrix (identity - no model transformation)
        model = np.eye(4, dtype=np.float32)
        
        # Combine all matrices: MVP = Projection * View * Model
        mvp = np.dot(proj, np.dot(view, model))
        
        # Set uniform
        mvp_loc = glGetUniformLocation(self.shader, "mvp")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp.flatten())
        
        color_loc = glGetUniformLocation(self.shader, "color")
        glUniform3f(color_loc, 1.0, 0.5, 0.2)
        
        # Draw cube
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        return True
    
    def on_mouse_press(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = (event.x, event.y)
            return True
    
    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = None
            return True
    
    def on_mouse_motion(self, widget, event):
        if self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            
            self.camera_rotation[1] += dx * 0.5
            self.camera_rotation[0] += dy * 0.5
            
            # Clamp pitch
            self.camera_rotation[0] = max(-89, min(89, self.camera_rotation[0]))
            
            self.queue_render()
            self.last_mouse_pos = (event.x, event.y)
            return True
    
    def on_scroll(self, widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.camera_distance = max(1.0, self.camera_distance - 0.2)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.camera_distance = min(10.0, self.camera_distance + 0.2)
        
        self.queue_render()
        return True

class CubeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="3D Cube Viewer - Fixed")
        self.set_default_size(800, 600)
        
        # Create a box layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(box)
        
        # Add instructions
        label = Gtk.Label()
        label.set_markup("<b>Controls:</b> Left click + drag to rotate, scroll to zoom")
        label.set_margin_top(5)
        label.set_margin_bottom(5)
        box.pack_start(label, False, False, 0)
        
        # Add GL area
        self.gl_area = GLView()
        box.pack_start(self.gl_area, True, True, 0)
        
        self.connect("destroy", Gtk.main_quit)

if __name__ == "__main__":
    print("Starting cube viewer...")
    window = CubeWindow()
    window.show_all()
    print("Window shown, starting main loop...")
    Gtk.main()