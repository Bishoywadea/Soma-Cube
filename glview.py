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
        self.rotation_x = 20.0
        self.rotation_y = -45.0
        self.zoom = 5.0
        self.last_mouse_pos = None
        
        # Set up events
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                       Gdk.EventMask.BUTTON_RELEASE_MASK |
                       Gdk.EventMask.POINTER_MOTION_MASK |
                       Gdk.EventMask.SCROLL_MASK)
        
        self.connect("button-press-event", self.on_mouse_press)
        self.connect("button-release-event", self.on_mouse_release)
        self.connect("motion-notify-event", self.on_mouse_motion)
        self.connect("scroll-event", self.on_scroll)
        
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
        # Cube vertices - each face needs its own vertices for proper rendering
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
        
        # Indices for drawing triangles
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
        
        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Set vertex attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        # Unbind
        glBindVertexArray(0)
    
    def on_render(self, area, context):
        if not self.shader:
            return False
            
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get dimensions
        width = self.get_allocated_width()
        height = self.get_allocated_height()
        if width == 0 or height == 0:
            return False
            
        glViewport(0, 0, width, height)
        
        # Use shader
        glUseProgram(self.shader)
        
        # Create matrices
        aspect = width / height
        
        # Projection matrix
        projection = self.perspective(45.0, aspect, 0.1, 100.0)
        
        # View matrix - camera looking at origin
        eye = [0, 0, self.zoom]
        center = [0, 0, 0]
        up = [0, 1, 0]
        view = self.look_at(eye, center, up)
        
        # Model matrix - apply rotations
        model = np.eye(4, dtype=np.float32)
        model = self.rotate_x(model, self.rotation_x)
        model = self.rotate_y(model, self.rotation_y)
        
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
    
    def look_at(self, eye, center, up):
        """Create look-at view matrix"""
        eye = np.array(eye, dtype=np.float32)
        center = np.array(center, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        result = np.eye(4, dtype=np.float32)
        result[0, :3] = s
        result[1, :3] = u
        result[2, :3] = -f
        result[:3, 3] = -np.dot(result[:3, :3], eye)
        
        return result
    
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
            return True
    
    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = None
            return True
    
    def on_mouse_motion(self, widget, event):
        if self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
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

class CubeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="3D Cube Viewer")
        self.set_default_size(800, 600)
        
        self.gl_area = GLView()
        self.add(self.gl_area)
        
        self.connect("destroy", Gtk.main_quit)

if __name__ == "__main__":
    window = CubeWindow()
    window.show_all()
    Gtk.main()