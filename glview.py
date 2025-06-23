import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import random

# Vertex shader with support for per-vertex colors
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    vertexColor = color;
}
"""

# Fragment shader
fragment_shader = """
#version 330 core
in vec3 vertexColor;
out vec4 frag_color;

uniform vec3 objectColor;
uniform float useVertexColor;
uniform float alpha = 1.0;  // Add alpha uniform

void main() {
    if (useVertexColor > 0.5) {
        frag_color = vec4(vertexColor, 1.0);
    } else {
        frag_color = vec4(objectColor, alpha);  // Use alpha value
    }
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
        self.camera_rotation = [0.0, 0.0]
        self.zoom = 10.0
        self.camera_position = [0.0, 1.7, 0.0] 
        
        # Mouse control
        self.last_mouse_pos = None
        self.dragging_object = False
        
        # Movement
        self.keys_pressed = set()
        self.movement_speed = 0.2
        self.render_timer = None
        
        # Objects in the scene
        self.objects = []
        self.selected_object = None
        
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
        self.cube_vao = None
        self.grid_vao = None
        self.shader = None

        self.selected_piece = None
        self.drag_offset = None

    def on_realize(self, area):
        self.make_current()
        
        # Initialize OpenGL
        glClearColor(0.15, 0.15, 0.15, 1.0)  # Dark background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Create shader
        self.shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Create geometry
        self.setup_cube()
        self.setup_grid()
        self.create_scene_objects()
        
    def setup_cube(self):
        """Create a unit cube mesh"""
        vertices = []
        colors = []
        
        # Define cube faces with colors
        faces = [
            # Front (z=0.5) - slightly different shades for each face
            ([-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]),
            # Back (z=-0.5)
            ([0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]),
            # Top (y=0.5)
            ([-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]),
            # Bottom (y=-0.5)
            ([-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]),
            # Right (x=0.5)
            ([0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]),
            # Left (x=-0.5)
            ([-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5])
        ]
        
        indices = []
        vertex_count = 0
        
        for face in faces:
            for vertex in face:
                vertices.extend(vertex)
                colors.extend([0.8, 0.8, 0.8])  # Default gray color
            
            # Two triangles per face
            base = vertex_count
            indices.extend([base, base+1, base+2, base, base+2, base+3])
            vertex_count += 4
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create VAO
        self.cube_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cube_vao)
        
        # Vertex buffer
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + colors.nbytes, None, GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, colors.nbytes, colors)
        
        # Index buffer
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        self.cube_indices = len(indices)
    
    def setup_grid(self):
        """Create a grid floor"""
        vertices = []
        colors = []
        
        grid_size = 20
        grid_step = 1.0
        y_level = 0.0  # Floor level
        
        # Grid lines
        for i in range(-grid_size, grid_size + 1):
            # Lines parallel to X axis
            vertices.extend([i * grid_step, y_level, -grid_size * grid_step])
            vertices.extend([i * grid_step, y_level, grid_size * grid_step])
            
            # Lines parallel to Z axis
            vertices.extend([-grid_size * grid_step, y_level, i * grid_step])
            vertices.extend([grid_size * grid_step, y_level, i * grid_step])
            
            # Color for grid lines
            if i == 0:
                # Axis lines are brighter
                colors.extend([0.5, 0.5, 0.5] * 4)
            else:
                # Regular grid lines
                colors.extend([0.3, 0.3, 0.3] * 4)
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        
        # Create VAO for grid
        self.grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.grid_vao)
        
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + colors.nbytes, None, GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, colors.nbytes, colors)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        self.grid_vertices = len(vertices) // 3
    
    def create_scene_objects(self):
        """Create the 7 Soma cube pieces in the scene"""
        self.objects = []
        cube_size = 0.6
        grid_size = 3
        grid_spacing = cube_size
        shadow_color = [0.05, 0.05, 0.05, 0.7]
        base_y = 1.2

        for x in range(grid_size):
            for z in range(grid_size):
                for y in range(grid_size):
                    self.objects.append({
                        'pos': [
                            (x - 1) * grid_spacing, 
                            base_y + (y - 1) * grid_spacing,
                            (z - 1) * grid_spacing
                        ],
                        'color': shadow_color,
                        'scale': [cube_size * 0.98] * 3, 
                        'is_shadow': True
                    })

        # Each piece: list of relative positions (x, y, z)
        pieces = [
            # Piece 1: V (3 cubes, bent)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],

            # Piece 2: L (4 cubes, 3 in a line + 1 bend)
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]],

            # Piece 3: T (4 cubes, T shape)
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]],

            # Piece 4: Z (4 cubes, zig-zag)
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],

            # Piece 5: A (3D stair)
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],

            # Piece 6: B (corner: L in 3D)
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]],

            # Piece 7: P (chair shape)
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
        ]


        # Place all pieces on the left side in a vertical queue
        queue_z = -5.0  # Z position for the queue
        horizontal_spacing = 5.0
        
        for i, piece in enumerate(pieces):
            piece_color = self.get_random_color()
            piece_x_offset = i * horizontal_spacing - (len(pieces) * horizontal_spacing)/2
            
            for cube_pos in piece:
                self.objects.append({
                    'pos': [
                        piece_x_offset + cube_pos[0] * cube_size,
                        cube_size/2 + cube_pos[1] * cube_size,
                        queue_z + cube_pos[2] * cube_size
                    ],
                    'color': piece_color,
                    'scale': [cube_size] * 3,
                    'piece_id': i  # Identify which piece this cube belongs to
                })

    def get_random_color(self):
        """Generate a random RGB color with good visibility"""
        return [
            random.uniform(0.3, 0.9),  # Red
            random.uniform(0.3, 0.9),  # Green
            random.uniform(0.3, 0.9)   # Blue
        ]

    def get_camera_vectors(self):
        """Calculate forward, right, and up vectors based on camera rotation"""
        pitch = math.radians(self.camera_rotation[0])
        yaw = math.radians(self.camera_rotation[1])
        
        forward = np.array([
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
            -math.cos(pitch) * math.cos(yaw)
        ])
        
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        return forward, right, up
    
    def check_object_hit(self, x, y):
        """Check if mouse click hits an object"""
        self.make_current()
        
        viewport_height = self.get_allocated_height()
        gl_y = viewport_height - y
        
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
        view = self.create_view_matrix()
        
        # Set uniforms that don't change per object
        view_loc = glGetUniformLocation(self.shader, "view")
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.T.flatten())
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.T.flatten())
        
        # Draw grid floor
        self.draw_grid()
        
        # Enable blending for shadows (should be done here, before drawing shadows)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # First draw all shadow cubes
        for obj in self.objects:
            if obj.get('is_shadow', False):
                self.draw_shadow_cube(obj['pos'], obj['scale'], obj['color'])
        
        # Disable blending for regular objects
        glDisable(GL_BLEND)
        
        # Then draw all regular cubes
        for obj in self.objects:
            if not obj.get('is_shadow', False):
                self.draw_cube(obj['pos'], obj['scale'], obj['color'])
        
        glUseProgram(0)
        return True
    
    def draw_grid(self):
        """Draw the grid floor"""
        model = np.eye(4, dtype=np.float32)
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T.flatten())
        
        # Use vertex colors for grid
        use_vertex_color_loc = glGetUniformLocation(self.shader, "useVertexColor")
        glUniform1f(use_vertex_color_loc, 1.0)
        
        glBindVertexArray(self.grid_vao)
        glDrawArrays(GL_LINES, 0, self.grid_vertices)
        glBindVertexArray(0)
    
    def draw_cube(self, position, scale, color):
        """Draw a cube at the specified position with given scale and color"""
        # Create model matrix
        model = np.eye(4, dtype=np.float32)
        model = self.translate(model, position[0], position[1], position[2])
        model = self.scale_matrix(model, scale[0], scale[1], scale[2])
        
        # Set uniforms
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T.flatten())
        
        color_loc = glGetUniformLocation(self.shader, "objectColor")
        glUniform3f(color_loc, color[0], color[1], color[2])
        
        use_vertex_color_loc = glGetUniformLocation(self.shader, "useVertexColor")
        glUniform1f(use_vertex_color_loc, 0.0)
        
        # Draw solid cube
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindVertexArray(self.cube_vao)
        glDrawElements(GL_TRIANGLES, self.cube_indices, GL_UNSIGNED_INT, None)
        
        # Draw wireframe outline
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glUniform3f(color_loc, 0.0, 0.0, 0.0)
        glLineWidth(1.5)
        glDrawElements(GL_TRIANGLES, self.cube_indices, GL_UNSIGNED_INT, None)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBindVertexArray(0)
    
    def create_view_matrix(self):
        """Create view matrix with camera rotations"""
        view = np.eye(4, dtype=np.float32)
        
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
    
    def scale_matrix(self, m, x, y, z):
        """Apply scale to matrix"""
        scale = np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return np.dot(scale, m)
    
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
            
            # Get clicked object
            clicked_obj = self.get_object_at_position(event.x, event.y)
            if clicked_obj and 'piece_id' in clicked_obj:
                self.selected_piece = clicked_obj['piece_id']
                self.drag_offset = None 
            else:
                self.selected_piece = None
                
            return True
    
    def on_mouse_release(self, widget, event):
        if event.button == 1:
            if self.selected_piece is not None:
                self.snap_to_grid(self.selected_piece)
                
            self.last_mouse_pos = None
            self.selected_piece = None
            return True
    
    def on_mouse_motion(self, widget, event):
        if self.last_mouse_pos and self.selected_piece is not None:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            
            # Get camera vectors
            forward, right, up = self.get_camera_vectors()
            
            # Calculate movement in world space
            move_right = right * dx * 0.01
            move_up = up * -dy * 0.01  # Invert Y axis
            
            # Combine movement
            delta = move_right + move_up
            
            # Move the entire piece
            self.move_piece(self.selected_piece, delta)
            
            self.queue_render()
            self.last_mouse_pos = (event.x, event.y)
            return True
        elif self.last_mouse_pos:
            # Original camera rotation code
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]
            self.camera_rotation[1] += dx * 0.5
            self.queue_render()
            self.last_mouse_pos = (event.x, event.y)
            return True
    
    def on_scroll(self, widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom = max(2.0, self.zoom - 0.5)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom = min(50.0, self.zoom + 0.5)
        
        self.queue_render()
        return True
    
    def on_key_press(self, widget, event):
        self.keys_pressed.add(event.keyval)
        
        if self.render_timer is None:
            self.render_timer = GLib.timeout_add(16, self.update_movement)
        
        if event.keyval == Gdk.KEY_r or event.keyval == Gdk.KEY_R:
            self.camera_rotation = [0.0, 0.0]
            self.camera_position = [0.0, 1.7, 0.0]  
            self.zoom = 10.0
            self.queue_render()
            
        return True
    
    def on_key_release(self, widget, event):
        self.keys_pressed.discard(event.keyval)
        
        if not self.keys_pressed and self.render_timer:
            GLib.source_remove(self.render_timer)
            self.render_timer = None
            
        return True
    
    def update_movement(self):
        """Update movement based on pressed keys - relative to camera orientation"""
        forward, right, up = self.get_camera_vectors()
        
        movement = np.array([0.0, 0.0, 0.0])
        
        # WASD movement
        if Gdk.KEY_w in self.keys_pressed or Gdk.KEY_W in self.keys_pressed:
            movement += forward * self.movement_speed
        if Gdk.KEY_s in self.keys_pressed or Gdk.KEY_S in self.keys_pressed:
            movement -= forward * self.movement_speed
        if Gdk.KEY_a in self.keys_pressed or Gdk.KEY_A in self.keys_pressed:
            movement -= right * self.movement_speed
        if Gdk.KEY_d in self.keys_pressed or Gdk.KEY_D in self.keys_pressed:
            movement += right * self.movement_speed
        if Gdk.KEY_space in self.keys_pressed:
            movement[1] += self.movement_speed
        if Gdk.KEY_Shift_L in self.keys_pressed or Gdk.KEY_Shift_R in self.keys_pressed:
            movement[1] -= self.movement_speed
        
        # Apply movement
        new_pos = np.array(self.camera_position) + movement
        
        # Enforce ground constraint (y >= 0)
        if new_pos[1] < 0:
            new_pos[1] = 0
        
        self.camera_position = new_pos.tolist()
            
        self.queue_render()
        return True

    def get_object_at_position(self, x, y):
        """Get the object under mouse cursor"""
        self.make_current()
        
        # Read depth buffer at mouse position
        viewport_height = self.get_allocated_height()
        gl_y = viewport_height - y
        
        depth = glReadPixels(int(x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
        
        if depth >= 1.0:
            return None
        
        # Convert mouse position to world coordinates
        aspect = self.get_allocated_width() / self.get_allocated_height()
        projection = self.perspective(45.0, aspect, 0.1, 100.0)
        view = self.create_view_matrix()
        
        # Unproject the point
        mouse_pos = self.unproject(x, gl_y, depth, view, projection)
        
        # Find closest object
        closest_obj = None
        min_dist = float('inf')
        
        for obj in self.objects:
            if obj.get('is_shadow', False):
                continue
                
            obj_pos = np.array(obj['pos'])
            dist = np.linalg.norm(mouse_pos - obj_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        return closest_obj

    def unproject(self, winx, winy, depth, view, projection):
        """Convert window coordinates to world coordinates"""
        # Calculate inverse matrix
        inv = np.linalg.inv(np.dot(projection, view))
        
        # Normalized device coordinates
        x = (2.0 * winx) / self.get_allocated_width() - 1.0
        y = (2.0 * winy) / self.get_allocated_height() - 1.0
        z = 2.0 * depth - 1.0
        
        # Homogeneous coordinates
        point = np.array([x, y, z, 1.0])
        
        # Transform to world coordinates
        world = np.dot(inv, point)
        
        # Perspective division
        world /= world[3]
        
        return world[:3]

    def get_piece_cubes(self, piece_id):
        """Get all cubes belonging to a piece"""
        return [obj for obj in self.objects if obj.get('piece_id', None) == piece_id]

    def move_piece(self, piece_id, delta):
        """Move all cubes in a piece by delta"""
        for obj in self.objects:
            if obj.get('piece_id', None) == piece_id:
                obj['pos'][0] += delta[0]
                obj['pos'][1] += delta[1]
                obj['pos'][2] += delta[2]

    def snap_to_grid(self, piece_id):
        """Snap piece to nearest grid position"""
        cubes = self.get_piece_cubes(piece_id)
        if not cubes:
            return
            
        # Find average position of piece
        avg_pos = np.mean([cube['pos'] for cube in cubes], axis=0)
        
        # Calculate nearest grid position
        grid_size = 0.6  # Should match your grid spacing
        snapped_pos = [
            round(avg_pos[0] / grid_size) * grid_size,
            round(avg_pos[1] / grid_size) * grid_size,
            round(avg_pos[2] / grid_size) * grid_size
        ]
        
        # Calculate delta to move piece
        delta = np.array(snapped_pos) - avg_pos
        
        # Move piece
        self.move_piece(piece_id, delta)

    def draw_shadow_cube(self, position, scale, color):
        """Draw a shadow cube with transparency"""
        # Enable blending for shadows
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create model matrix
        model = np.eye(4, dtype=np.float32)
        model = self.translate(model, position[0], position[1], position[2])
        model = self.scale_matrix(model, scale[0], scale[1], scale[2])
        
        # Set uniforms
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T.flatten())
        
        color_loc = glGetUniformLocation(self.shader, "objectColor")
        glUniform3f(color_loc, color[0], color[1], color[2])
        
        alpha_loc = glGetUniformLocation(self.shader, "alpha")
        glUniform1f(alpha_loc, color[3] if len(color) > 3 else 1.0)
        
        use_vertex_color_loc = glGetUniformLocation(self.shader, "useVertexColor")
        glUniform1f(use_vertex_color_loc, 0.0)
        
        # Draw solid cube (no wireframe for shadows)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindVertexArray(self.cube_vao)
        glDrawElements(GL_TRIANGLES, self.cube_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        glDisable(GL_BLEND)

class CubeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="3D Cube World")
        self.set_default_size(900, 700)
        
        # Create a box layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(box)
        
        # Add title bar with info
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        info_box.set_margin_left(10)
        info_box.set_margin_right(10)
        info_box.set_margin_top(5)
        info_box.set_margin_bottom(5)
        
        # Title
        title_label = Gtk.Label()
        title_label.set_markup("<b>3D Cube World</b>")
        info_box.pack_start(title_label, False, False, 10)
        
        # Separator
        info_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 10)
        
        # Controls info
        controls_label = Gtk.Label()
        controls_label.set_markup(
            "<b>Controls:</b> " +
            "WASD: Move | Mouse: Look around | Scroll: Zoom | Space/Shift: Up/Down | R: Reset"
        )
        info_box.pack_start(controls_label, False, False, 0)
        
        box.pack_start(info_box, False, False, 0)
        
        # Add separator
        box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 0)
        
        # Add GL area
        self.gl_area = GLView()
        box.pack_start(self.gl_area, True, True, 0)
        
        # Status bar
        self.statusbar = Gtk.Statusbar()
        box.pack_start(self.statusbar, False, False, 0)
        
        # Update status periodically
        GLib.timeout_add(100, self.update_status)
        
        # Focus the GL area for keyboard input
        self.gl_area.grab_focus()
        
        self.connect("destroy", Gtk.main_quit)
    
    def update_status(self):
        """Update status bar with camera position"""
        if hasattr(self.gl_area, 'camera_position'):
            pos = self.gl_area.camera_position
            self.statusbar.pop(0)
            self.statusbar.push(0, 
                f"Camera Position: X: {pos[0]:.1f}, Y: {pos[1]:.1f}, Z: {pos[2]:.1f} | " +
                f"Objects: {len(self.gl_area.objects)}")
        return True

if __name__ == "__main__":
    window = CubeWindow()
    window.show_all()
    Gtk.main()