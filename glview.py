import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib, GObject
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import random
import ctypes
from PIL import Image 

# Vertex shader with support for per-vertex colors
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texCoord; // Add texture coordinate attribute

out vec3 vertexColor;
out vec2 TexCoord; // Pass texCoord to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    vertexColor = color;
    TexCoord = texCoord;
}
"""

# Fragment shader
fragment_shader = """
#version 330 core
in vec3 vertexColor;
in vec2 TexCoord;

out vec4 frag_color;

uniform vec3 objectColor;
uniform float useVertexColor;
uniform float useTexture; // Add a switch for texturing
uniform sampler2D ourTexture; // The texture sampler
uniform float alpha = 1.0;

void main() {
    if (useTexture > 0.5) {
        frag_color = texture(ourTexture, TexCoord);
    } else if (useVertexColor > 0.5) {
        frag_color = vec4(vertexColor, 1.0);
    } else {
        frag_color = vec4(objectColor, alpha);
    }
}
"""


class GLView(Gtk.GLArea):
    __gsignals__ = {
        'puzzle-completed': (GObject.SIGNAL_RUN_FIRST, None, ())
    }

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
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.SCROLL_MASK
            | Gdk.EventMask.KEY_PRESS_MASK
            | Gdk.EventMask.KEY_RELEASE_MASK
        )

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
        self.floor_vao = None
        self.floor_texture = None
        self.shader = None

        self.selected_piece = None
        self.drag_offset = None

        self.grid_step = 0.6

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
            compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )

        self.load_texture("PavingStones.png") 

        # Create geometry
        self.setup_cube()
        self.setup_grid()
        self.setup_floor()
        self.create_scene_objects()
        GLib.idle_add(self.update_controls_hud)

    def load_texture(self, filename):
        """Loads a texture from an image file."""
        self.floor_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.floor_texture)

        # Set texture wrapping and filtering options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load image with Pillow
        try:
            image = Image.open(filename)
            # OpenGL expects the 0.0 coordinate on the y-axis to be at the bottom,
            # but images usually have 0.0 at the top.
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
        except FileNotFoundError:
            print(f"Error: Texture file '{filename}' not found.")
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)

    def setup_floor(self):
        """Create a textured floor quad."""
        y_level = -0.51  # Slightly below the grid lines to avoid z-fighting
        size = 50.0  # How large the floor is
        texture_repeats = 50.0 # How many times the texture repeats across the floor

        vertices = np.array([
            # positions      # texture coords
            -size, y_level,  size,  0.0, texture_repeats,
             size, y_level,  size,  texture_repeats, texture_repeats,
             size, y_level, -size,  texture_repeats, 0.0,

            -size, y_level,  size,  0.0, texture_repeats,
             size, y_level, -size,  texture_repeats, 0.0,
            -size, y_level, -size,  0.0, 0.0
        ], dtype=np.float32)

        self.floor_vao = glGenVertexArrays(1)
        glBindVertexArray(self.floor_vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Texture coord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def setup_cube(self):
        """Create a unit cube mesh"""
        vertices = []
        colors = []

        # Define cube faces with colors
        faces = [
            # Front (z=0.5) - slightly different shades for each face
            ([-0.5, -0.5, 0.5],
             [0.5, -0.5, 0.5],
             [0.5, 0.5, 0.5],
             [-0.5, 0.5, 0.5]),
            # Back (z=-0.5)
            (
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ),
            # Top (y=0.5)
            ([-0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, -0.5],
             [-0.5, 0.5, -0.5]),
            # Bottom (y=-0.5)
            (
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ),
            # Right (x=0.5)
            ([0.5, -0.5, 0.5],
             [0.5, -0.5, -0.5],
             [0.5, 0.5, -0.5],
             [0.5, 0.5, 0.5]),
            # Left (x=-0.5)
            (
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ),
        ]

        indices = []
        vertex_count = 0

        for face in faces:
            for vertex in face:
                vertices.extend(vertex)
                colors.extend([0.8, 0.8, 0.8])  # Default gray color

            # Two triangles per face
            base = vertex_count
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
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
        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.nbytes + colors.nbytes,
            None,
            GL_STATIC_DRAW
        )
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBufferSubData(GL_ARRAY_BUFFER,
                        vertices.nbytes,
                        colors.nbytes,
                        colors)

        # Index buffer
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes)
        )
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        self.cube_indices = len(indices)

    def setup_grid(self):
        """Create a grid floor"""
        vertices = []
        colors = []

        grid_size = 20
        grid_step = 1.0
        y_level = -0.5  # Floor level

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
        glBufferData(
            GL_ARRAY_BUFFER, vertices.nbytes + colors.nbytes, None, GL_STATIC_DRAW
        )
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, colors.nbytes, colors)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes)
        )
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        self.grid_vertices = len(vertices) // 3

    def is_valid_position(self, piece_type, position, rotation):
        """Check if position is valid (similar to first implementation)"""
        new_positions = self.get_piece_positions(piece_type, position, rotation)

        # Check collision with occupied spaces
        for obj in self.objects:
            if not obj.get("is_shadow", False) and "piece_id" in obj:
                obj_pos = tuple(round(p) for p in obj["pos"])
                if obj_pos in new_positions:
                    return False

        return True

    def is_valid_movement(self, piece_id, new_positions):
        """Check if piece movement is valid"""
        # Check each new position
        for new_pos in new_positions:
            # Check collision with other pieces
            for obj in self.objects:
                if (
                    not obj.get("is_shadow", False)
                    and obj.get("piece_id") != piece_id
                    and "piece_id" in obj
                ):
                    if np.allclose(new_pos, obj["pos"], atol=0.1):
                        return False

            # Optional: Check bounds (keep pieces within reasonable area)
            if abs(new_pos[0]) > 10 or abs(new_pos[1]) > 10 or abs(new_pos[2]) > 10:
                return False

        return True

    def is_valid_rotation(self, piece_id, new_positions):
        """Check if piece rotation is valid"""
        return self.is_valid_movement(piece_id, new_positions)

    def get_piece_positions(self, piece_type, position, rotation):
        """Get all cube positions for a piece type at given position/rotation"""
        # This would need to be adapted from the Pygame version
        # For now, using the current piece positions
        piece_id = None
        for obj in self.objects:
            if obj.get("piece_type") == piece_type:
                piece_id = obj.get("piece_id")
                break

        if piece_id is None:
            return set()

        positions = set()
        for obj in self.objects:
            if obj.get("piece_id") == piece_id:
                positions.add(tuple(round(p) for p in obj["pos"]))

        return positions

    def create_scene_objects(self):
        """Create the 7 Soma cube pieces in the scene"""
        self.objects = []
        cube_size = 1.0  # Integer grid size
        grid_size = 3
        grid_spacing = 1.0
        shadow_color = [0.05, 0.05, 0.05, 0.7]
        base_y = 1  # Base height for the 3x3x3 grid

        # Create shadow cubes (the target 3x3x3 grid)
        for x in range(grid_size):
            for z in range(grid_size):
                for y in range(grid_size):
                    self.objects.append(
                        {
                            "pos": [
                                (x - 1) * grid_spacing,  # -1, 0, 1
                                base_y + (y - 1) * grid_spacing,  # 0, 1, 2
                                (z - 1) * grid_spacing,  # -1, 0, 1
                            ],
                            "color": shadow_color,
                            "scale": [cube_size * 0.98] * 3,
                            "is_shadow": True,
                        }
                    )

        # Define the 7 Soma pieces
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
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        ]

        # Colors for pieces (bright, distinct colors)
        piece_colors = [
            [0.9, 0.2, 0.2],  # Red
            [0.2, 0.9, 0.2],  # Green
            [0.2, 0.2, 0.9],  # Blue
            [0.9, 0.9, 0.2],  # Yellow
            [0.9, 0.2, 0.9],  # Magenta
            [0.2, 0.9, 0.9],  # Cyan
            [0.9, 0.5, 0.2],  # Orange
        ]

        # Place pieces around the 3x3x3 grid (not overlapping)
        start_x = -8  # Start position for piece queue
        piece_spacing = 4  # Space between pieces

        for i, piece in enumerate(pieces):
            piece_color = piece_colors[i]

            # Calculate base position for this piece
            base_x = start_x + (i * piece_spacing)
            base_y = 0  # Place pieces on the ground
            base_z = -5  # In front of the grid

            # Create cubes for this piece
            for cube_offset in piece:
                self.objects.append(
                    {
                        "pos": [
                            base_x + cube_offset[0] * cube_size,
                            base_y + cube_offset[1] * cube_size,
                            base_z + cube_offset[2] * cube_size,
                        ],
                        "color": piece_color,
                        "scale": [cube_size] * 3,
                        "piece_id": i,  # Identify which piece this cube belongs to
                        "piece_type": f"piece_{i}",  # Add piece type for reference
                    }
                )

    def get_random_color(self):
        """Generate a random RGB color with good visibility"""
        return [
            random.uniform(0.3, 0.9),  # Red
            random.uniform(0.3, 0.9),  # Green
            random.uniform(0.3, 0.9),  # Blue
        ]

    def get_camera_vectors(self):
        """Calculate forward, right, and up vectors based on camera rotation"""
        pitch = math.radians(self.camera_rotation[0])
        yaw = math.radians(self.camera_rotation[1])

        forward = np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                -math.sin(pitch),
                -math.cos(pitch) * math.cos(yaw),
            ]
        )

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

        depth = glReadPixels(int(x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][
            0
        ]

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
        self.draw_floor()

        # Enable blending for shadows (should be done here, before drawing shadows)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # First draw all shadow cubes
        for obj in self.objects:
            if obj.get("is_shadow", False):
                self.draw_shadow_cube(obj["pos"], obj["scale"], obj["color"])

        # Disable blending for regular objects
        glDisable(GL_BLEND)

        # Then draw all regular cubes
        for obj in self.objects:
            if not obj.get("is_shadow", False):
                self.draw_cube(obj["pos"], obj["scale"], obj["color"])

        glUseProgram(0)
        return True

    def draw_floor(self):
        """Draw the textured floor."""
        model = np.eye(4, dtype=np.float32)
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T.flatten())

        # Set uniforms to use the texture
        glUniform1f(glGetUniformLocation(self.shader, "useTexture"), 1.0)
        glUniform1f(glGetUniformLocation(self.shader, "useVertexColor"), 0.0)

        # Bind the texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.floor_texture)
        glUniform1i(glGetUniformLocation(self.shader, "ourTexture"), 0)

        # Draw the floor
        glBindVertexArray(self.floor_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        # Reset texture uniform so other objects are not affected
        glUniform1f(glGetUniformLocation(self.shader, "useTexture"), 0.0)

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

        # Highlight selected object
        if self.selected_object and np.allclose(
            self.selected_object["pos"], position, atol=0.01
        ):
            glUniform3f(
                glGetUniformLocation(self.shader, "objectColor"),
                min(color[0] * 1.5, 1.0),
                min(color[1] * 1.5, 1.0),
                min(color[2] * 1.5, 1.0),
            )
        else:
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
        glUniform3f(glGetUniformLocation(self.shader, "objectColor"), 0.0, 0.0, 0.0)
        glLineWidth(1.5)
        glDrawElements(GL_TRIANGLES, self.cube_indices, GL_UNSIGNED_INT, None)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindVertexArray(0)

    def create_view_matrix(self):
        """Create view matrix with camera rotations"""
        view = np.eye(4, dtype=np.float32)

        view = self.translate(
            view,
            -self.camera_position[0],
            -self.camera_position[1],
            -(self.camera_position[2] + self.zoom),
        )

        view = self.rotate_x(view, self.camera_rotation[0])
        view = self.rotate_y(view, self.camera_rotation[1])

        return view

    def perspective(self, fovy, aspect, near, far):
        """Create perspective projection matrix"""
        f = 1.0 / math.tan(math.radians(fovy) / 2.0)
        nf = 1.0 / (near - far)

        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) * nf, 2 * far * near * nf],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def translate(self, m, x, y, z):
        """Apply translation to matrix"""
        trans = np.array(
            [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float32
        )
        return np.dot(trans, m)

    def scale_matrix(self, m, x, y, z):
        """Apply scale to matrix"""
        scale = np.array(
            [[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        return np.dot(scale, m)

    def rotate_x(self, m, angle):
        """Rotate around X axis"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        rot = np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        return np.dot(rot, m)

    def rotate_y(self, m, angle):
        """Rotate around Y axis"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        rot = np.array(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        return np.dot(rot, m)

    def rotate_z(self, m, angle):
        """Rotate around Z axis"""
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        rot = np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        return np.dot(rot, m)

    def on_mouse_press(self, widget, event):
        if event.button == 1:
            self.grab_focus()

            # Get clicked object
            clicked_obj = self.get_object_at_position(event.x, event.y)
            if clicked_obj and not clicked_obj.get("is_shadow", False):
                self.selected_object = clicked_obj
                print(f"Selected object at position: {clicked_obj['pos']}")
            else:
                self.selected_object = None
                self.last_mouse_pos = (event.x, event.y)
                print("No object selected")

            self.queue_render()
            return True

    def on_mouse_release(self, widget, event):
        if event.button == 1:
            self.last_mouse_pos = None
            return True

    def on_mouse_motion(self, widget, event):
        if self.last_mouse_pos and self.selected_object is None:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]

            # Update camera rotation angles
            self.camera_rotation[1] += dx * 0.5
            # Clamp up/down rotation to prevent flipping
            self.camera_rotation[0] = max(
                -89.0, min(89.0, self.camera_rotation[0] + dy * 0.5)
            )

            self.queue_render()
            self.last_mouse_pos = (event.x, event.y)

            # Update the controls HUD since camera orientation changed
            self.update_controls_hud()
            return True
        return False

    def on_scroll(self, widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom = max(2.0, self.zoom - 0.5)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom = min(50.0, self.zoom + 0.5)

        self.queue_render()
        return True

    def on_key_press(self, widget, event):
        # Handle object movement if an object is selected
        if self.selected_object:
            moved = False

            # Discrete movement with UIOJKL
            if event.keyval == Gdk.KEY_u or event.keyval == Gdk.KEY_U:  # Up
                self.move_object_discrete(self.selected_object, [0, 1, 0])
                moved = True
            elif event.keyval == Gdk.KEY_o or event.keyval == Gdk.KEY_O:  # Down
                self.move_object_discrete(self.selected_object, [0, -1, 0])
                moved = True
            elif event.keyval == Gdk.KEY_j or event.keyval == Gdk.KEY_J:  # Left
                self.move_object_discrete(self.selected_object, [-1, 0, 0])
                moved = True
            elif event.keyval == Gdk.KEY_l or event.keyval == Gdk.KEY_L:  # Right
                self.move_object_discrete(self.selected_object, [1, 0, 0])
                moved = True
            elif event.keyval == Gdk.KEY_i or event.keyval == Gdk.KEY_I:  # Forward
                self.move_object_discrete(self.selected_object, [0, 0, -1])
                moved = True
            elif event.keyval == Gdk.KEY_k or event.keyval == Gdk.KEY_K:  # Backward
                self.move_object_discrete(self.selected_object, [0, 0, 1])
                moved = True

            # Rotation with number keys
            elif event.keyval == Gdk.KEY_1:
                self.rotate_object(
                    self.selected_object, "x"
                )  # No angle parameter, defaults to 90
                moved = True
            elif event.keyval == Gdk.KEY_2:
                self.rotate_object(self.selected_object, "y")
                moved = True
            elif event.keyval == Gdk.KEY_3:
                self.rotate_object(self.selected_object, "z")
                moved = True

            if moved:
                self.queue_render()
                return True

        # Original camera movement code
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

        depth = glReadPixels(int(x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][
            0
        ]

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
        min_dist = float("inf")

        for obj in self.objects:
            if obj.get("is_shadow", False):
                continue

            obj_pos = np.array(obj["pos"])
            dist = np.linalg.norm(mouse_pos - obj_pos)

            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        if min_dist > 1.2:
            return None

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
        return [obj for obj in self.objects if obj.get("piece_id", None) == piece_id]

    def move_piece(self, piece_id, delta):
        """Move all cubes in a piece by delta with collision checking"""
        # First, calculate all new positions
        test_positions = []
        piece_cubes = []

        for obj in self.objects:
            if obj.get("piece_id", None) == piece_id:
                piece_cubes.append(obj)
                new_pos = [
                    obj["pos"][0] + delta[0],
                    obj["pos"][1] + delta[1],
                    obj["pos"][2] + delta[2],
                ]
                test_positions.append(new_pos)

        # Check if move is valid
        if not self.check_collision_at_positions(piece_id, test_positions):
            # Apply the movement
            for i, cube in enumerate(piece_cubes):
                cube["pos"] = test_positions[i]

    def check_bounds(self, positions):
        """Check if positions are within reasonable bounds"""
        for pos in positions:
            if abs(pos[0]) > 20 or pos[1] < 0 or pos[1] > 10 or abs(pos[2]) > 15:
                return False
        return True

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

    def is_valid_placement(self, piece_id):
        """Check if current piece placement is valid (all cubes align with shadow grid)"""
        cubes = self.get_piece_cubes(piece_id)
        if not cubes:
            return False

        shadow_positions = []
        for obj in self.objects:
            if obj.get("is_shadow", False):
                shadow_positions.append(np.array(obj["pos"]))

        # Check if all piece cubes align with shadow positions
        for cube in cubes:
            cube_pos = np.array(cube["pos"])
            aligned = False

            for shadow_pos in shadow_positions:
                if np.allclose(cube_pos, shadow_pos, atol=0.1):
                    aligned = True
                    break

            if not aligned:
                return False

        return True

    def check_collision(self, piece_id):
        """Check if piece collides with other placed pieces"""
        piece_cubes = self.get_piece_cubes(piece_id)
        if not piece_cubes:
            return False

        # Get positions of all other pieces
        other_positions = []
        for obj in self.objects:
            if (
                not obj.get("is_shadow", False)
                and obj.get("piece_id", None) != piece_id
            ):
                other_positions.append(np.array(obj["pos"]))

        # Check for collisions
        for cube in piece_cubes:
            cube_pos = np.array(cube["pos"])
            for other_pos in other_positions:
                if np.allclose(cube_pos, other_pos, atol=0.1):
                    return True  # Collision detected

        return False

    def check_puzzle_complete(self):
        """Check if all 27 positions in the 3x3x3 cube are filled"""
        shadow_positions = []
        filled_positions = []

        for obj in self.objects:
            if obj.get("is_shadow", False):
                shadow_positions.append(tuple(obj["pos"]))
            elif "piece_id" in obj:
                filled_positions.append(tuple(obj["pos"]))

        if len(filled_positions) != len(shadow_positions):
            return False

        # Check if all shadow positions are filled
        for shadow_pos in shadow_positions:
            filled = False
            for filled_pos in filled_positions:
                if np.allclose(shadow_pos, filled_pos, atol=0.1):
                    filled = True
                    break
            if not filled:
                return False
        print("############## bug1")
        return True

    def move_object_discrete(self, obj, direction):
        """Move object in discrete steps with collision and bounds detection"""
        if obj and not obj.get("is_shadow", False):
            grid_step = 1.0

            if "piece_id" in obj:
                piece_id = obj["piece_id"]

                # Calculate new positions
                test_positions = []
                piece_cubes = []

                for cube in self.objects:
                    if cube.get("piece_id") == piece_id:
                        piece_cubes.append(cube)
                        new_pos = [
                            cube["pos"][0] + direction[0] * grid_step,
                            cube["pos"][1] + direction[1] * grid_step,
                            cube["pos"][2] + direction[2] * grid_step,
                        ]
                        test_positions.append(new_pos)

                # Check bounds and collisions
                if self.check_bounds(
                    test_positions
                ) and not self.check_collision_at_positions(piece_id, test_positions):
                    # Apply the movement
                    for i, cube in enumerate(piece_cubes):
                        cube["pos"] = test_positions[i]

                    if self.check_puzzle_complete():
                        self.emit('puzzle-completed')

                else:
                    if not self.check_bounds(test_positions):
                        print("Out of bounds!")
                    else:
                        print("Collision detected!")
                    GLib.timeout_add(2000, lambda: self.statusbar.pop(0))

    def rotate_object(self, obj, axis, angle=90):
        """Rotate entire piece around its center with collision detection"""
        if obj and not obj.get("is_shadow", False):
            if "piece_id" in obj:
                piece_id = obj["piece_id"]

                # Get all cubes of this piece
                piece_cubes = [o for o in self.objects if o.get("piece_id") == piece_id]

                # Calculate piece center (rounded to grid)
                positions = [np.array(cube["pos"]) for cube in piece_cubes]
                center = np.mean(positions, axis=0)
                center = np.round(center)

                # Calculate new positions after rotation
                test_positions = []
                for cube in piece_cubes:
                    # Translate to origin
                    relative_pos = np.array(cube["pos"]) - center

                    # Apply rotation
                    if axis == "x":
                        rotated = self.rotate_vector_x(relative_pos, 90)
                    elif axis == "y":
                        rotated = self.rotate_vector_y(relative_pos, 90)
                    elif axis == "z":
                        rotated = self.rotate_vector_z(relative_pos, 90)

                    # Translate back and round
                    new_pos = rotated + center
                    new_pos = [round(p) for p in new_pos.tolist()]
                    test_positions.append(new_pos)

                # Check if rotation is valid (no collisions)
                if not self.check_collision_at_positions(piece_id, test_positions):
                    # Apply the rotation
                    for i, cube in enumerate(piece_cubes):
                        cube["pos"] = test_positions[i]

                    if self.check_puzzle_complete():
                        self.emit('puzzle-completed')
                else:
                    print("Collision detected! Rotation blocked.")
                    GLib.timeout_add(2000, lambda: self.statusbar.pop(0))

    def rotate_vector_x(self, vec, angle):
        """Rotate a 3D vector around X axis"""
        rad = math.radians(angle)
        c = math.cos(rad)
        s = math.sin(rad)
        return np.array([vec[0], vec[1] * c - vec[2] * s, vec[1] * s + vec[2] * c])

    def rotate_vector_y(self, vec, angle):
        """Rotate a 3D vector around Y axis"""
        rad = math.radians(angle)
        c = math.cos(rad)
        s = math.sin(rad)
        return np.array([vec[0] * c + vec[2] * s, vec[1], -vec[0] * s + vec[2] * c])

    def rotate_vector_z(self, vec, angle):
        """Rotate a 3D vector around Z axis"""
        rad = math.radians(angle)
        c = math.cos(rad)
        s = math.sin(rad)
        return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c, vec[2]])

    def check_collision_at_positions(self, piece_id, test_positions):
        """Check if any of the test positions collide with existing pieces"""
        # Get all occupied positions except from the current piece
        occupied_positions = set()

        for obj in self.objects:
            if (
                not obj.get("is_shadow", False)
                and obj.get("piece_id") is not None
                and obj.get("piece_id") != piece_id
            ):
                # Round positions to ensure integer comparison
                pos = tuple(round(p) for p in obj["pos"])
                occupied_positions.add(pos)

        # Check if any test position collides
        for test_pos in test_positions:
            rounded_pos = tuple(round(p) for p in test_pos)
            if rounded_pos in occupied_positions:
                return True  # Collision detected

        return False  # No collision

    def update_controls_hud(self):
        if not hasattr(self, "hud_labels"):
            return

        # Define the world axes and their corresponding movement keys
        key_map = {
            "x": "L",
            "-x": "J",
            "y": "U",
            "-y": "O",
            "z": "K",
            "-z": "I",  # Note: Z is often "into the screen"
        }
        world_axes = {
            "x": np.array([1, 0, 0]),
            "-x": np.array([-1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "-y": np.array([0, -1, 0]),
            "z": np.array([0, 0, 1]),
            "-z": np.array([0, 0, -1]),
        }

        cam_forward, cam_right, cam_up = self.get_camera_vectors()

        # Helper function to find the best matching world axis
        def get_best_axis(cam_vector):
            dots = [np.dot(cam_vector, axis) for axis in world_axes.values()]
            max_dot_index = np.argmax(
                np.abs(dots)
            )  # Use abs to find alignment regardless of sign
            best_axis_name = list(world_axes.keys())[max_dot_index]
            # Refine sign
            if dots[max_dot_index] < 0:
                # Flip the sign if the dot product is negative
                best_axis_name = (
                    best_axis_name.replace("-", "")
                    if "-" in best_axis_name
                    else "-" + best_axis_name
                )
            return best_axis_name

        # Determine the mapping for each camera direction
        right_axis = get_best_axis(cam_right)
        left_axis = get_best_axis(-cam_right)
        up_axis = get_best_axis(cam_up)
        down_axis = get_best_axis(-cam_up)
        fwd_axis = get_best_axis(cam_forward)
        back_axis = get_best_axis(-cam_forward)

        # Update the HUD labels
        self.hud_labels["right"].set_markup(f"<b>Right:</b>    ({key_map[right_axis]})")
        self.hud_labels["left"].set_markup(f"<b>Left:</b>     ({key_map[left_axis]})")
        self.hud_labels["up"].set_markup(f"<b>Up:</b>       ({key_map[up_axis]})")
        self.hud_labels["down"].set_markup(f"<b>Down:</b>     ({key_map[down_axis]})")
        self.hud_labels["forward"].set_markup(f"<b>Forward:</b>  ({key_map[fwd_axis]})")
        self.hud_labels["backward"].set_markup(
            f"<b>Backward:</b> ({key_map[back_axis]})"
        )

    def reset_puzzle(self):
        """Resets all pieces to their starting positions."""
        self.create_scene_objects()
        self.selected_object = None
        self.queue_render()


class CubeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="3D Cube World")
        self.set_default_size(900, 700)

        # Main vertical box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(main_box)

        # Info bar at top
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        info_box.set_margin_left(10)
        info_box.set_margin_right(10)
        info_box.set_margin_top(5)
        info_box.set_margin_bottom(5)

        title_label = Gtk.Label()
        title_label.set_markup("<b>3D Cube World</b>")
        info_box.pack_start(title_label, False, False, 10)

        info_box.pack_start(
            Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 10
        )

        controls_label = Gtk.Label()
        controls_label.set_markup(
            "<b>Controls:</b> "
            + "WASD: Camera | Click: Select | UIOJKL: Move Object | 123: Rotate XYZ | R: Reset"
        )
        info_box.pack_start(controls_label, False, False, 0)

        main_box.pack_start(info_box, False, False, 0)
        main_box.pack_start(
            Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 0
        )

        # --- Overlay and GLView setup ---
        overlay = Gtk.Overlay()
        main_box.pack_start(overlay, True, True, 0)

        self.gl_area = GLView()
        overlay.add(self.gl_area)

        # HUD Grid
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
        self.label_up = Gtk.Label(label="Up:")
        self.label_down = Gtk.Label(label="Down:")
        self.label_left = Gtk.Label(label="Left:")
        self.label_right = Gtk.Label(label="Right:")
        self.label_fwd = Gtk.Label(label="Forward:")
        self.label_back = Gtk.Label(label="Backward:")

        controls_hud.attach(hud_title, 0, 0, 2, 1)
        controls_hud.attach(self.label_up, 0, 1, 2, 1)
        controls_hud.attach(self.label_down, 0, 2, 2, 1)
        controls_hud.attach(self.label_left, 0, 3, 2, 1)
        controls_hud.attach(self.label_right, 0, 4, 2, 1)
        controls_hud.attach(self.label_fwd, 0, 5, 2, 1)
        controls_hud.attach(self.label_back, 0, 6, 2, 1)

        # Link HUD to GLView
        self.gl_area.hud_labels = {
            "up": self.label_up,
            "down": self.label_down,
            "left": self.label_left,
            "right": self.label_right,
            "forward": self.label_fwd,
            "backward": self.label_back,
        }

        # Status bar
        self.statusbar = Gtk.Statusbar()
        main_box.pack_start(self.statusbar, False, False, 0)
        self.gl_area.statusbar = self.statusbar

        # Focus and periodic updates
        self.gl_area.grab_focus()
        GLib.timeout_add(100, self.update_status)

        self.connect("destroy", Gtk.main_quit)

    def update_status(self):
        """Update status bar with camera position and object count."""
        if hasattr(self.gl_area, "camera_position"):
            pos = self.gl_area.camera_position
            self.statusbar.pop(0)
        return True


if __name__ == "__main__":
    window = CubeWindow()
    window.show_all()
    Gtk.main()
