import open3d as o3d
import numpy as np
import cv2
import pyrender
import trimesh
import matplotlib.pyplot as plt

class Processor3D:
    def __init__(self, obj_path, template_width=9, template_height=16):
        self.obj_path = obj_path
        self.template_width = template_width
        self.template_height = template_height
        self.scale_factor = None  # Determined during alignment
        self.vis = None
        self.template_width = 9
        self.template_height = 16
        self.scale_factor = 0.2
        self.offset_z = 0.0
        self.offset_y = 0.0
        self.offset_x = 0.0
        self.offset_rot = 0.0
        self.rot_axes = 0
        self.direction_counter = 1
        self.remain_sitted_counter = 20
        self.start_rotation = True
        self.rotate_initially_once = True
        self.switch_direction = False


    def visualize_mesh(self):
        # Load the .obj file
        mesh = o3d.io.read_triangle_mesh(self.obj_path)

        # Ensure the mesh is not empty
        if not mesh.is_empty():
            # Get the vertices (points) as a NumPy array
            vertices = np.asarray(mesh.vertices)

            print("Vertices shape:", vertices.shape)
            print("First 5 vertices:\n", vertices[:5])  # Display the first 5 vertices

            # Visualize the mesh for context
            o3d.visualization.draw_geometries([mesh])
        else:
            print("Failed to load the .obj file.")

    def scale_dimensions(self):
        # Scaled dimensions
        scaled_width = self.template_width * self.scale_factor
        scaled_height = self.template_height * self.scale_factor
        scaled_depth = self.template_width * self.scale_factor  # Assuming the cube's depth matches the width
        return scaled_width, scaled_height, scaled_depth


    def align_point(self, v):
        """
        Aligns the given point `v` to match the scaled and centered cube's alignment.
        """
        scaled_width, scaled_height, scaled_depth = self.scale_dimensions()
        # Centering offsets
        offset_x = (self.template_width - scaled_width) / 2
        offset_y = (self.template_height - scaled_height) / 2
        offset_z = -scaled_depth / 2  # Centering the depth
        # Transformation to scale and center the points
        scaled_v = v * self.scale_factor  # Scale the point
        aligned_v = scaled_v + np.array([offset_x, offset_y, offset_z])  # Apply offsets
        return aligned_v


    def rot_x(self,t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return m

    def rot_y(self,t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[ct, 0, st], [0, 1, 0], [st, 0, ct]])
        return m

    def rot_z(self,t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        return m


    def initial_rotation(self, mesh):
        # Initial rotation (align the object)
        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(-np.pi / 2)
        mesh.apply_transform(T)
        for i in range(len(mesh.vertices)):
            mesh.vertices[i] = self.align_point(mesh.vertices[i])

    def initial_scaling(self, mesh, scale_factor=7.0):
        # Scaling
        scaling_matrix = np.eye(4)
        scaling_matrix[0, 0] = scaling_matrix[1, 1] = scaling_matrix[2, 2] = scale_factor
        mesh.apply_transform(scaling_matrix)

    def walk(self, mesh):
        self.initial_rotation(mesh)
        # rotate 45 degrees in x-y
        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = self.offset_x
        translation_matrix[1, 3] = self.offset_y
        self.direction_counter += 1
        if self.direction_counter % 30 == 0:
            self.switch_direction = ~self.switch_direction
        self.offset_x = self.offset_x - 0.1 if self.switch_direction else self.offset_x + 0.1
        self.offset_y = self.offset_y - 0.1 if self.switch_direction else self.offset_y + 0.1
        rotation_matrix = self.rot_z(np.radians(135)) if self.switch_direction else self.rot_z(np.radians(-45))
        animation_transform = np.eye(4)
        animation_transform[0:3, 0:3] = rotation_matrix  # Apply rotation
        animation_transform = animation_transform @ translation_matrix  # Apply jump
        mesh.apply_transform(animation_transform)
        current_centroid = np.mean(mesh.vertices, axis=0)
        target_center = np.array([
            self.template_width / 2,
            self.template_height / 2,
            self.template_width / 2  # Assuming the template is a cube
        ])
        # Calculate the offset needed to align with the diagonal
        recenter_offset = target_center - current_centroid

        # Create the recentering transformation
        recenter_matrix = np.eye(4)
        recenter_matrix[0:3, 3] = recenter_offset

        # Apply the recentering transformation
        mesh.apply_transform(recenter_matrix @ translation_matrix)



    def continuous_rotation(self, mesh):
        # Increment rotation angle
        rotation_step = np.radians(-10)  # Rotate 10 degrees per frame
        self.offset_rot += rotation_step
        if self.offset_rot <= -2 * np.pi:  # Reset after a full rotation
            self.offset_rot += 2 * np.pi
            self.start_rotation = False

        # Increment jump (Y offset)
        jump_amplitude = 4.0
        jump_step = 0.1
        self.offset_y += jump_step

        # Simulate a jump using a sinusoidal pattern
        jump_height = jump_amplitude * np.sin(self.offset_y)

        # Apply rotation and translation
        if self.rot_axes == 0:
            rotation_matrix = self.rot_x(self.offset_rot)  # Time-dependent rotation matrix
        elif self.rot_axes == 1:
            rotation_matrix = self.rot_y(self.offset_rot)
        else:
            rotation_matrix = self.rot_z(self.offset_rot)
        translation_matrix = np.eye(4)
        translation_matrix[1, 3] = jump_height  # Apply jump to Y axis

        animation_transform = np.eye(4)
        animation_transform[0:3, 0:3] = rotation_matrix  # Apply rotation
        animation_transform = animation_transform @ translation_matrix  # Apply jump
        mesh.apply_transform(animation_transform)

    def draw(self, frame_for_display, rvec, tvec, calibration_matrix, move_object=True, show=False):
        mesh = trimesh.load(self.obj_path)
        if isinstance(mesh, trimesh.Scene):
            # If it's a Scene, access its meshes via the 'geometry' attribute
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        mesh.rezero()
        self.initial_scaling(mesh)
        if move_object and self.start_rotation:
            # === Frame-Based Animation ===
            if self.rotate_initially_once:
                self.rotate_initially_once = False
                self.initial_rotation(mesh)
            else:
                #self.continuous_rotation(mesh)
                self.walk(mesh)
        else:
            self.initial_rotation(mesh)
            self.remain_sitted_counter -= 1
            if self.remain_sitted_counter < 0:
                self.remain_sitted_counter = 20
                self.start_rotation = True
                self.rot_axes += 1
                if self.rot_axes > 2:
                    self.rot_axes = 0

        # Camera pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Align vertices
        # for i in range(len(mesh.vertices)):
        #     mesh.vertices[i] = self.align_point(mesh.vertices[i])

        # Add mesh to the scene
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        scene.add(mesh)

        # Add camera
        self.camera = pyrender.IntrinsicsCamera(
            calibration_matrix[0, 0], calibration_matrix[1, 1], calibration_matrix[0, 2], calibration_matrix[1, 2],
            zfar=10000, name="cam"
        )
        light_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 10.0],
                [0.0, 0.0, 1.0, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.cam_node = scene.add(self.camera, pose=light_pose)

        # Add light
        light = pyrender.SpotLight(color=255 * np.ones(3), intensity=3000.0, innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=light_pose)

        # Render
        self.r = pyrender.OffscreenRenderer(frame_for_display.shape[1], frame_for_display.shape[0])
        self.flag = pyrender.constants.RenderFlags.RGBA
        scene.set_pose(self.cam_node, camera_pose)
        color, depth = self.r.render(scene, flags=self.flag)
        frame_for_display[color[:, :, 3] != 0] = color[:, :, [2, 1, 0]][color[:, :, 3] != 0]

        if show:
            plt.imshow(frame_for_display)
            plt.show(block=True)
        return frame_for_display


    def draw_old(self, frame_for_display, rvec, tvec, calibration_matrix, show=False):
        mesh = trimesh.load(self.obj_path)
        if isinstance(mesh, trimesh.Scene):
            # If it's a Scene, access its meshes via the 'geometry' attribute
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        T = np.eye(4)
        T[0:3, 0:3] = self.rot(np.pi / 2)
        mesh.apply_transform(T)

        # Generate vertex colors once and store them
        if not hasattr(self, "cached_vertex_colors"):
            np.random.seed(42)
            self.cached_vertex_colors = np.random.rand(len(mesh.vertices), 3)
        mesh.vertex_colors = self.cached_vertex_colors
        # Convert trimesh to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()


        # ===== Update cam pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()

            self.vis.create_window(width=frame_for_display.shape[1], height=frame_for_display.shape[0])

            # Add the mesh to the visualizer
            self.vis.add_geometry(o3d_mesh)

        # Set the camera pose using Open3D's view control
        view_control = self.vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(
            self.create_o3d_camera_param(frame_for_display, camera_pose, calibration_matrix))
        for i in range(len(o3d_mesh.vertices)):
            o3d_mesh.vertices[i] = self.align_point(o3d_mesh.vertices[i])
        self.vis.update_geometry(o3d_mesh)  # Update to get the position after applying the transformation
        self.vis.poll_events()
        self.vis.update_renderer()
        # Capture the rendered image
        rendered_image = self.vis.capture_screen_float_buffer(do_render=True)
        rendered_image = np.asarray(rendered_image)
        rendered_image = (rendered_image * 255).astype(np.uint8)

        # Create mask (black pixels are part of the mesh)
        mask = np.all(rendered_image != [255, 255, 255], axis=-1)
        if np.any(mask):
            # Resize Mask and Rendered Image
            mask = cv2.resize(mask.astype(np.uint8), (frame_for_display.shape[1], frame_for_display.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
            rendered_image = cv2.resize(rendered_image, (frame_for_display.shape[1], frame_for_display.shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
            mask = mask.astype(np.float32)  # convert to float
            mask = np.clip(mask, 0, 1)  # make sure mask values are between 0 and 1
            mask = np.stack([mask] * 3, axis=-1)  # make it 3D

            # Perform alpha blending
            frame_for_display = (frame_for_display * (1 - mask) + rendered_image * mask).astype(np.uint8)
        if show:
            #o3d.visualization.draw_geometries([o3d_mesh], window_name="Open3D Visualization")
            plt.imshow(frame_for_display)
            plt.show()

        return frame_for_display

    def create_o3d_camera_param(self, frame_for_display, camera_pose, camera_matrix):

            camera_param = o3d.camera.PinholeCameraParameters()
            camera_param.intrinsic = o3d.camera.PinholeCameraIntrinsic(frame_for_display.shape[1], frame_for_display.shape[0], camera_matrix[0, 0], camera_matrix[1, 1],
                                                                       camera_matrix[0, 2], camera_matrix[1, 2])
            camera_param.extrinsic = camera_pose
            return camera_param