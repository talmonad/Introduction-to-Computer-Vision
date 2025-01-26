import open3d as o3d
import numpy as np
import cv2
import pyrender
import trimesh

class Processor3D:
    def __init__(self, obj_path, template_width=9, template_height=16):
        self.obj_path = obj_path
        self.template_width = template_width
        self.template_height = template_height
        self.scale_factor = None  # Determined during alignment

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

    def align_point(self, v):
        """
        Aligns the given point `v` to match the scaled and centered cube's alignment.
        """
        # Cube dimensions and scaling
        template_width = 9
        template_height = 16
        scale_factor = 0.2

        # Scaled dimensions
        scaled_width = template_width * scale_factor
        scaled_height = template_height * scale_factor
        scaled_depth = template_width * scale_factor  # Assuming the cube's depth matches the width

        # Centering offsets
        offset_x = (template_width - scaled_width) / 2
        offset_y = (template_height - scaled_height) / 2
        offset_z = -scaled_depth / 2  # Centering the depth

        # Transformation to scale and center the points
        scaled_v = v * scale_factor  # Scale the point
        aligned_v = scaled_v + np.array([offset_x, offset_y, offset_z])  # Apply offsets
        return aligned_v


    def rot(self,t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return m


    def draw(self, frame_for_display, rvec, tvec, calibration_matrix):
        mesh = trimesh.load(self.obj_path)
        if isinstance(mesh, trimesh.Scene):
            # If it's a Scene, access its meshes via the 'geometry' attribute
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        mesh.rezero()
        scaling_matrix = np.eye(4)
        scaling_matrix[0, 0] = scaling_matrix[1, 1] = scaling_matrix[2, 2] = 7.0

        # Apply the scaling transformation to the mesh
        mesh.apply_transform(scaling_matrix)
        T = np.eye(4)
        T[0:3, 0:3] = self.rot(-np.pi / 2)
        mesh.apply_transform(T)
        # ===== update cam pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)

        # opengl transformation
        # https://stackoverflow.com/a/18643735/4879610
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        # 180 about x
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # Load the trimesh and put it in a scene
        for i in range(len(mesh.vertices)):
            mesh.vertices[i] = self.align_point(mesh.vertices[i])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        scene.add(mesh)

        # add temp cam
        self.camera = pyrender.IntrinsicsCamera(
            calibration_matrix[0, 0], calibration_matrix[1, 1], calibration_matrix[0, 2], calibration_matrix[1, 2], zfar=10000, name="cam"
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

        # Set up the light -- a single spot light in z+
        light = pyrender.SpotLight(color=255 * np.ones(3), intensity=3000.0, innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=light_pose)

        self.r = pyrender.OffscreenRenderer(720, 1280)
        # add the A flag for the masking
        self.flag = pyrender.constants.RenderFlags.RGBA
        scene.set_pose(self.cam_node, camera_pose)

        # ====== Render the scene
        color, depth = self.r.render(scene, flags=self.flag)
        frame_for_display[color[:, :, 3] != 0] = color[:, :, [2, 1, 0]][color[:, :, 3] != 0]
        return frame_for_display