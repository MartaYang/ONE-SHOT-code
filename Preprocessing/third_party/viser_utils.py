import torch
import os
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
import cv2
import numpy as np
import matplotlib.cm as cm
import viser
import viser.transforms as tf
import time
import trimesh
import dataclasses
from scipy.spatial.transform import Rotation
from skimage.morphology import binary_dilation, binary_erosion, disk
from dust3r.viz import (
    add_scene_cam,
    CAM_COLORS,
    OPENGL,
    pts3d_to_trimesh,
    cat_meshes,
)


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


def segment_sky(image):
    import cv2
    from scipy import ndimage

    # Convert to HSV
    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255 * image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # add luminous gray
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # keep only largest CC
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask2.view(np.uint8), connectivity=8
    )
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.in1d(labels, selection).reshape(labels.shape)

    # Apply mask
    return torch.from_numpy(mask3)


def convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    show_cam=True,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
    save_name=None,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size,
            )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None:
        save_name = "scene"
    outfile = os.path.join(outdir, save_name + ".glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh):
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:

        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])

        x[np.logical_not(mask)] = vmin

    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    """
    turn a grayscale image into a color image
    :param x: torch.Tensor, grayscale image, [H, W] or [B, H, W]
    :param mask: torch.Tensor or None, mask image, [H, W] or [B, H, W] or None
    """

    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out

def get_color(idx):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    colors_path = os.path.join(root_dir, "models/smpl_colors.txt")
    colors = np.loadtxt(colors_path).astype(int)
    return colors[idx % len(colors)]

class SceneHumanViewer:
    def __init__(
        self,
        pc_list,
        color_list,
        conf_list,
        cam_dict,
        all_smpl_verts,
        smpl_faces,
        smpl_id,
        msk_list,
        gt_cam_dict=None,
        gt_smpl_verts=None,
        image_mask=None,
        edge_color_list=None,
        device="cpu",
        port=8080,
        show_camera=True,
        show_gt_camera=False,
        show_gt_smpl=False,
        vis_threshold=1,
        msk_threshold=0.1,
        mask_morph=0,
        size=512,
        downsample_factor=10,
        smpl_downsample_factor=1,
        camera_downsample_factor=1,
    ):
        self.size=size
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        self.device = device
        self.conf_list = conf_list
        self.msk_list = msk_list
        self.vis_threshold = vis_threshold
        self.msk_threshold = msk_threshold
        self.mask_morph = mask_morph
        self.show_background = True
        self.show_foreground = False
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, msk_list, 
            all_smpl_verts, smpl_faces, smpl_id, edge_color_list, gt_smpl_verts
        )
        # Fast lookup from step id to its sequential index
        self.step_to_index = {step: idx for idx, step in enumerate(self.all_steps)}
        self.cam_dict = cam_dict
        self.gt_cam_dict = gt_cam_dict
        self.gt_smpl_verts = gt_smpl_verts
        self.num_frames = len(self.all_steps)
        self.image_mask = image_mask
        self.show_camera = show_camera
        self.show_gt_camera = show_gt_camera and gt_cam_dict is not None
        self.show_gt_smpl = show_gt_smpl and gt_smpl_verts is not None
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list]
        self.via_points = []
        self._updating_point_clouds = False
        
        # Performance optimization for dynamic opacity
        self._last_opacity_update_step = -1
        self._opacity_update_throttle = 0  # Frames to skip between updates
        self._opacity_frame_counter = 0

        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")
        button5 = self.server.gui.add_button("Hybrid (Current SMPL + All Points)")
        self.is_render = False
        self.fourd = False
        self.hybrid_mode = False

        @button3.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = True
            self.hybrid_mode = False

        @button4.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False
            self.hybrid_mode = False

        @button5.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False
            self.hybrid_mode = True

        self.gui_show_background = self.server.add_gui_checkbox(
            "Show Background", True)

        @self.gui_show_background.on_update
        def _(_) -> None:
            self.show_background = self.gui_show_background.value
            self._update_point_clouds()

        self.gui_show_foreground = self.server.add_gui_checkbox(
            "Show Foreground", False)

        @self.gui_show_foreground.on_update
        def _(_) -> None:
            self.show_foreground = self.gui_show_foreground.value
            self._update_point_clouds()

        self.gui_show_smpl = self.server.add_gui_checkbox("Show SMPL", True)

        @self.gui_show_smpl.on_update
        def _(_) -> None:
            # Update SMPL visibility considering downsampling factor
            if hasattr(self, 'mesh_handles') and hasattr(self, 'frame_nodes'):
                self._update_smpl_visibility()

        self.focal_slider = self.server.add_gui_slider(
            "Focal Length",
            min=0.1,
            max=99999,
            step=1,
            initial_value=533,
        )

        self.psize_slider = self.server.add_gui_slider(
            "Point Size",
            min=0.0001,
            max=0.1,
            step=0.0001,
            initial_value=0.005,
        )
        self.camsize_slider = self.server.add_gui_slider(
            "Camera Size",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.1,
        )

        self.downsample_slider = self.server.add_gui_slider(
            "Downsample Factor",
            min=1,
            max=1000,
            step=1,
            initial_value=downsample_factor,
        )
        self.vis_threshold_slider = self.server.add_gui_slider(
            "Visibility Threshold",
            min=0.1,
            max=50.0,
            step=0.1,
            initial_value=self.vis_threshold,
        )
        
        self.mask_morph_slider = self.server.add_gui_slider(
            "Mask Morphology",
            min=-20.0,
            max=100.0,
            step=0.1,
            initial_value=self.mask_morph,
        )

        # SMPL downsampling controls
        self.smpl_downsample_slider = self.server.add_gui_slider(
            "SMPL Downsample",
            min=1,
            max=200,
            step=1,
            initial_value=smpl_downsample_factor,
        )
        
        # Camera downsampling controls
        self.camera_downsample_slider = self.server.add_gui_slider(
            "Camera Downsample",
            min=1,
            max=200,
            step=1,
            initial_value=camera_downsample_factor,
        )
        
        @self.camera_downsample_slider.on_update
        def _(_) -> None:
            # Apply camera downsampling changes immediately
            if hasattr(self, 'cam_handles'):
                self._update_camera_visibility()
            if hasattr(self, 'gt_cam_handles'):
                self._update_gt_camera_visibility()

        # Mesh opacity by time controls
        self.mesh_time_opacity_checkbox = self.server.add_gui_checkbox(
            "Mesh Opacity by Time", False
        )
        self.min_mesh_opacity_slider = self.server.add_gui_slider(
            "Min Mesh Opacity",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=0.1,
        )

        @self.mesh_time_opacity_checkbox.on_update
        def _(_) -> None:
            # Apply opacity changes to existing meshes immediately
            if hasattr(self, 'mesh_handles'):
                # If dynamic opacity is enabled, prefer dynamic update
                if self.dynamic_opacity_checkbox.value and hasattr(self, 'current_step_index'):
                    self._update_dynamic_opacities(self.current_step_index, force_update=True)
                else:
                    self._update_mesh_opacities()

        @self.min_mesh_opacity_slider.on_update
        def _(_) -> None:
            # Apply opacity changes to existing meshes immediately
            if hasattr(self, 'mesh_handles'):
                # If dynamic opacity is enabled, prefer dynamic update so min opacity affects decay baseline
                if self.dynamic_opacity_checkbox.value and hasattr(self, 'current_step_index'):
                    self._update_dynamic_opacities(self.current_step_index, force_update=True)
                else:
                    self._update_mesh_opacities()

        # Dynamic opacity controls
        self.dynamic_opacity_checkbox = self.server.add_gui_checkbox(
            "Dynamic Opacity", False
        )
        self.opacity_decay_len_slider = self.server.add_gui_slider(
            "Decay Length",
            min=1,
            max=max(10, len(self.all_steps)),
            step=1,
            initial_value=min(10, max(1, len(self.all_steps))),
        )
        
        # Performance control for dynamic opacity
        self.opacity_update_throttle_slider = self.server.add_gui_slider(
            "Opacity Update Throttle",
            min=0,
            max=10,
            step=1,
            initial_value=0,
            hint="Skip N frames between opacity updates (0=update every frame, higher=better performance)"
        )
        
        # Performance preset buttons
        self.performance_preset_buttons = self.server.add_gui_button_group(
            "Performance Presets", 
            ("High Quality", "Balanced")
        )

        @self.dynamic_opacity_checkbox.on_update
        def _(_) -> None:
            if hasattr(self, 'mesh_handles') and hasattr(self, 'current_step_index'):
                if self.dynamic_opacity_checkbox.value:
                    # Turning on dynamic opacity: update immediately
                    self._update_dynamic_opacities(self.current_step_index, force_update=True)
                else:
                    # Turning off dynamic opacity: fall back to static-by-time if enabled,
                    # otherwise set all to fully opaque
                    if self.mesh_time_opacity_checkbox.value:
                        self._update_mesh_opacities()
                    else:
                        for handle in self.mesh_handles:
                            handle.opacity = 1.0
                        for handle in self.gt_mesh_handles:
                            handle.opacity = 1.0

        @self.opacity_decay_len_slider.on_update
        def _(_) -> None:
            if hasattr(self, 'mesh_handles') and hasattr(self, 'current_step_index'):
                self._update_dynamic_opacities(self.current_step_index, force_update=True)
                
        @self.opacity_update_throttle_slider.on_update
        def _(_) -> None:
            self._opacity_update_throttle = int(self.opacity_update_throttle_slider.value)
            self._opacity_frame_counter = 0  # Reset counter when throttle changes
            # Force update to apply changes immediately
            if hasattr(self, 'current_step_index') and self.dynamic_opacity_checkbox.value:
                self._update_dynamic_opacities(self.current_step_index, force_update=True)
        
        @self.performance_preset_buttons.on_click
        def _(_) -> None:
            preset = self.performance_preset_buttons.value
            if preset == "High Quality":
                # Best visual quality, may be slower
                self.opacity_update_throttle_slider.value = 0
                self._opacity_update_throttle = 0
            elif preset == "Balanced":
                # Good balance of quality and performance
                self.opacity_update_throttle_slider.value = 2
                self._opacity_update_throttle = 2
                
            self._opacity_frame_counter = 0
            # Apply preset immediately
            if hasattr(self, 'current_step_index') and self.dynamic_opacity_checkbox.value:
                self._update_dynamic_opacities(self.current_step_index, force_update=True)

        self.show_camera_checkbox = self.server.add_gui_checkbox(
            "Show Camera", 
            initial_value=self.show_camera
        )

        self.show_gt_camera_checkbox = self.server.add_gui_checkbox(
            "Show GT Camera", 
            initial_value=self.show_gt_camera
        )

        self.show_gt_smpl_checkbox = self.server.add_gui_checkbox(
            "Show GT SMPL", 
            initial_value=self.show_gt_smpl
        )
   
        self.pc_handles = []
        self.cam_handles = []
        self.gt_cam_handles = []
        self.mesh_handles = []
        self.mesh_step_mapping = []
        self.gt_mesh_handles = []
        self.gt_mesh_step_mapping = []

        @self.psize_slider.on_update
        def _(_) -> None:
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_) -> None:
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale
            for handle in self.gt_cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale

        @self.downsample_slider.on_update
        def _(_) -> None:
            # when downsampling factor changes, regenerate all point clouds
            if hasattr(self, 'frame_nodes'):
                self._update_point_clouds()

        @self.show_camera_checkbox.on_update
        def _(_) -> None:
            # update internal state
            self.show_camera = self.show_camera_checkbox.value
            
            if self.show_camera:
                # if camera display is enabled, ensure all cameras are visible with downsampling
                self._update_camera_visibility()
                    
                # check if any cameras are missing
                if hasattr(self, 'frame_nodes') and len(self.cam_handles) < len(self.frame_nodes):
                    for i in range(len(self.cam_handles), len(self.frame_nodes)):
                        if i < len(self.all_steps):
                            step = self.all_steps[i]
                            self.add_camera(step)
                    # Apply downsampling to newly added cameras
                    self._update_camera_visibility()
            else:
                # if camera display is disabled, hide all cameras
                for handle in self.cam_handles:
                    handle.visible = False

        @self.show_gt_camera_checkbox.on_update
        def _(_) -> None:
            # update internal state
            self.show_gt_camera = self.show_gt_camera_checkbox.value
            
            if self.show_gt_camera and self.gt_cam_dict is not None:
                # if GT camera display is enabled, ensure all GT cameras are visible with downsampling
                self._update_gt_camera_visibility()
                    
                # check if any GT cameras are missing
                if hasattr(self, 'frame_nodes') and len(self.gt_cam_handles) < len(self.frame_nodes):
                    for i in range(len(self.gt_cam_handles), len(self.frame_nodes)):
                        if i < len(self.all_steps):
                            step = self.all_steps[i]
                            self.add_gt_camera(step)
                    # Apply downsampling to newly added GT cameras
                    self._update_gt_camera_visibility()
            else:
                # if GT camera display is disabled, hide all GT cameras
                for handle in self.gt_cam_handles:
                    handle.visible = False

        @self.show_gt_smpl_checkbox.on_update
        def _(_) -> None:
            # update internal state
            self.show_gt_smpl = self.show_gt_smpl_checkbox.value
            
            if self.show_gt_smpl and self.gt_smpl_verts is not None:
                # if GT SMPL display is enabled, ensure all GT meshes are visible
                for handle in self.gt_mesh_handles:
                    handle.visible = True
                    
                # check if any GT meshes are missing
                if hasattr(self, 'frame_nodes') and len(self.gt_mesh_handles) < len(self.frame_nodes):
                    for i in range(len(self.gt_mesh_handles), len(self.frame_nodes)):
                        if i < len(self.all_steps):
                            step = self.all_steps[i]
                            self.add_gt_smpl(step)
            else:
                # if GT SMPL display is disabled, hide all GT meshes
                for handle in self.gt_mesh_handles:
                    handle.visible = False

        @self.vis_threshold_slider.on_update
        def _(_) -> None:
            # when visibility threshold changes, update threshold and regenerate point clouds
            self.vis_threshold = self.vis_threshold_slider.value
            if hasattr(self, 'frame_nodes'):
                self._update_point_clouds()
        
        @self.mask_morph_slider.on_update
        def _(_) -> None:
            # when mask morphology changes, update morphology and regenerate point clouds
            self.mask_morph = self.mask_morph_slider.value
            if hasattr(self, 'frame_nodes'):
                self._update_point_clouds()

        @self.smpl_downsample_slider.on_update
        def _(_) -> None:
            # when SMPL downsampling factor changes, update SMPL visibility
            if hasattr(self, 'mesh_handles') and hasattr(self, 'frame_nodes'):
                self._update_smpl_visibility()

        self.server.on_client_connect(self._connect_client)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    @staticmethod
    def generate_pseudo_intrinsics(h, w):
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(
            np.float32
        )

    def get_ray_map(self, c2w, h, w, intrinsics=None):
        if intrinsics is None:
            intrinsics = self.generate_pseudo_intrinsics(h, w)
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        ro = np.broadcast_to(ro, (h, w, 3))
        ray_map = np.concatenate([ro, rd], axis=-1)
        return ray_map

    def set_camera_loc(camera, pose, K):
        """
        pose: 4x4 matrix
        K: 3x3 matrix
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        aspect = float(cx) / float(cy)
        fov = 2 * np.arctan(2 * cx / fx)
        wxyz_xyz = tf.SE3.from_matrix(pose).wxyz_xyz
        wxyz = wxyz_xyz[:4]
        xyz = wxyz_xyz[4:]
        camera.wxyz = wxyz
        camera.position = xyz
        camera.fov = fov
        camera.aspect = aspect

    def _connect_client(self, client: viser.ClientHandle):
        from dust3r.inference import inference_step
        from dust3r.utils.geometry import geotrf

        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")
        fov_panel = client.gui.add_text(
            "fov:", f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
        )
        aspect_panel = client.gui.add_text("aspect:", "1.0")

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self.server.atomic():
                wxyz_panel.value = f"{client.camera.wxyz}"
                position_panel.value = f"{client.camera.position}"
                fov_panel.value = (
                    f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
                )
                aspect_panel.value = "1.0"


    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):

        image[:border_width, :, 0] = color[0]  # Red channel
        image[:border_width, :, 1] = color[1]  # Green channel
        image[:border_width, :, 2] = color[2]  # Blue channel
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]

        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]

        return image

    def read_data(
            self, 
            pc_list, 
            color_list, 
            conf_list,
            msk_list,
            all_smpl_verts, 
            smpl_faces, 
            smpl_id,
            edge_color_list=None,
            gt_smpl_verts=None):
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update(
                {
                    step: {
                        "pc": pc,
                        "color": color_list[i],
                        "conf": conf_list[i],
                        "msk": msk_list[i],
                        "verts": all_smpl_verts[i],
                        "faces": smpl_faces,
                        "smpl_id": smpl_id[i],
                        "edge_color": (
                            None if edge_color_list[i] is None else edge_color_list[i]
                        ),
                        "gt_verts": (
                            None if gt_smpl_verts is None else gt_smpl_verts[i]
                        ),
                    }
                }
            )
            step_list.append(step)
        normalized_indices = (
            np.array(list(range(len(pc_list))))
            / np.array(list(range(len(pc_list)))).max()
        )
        cmap = cm.viridis
        self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        msk=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
        downsample_factor=1,
    ):

        pred_pts = pc.reshape(-1, 3)  # [N, 3]

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)

        if msk is not None:
            msk_2d = msk[0].copy()
            fg_mask = msk_2d > self.msk_threshold
            
            if abs(self.mask_morph) > 0:
                if self.mask_morph > 0:
                    fg_mask = binary_dilation(fg_mask, disk(abs(self.mask_morph)))
                else:
                    fg_mask = binary_erosion(fg_mask, disk(abs(self.mask_morph)))
            
            fg_mask = fg_mask.reshape(-1)
            
            if self.show_foreground and self.show_background:
                display_mask = np.ones_like(fg_mask, dtype=bool)
            elif self.show_foreground and not self.show_background:
                display_mask = fg_mask
            elif not self.show_foreground and self.show_background:
                display_mask = ~fg_mask
            else:
                return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
            
            if conf is not None:
                conf = conf[0].reshape(-1)
                final_mask = display_mask & (conf > self.vis_threshold)
            else:
                final_mask = display_mask
                
            pred_pts = pred_pts[final_mask]
            color = color[final_mask]
            
        elif conf is not None:
            conf = conf[0].reshape(-1)
            pred_pts = pred_pts[conf > self.vis_threshold]
            color = color[conf > self.vis_threshold]

        # apply downsampling
        if downsample_factor > 1 and len(pred_pts) > 0:
            indices = np.arange(0, len(pred_pts), downsample_factor)
            pred_pts = pred_pts[indices]
            color = color[indices]

        return pred_pts, color

    def _update_point_clouds(self):
        # prevent duplicate update
        if self._updating_point_clouds:
            return
        
        self._updating_point_clouds = True
        try:
            # safely clear existing point cloud display
            for handle in self.pc_handles:
                try:
                    handle.remove()
                except (KeyError, AttributeError):
                    # ignore already deleted or non-existent handles
                    pass
            self.pc_handles.clear()
            
            for i, step in enumerate(self.all_steps):
                if hasattr(self, 'frame_nodes') and i < len(self.frame_nodes):
                    self._add_pc_for_step(step)
        finally:
            self._updating_point_clouds = False

    def _add_pc_for_step(self, step):
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        msk = self.pcs[step]["msk"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, msk, edge_color, set_border_color=True,
            downsample_factor=self.downsample_slider.value
        )

        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=0.005,
            )
        )

    def _compute_opacity_for_index(self, step_index):
        """Compute opacity so that earlier frames are more transparent, later frames more opaque.
        - step_index = 0   => opacity close to min_opacity (most transparent)
        - step_index = N-1 => opacity close to 1.0 (most opaque)
        """
        if not self.mesh_time_opacity_checkbox.value or self.num_frames <= 1:
            return None
        ratio = step_index / (self.num_frames - 1)
        min_opacity = float(self.min_mesh_opacity_slider.value)
        return min_opacity + ratio * (1.0 - min_opacity)

    def _update_mesh_opacities(self):
        # If dynamic opacity is enabled, avoid overriding dynamic values
        if self.dynamic_opacity_checkbox.value:
            return
        # Update predicted meshes
        for mesh_idx, handle in enumerate(self.mesh_handles):
            if mesh_idx < len(self.mesh_step_mapping):
                step = self.mesh_step_mapping[mesh_idx]
                step_index = self.step_to_index.get(step, 0)
                handle.opacity = self._compute_opacity_for_index(step_index)
        # Update GT meshes
        for mesh_idx, handle in enumerate(self.gt_mesh_handles):
            if mesh_idx < len(self.gt_mesh_step_mapping):
                step = self.gt_mesh_step_mapping[mesh_idx]
                step_index = self.step_to_index.get(step, 0)
                handle.opacity = self._compute_opacity_for_index(step_index)

    def _compute_dynamic_opacity(self, age: int):
        """Compute dynamic opacity by age (0 for current frame).
        Linear decay to min opacity over decay_len frames.
        """
        min_opacity = float(self.min_mesh_opacity_slider.value)
        decay_len = max(1, int(self.opacity_decay_len_slider.value))
        if age <= 0:
            return 1.0
        ratio = min(1.0, age / decay_len)
        return 1.0 - ratio * (1.0 - min_opacity)

    def _update_dynamic_opacities(self, current_step_index: int, force_update: bool = False):
        if not self.dynamic_opacity_checkbox.value:
            return
            
        # Performance optimization: skip updates based on throttle setting
        if not force_update and hasattr(self, '_opacity_update_throttle'):
            if self._last_opacity_update_step == current_step_index:
                return  # Already updated for this step
                
            if self._opacity_update_throttle > 0:
                if self._opacity_frame_counter < self._opacity_update_throttle:
                    self._opacity_frame_counter += 1
                    return
                else:
                    self._opacity_frame_counter = 0  # Reset counter
        
        self._last_opacity_update_step = current_step_index
        
        # predicted meshes
        for mesh_idx, handle in enumerate(self.mesh_handles):
            if mesh_idx < len(self.mesh_step_mapping):
                step = self.mesh_step_mapping[mesh_idx]
                idx = self.step_to_index.get(step, 0)
                age = current_step_index - idx
                if age < 0:
                    # future frames (if visible) keep opaque
                    handle.opacity = 1.0
                else:
                    handle.opacity = self._compute_dynamic_opacity(age)
        # GT meshes
        for mesh_idx, handle in enumerate(self.gt_mesh_handles):
            if mesh_idx < len(self.gt_mesh_step_mapping):
                step = self.gt_mesh_step_mapping[mesh_idx]
                idx = self.step_to_index.get(step, 0)
                age = current_step_index - idx
                if age < 0:
                    handle.opacity = 1.0
                else:
                    handle.opacity = self._compute_dynamic_opacity(age)

    def _update_smpl_visibility(self):
        """Update SMPL mesh visibility based on downsampling factor"""
        downsample_factor = int(self.smpl_downsample_slider.value)
        
        # Update predicted meshes
        for mesh_idx, handle in enumerate(self.mesh_handles):
            if mesh_idx < len(self.mesh_step_mapping):
                step = self.mesh_step_mapping[mesh_idx]
                step_index = self.step_to_index.get(step, 0)
                # Only show meshes at intervals defined by downsample_factor
                should_show = (step_index % downsample_factor == 0)
                # Only hide if SMPL is enabled and this mesh should be hidden
                if self.gui_show_smpl.value:
                    handle.visible = should_show
                else:
                    handle.visible = False
        
        # Update GT meshes
        for mesh_idx, handle in enumerate(self.gt_mesh_handles):
            if mesh_idx < len(self.gt_mesh_step_mapping):
                step = self.gt_mesh_step_mapping[mesh_idx]
                step_index = self.step_to_index.get(step, 0)
                # Only show meshes at intervals defined by downsample_factor
                should_show = (step_index % downsample_factor == 0)
                # Only hide if GT SMPL is enabled and this mesh should be hidden
                if self.show_gt_smpl:
                    handle.visible = should_show
                else:
                    handle.visible = False

    def add_pc(self, step):
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        msk = self.pcs[step]["msk"]
        verts = self.pcs[step]["verts"]
        faces = self.pcs[step]["faces"]
        smpl_id = self.pcs[step]["smpl_id"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, msk, edge_color, set_border_color=True,
            downsample_factor=self.downsample_slider.value
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=0.005,
            )
        )
        if len(verts) > 0:
            for tid, vert in enumerate(verts):
                step_idx = self.step_to_index.get(step, 0)
                mesh_opacity = self._compute_opacity_for_index(step_idx)
                mesh_handle = self.server.scene.add_mesh_simple(
                    name=f"/frames/{step}/human_{tid}",
                    vertices=vert,
                    faces=faces,
                    flat_shading=False,
                    wireframe=False,
                    opacity=mesh_opacity,
                    color=get_color(smpl_id[tid]),
                )
                self.mesh_handles.append(mesh_handle)
                self.mesh_step_mapping.append(step)
        
        # Add GT SMPL mesh if enabled
        if self.show_gt_smpl:
            self.add_gt_smpl(step)

    def add_camera(self, step):
        cam = self.cam_dict
        focal = cam["focal"][step]
        pp = cam["pp"][step]
        R = cam["R"][step]
        t = cam["t"][step]

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))
        self.cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=0.1,
                color=tuple(self.camera_colors[step][:3]),
            )
        )

    def add_gt_camera(self, step):
        if self.gt_cam_dict is None:
            return
            
        cam = self.gt_cam_dict
        focal = cam["focal"][step]
        pp = cam["pp"][step]
        R = cam["R"][step]
        t = cam["t"][step]

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.gt_cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/gt_camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=0.1,
                color=(166, 166, 166),
            )
        )
    
    def _update_camera_visibility(self):
        """Update camera visibility based on downsample factor"""
        if not hasattr(self, 'cam_handles') or not self.show_camera:
            return
            
        camera_downsample_factor = int(self.camera_downsample_slider.value)
        
        for i, handle in enumerate(self.cam_handles):
            if i < len(self.all_steps):
                step_index = self.step_to_index.get(self.all_steps[i], i)
                should_show = (step_index % camera_downsample_factor == 0)
                handle.visible = should_show
    
    def _update_gt_camera_visibility(self):
        """Update GT camera visibility based on downsample factor"""
        if not hasattr(self, 'gt_cam_handles') or not self.show_gt_camera:
            return
            
        camera_downsample_factor = int(self.camera_downsample_slider.value)
        
        for i, handle in enumerate(self.gt_cam_handles):
            if i < len(self.all_steps):
                step_index = self.step_to_index.get(self.all_steps[i], i)
                should_show = (step_index % camera_downsample_factor == 0)
                handle.visible = should_show

    def add_gt_smpl(self, step):
        if self.gt_smpl_verts is None:
            return
            
        gt_verts = self.pcs[step]["gt_verts"]
        faces = self.pcs[step]["faces"]
        smpl_id = 51
        
        if gt_verts is not None and len(gt_verts) > 0:
            for tid, vert in enumerate(gt_verts):
                step_idx = self.step_to_index.get(step, 0)
                mesh_opacity = self._compute_opacity_for_index(step_idx)
                mesh_handle = self.server.scene.add_mesh_simple(
                    name=f"/frames/{step}/gt_human_{tid}",
                    vertices=vert,
                    faces=faces,
                    flat_shading=False,
                    wireframe=False,
                    opacity=mesh_opacity,
                    color=(100, 100, 100),
                )
                self.gt_mesh_handles.append(mesh_handle)
                self.gt_mesh_step_mapping.append(step)

    def animate(self):
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Train Step",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=False,
            )
            gui_next_frame = self.server.add_gui_button("Next Step", disabled=False)
            gui_prev_frame = self.server.add_gui_button("Prev Step", disabled=False)
            gui_playing = self.server.add_gui_checkbox("Playing", False)
            gui_framerate = self.server.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=1
            )
            gui_framerate_options = self.server.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )
            
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = gui_timestep.value
        self.current_step_index = prev_timestep

        @gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            with self.server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
            prev_timestep = current_timestep
            self.current_step_index = current_timestep
            # dynamic opacity update on step change
            if self.dynamic_opacity_checkbox.value:
                self._update_dynamic_opacities(current_timestep)
            self.server.flush()  # Optional!

        self.server.add_frame(
            "/frames",
            show_axes=False,
        )
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.add_frame(
                    f"/frames/{step}",
                    show_axes=False,
                )
            )
            self.add_pc(step)
            if self.show_camera:
                self.add_camera(step)
            if self.show_gt_camera:
                self.add_gt_camera(step)

        prev_timestep = gui_timestep.value
        while True:
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

            current_step = gui_timestep.value
            self.current_step_index = current_step
            
            for i, frame_node in enumerate(self.frame_nodes):
                if self.hybrid_mode:
                    frame_node.visible = i <= current_step
                else:
                    frame_node.visible = i <= current_step if not self.fourd else i == current_step
            
            # When playing, continuously update dynamic opacities (with throttling for performance)
            if self.dynamic_opacity_checkbox.value:
                self._update_dynamic_opacities(current_step)

            show_mesh = self.gui_show_smpl.value
            show_gt_mesh = self.show_gt_smpl
            downsample_factor = int(self.smpl_downsample_slider.value)
            
            if self.hybrid_mode:
                for mesh_idx, mesh_handle in enumerate(self.mesh_handles):
                    if mesh_idx < len(self.mesh_step_mapping):
                        mesh_step = self.mesh_step_mapping[mesh_idx]
                        step_index = self.step_to_index.get(mesh_step, 0)
                        should_show_by_downsample = (step_index % downsample_factor == 0)
                        mesh_handle.visible = (mesh_step == current_step) and show_mesh and should_show_by_downsample
                for mesh_idx, gt_mesh_handle in enumerate(self.gt_mesh_handles):
                    if mesh_idx < len(self.gt_mesh_step_mapping):
                        gt_mesh_step = self.gt_mesh_step_mapping[mesh_idx]
                        step_index = self.step_to_index.get(gt_mesh_step, 0)
                        should_show_by_downsample = (step_index % downsample_factor == 0)
                        gt_mesh_handle.visible = (gt_mesh_step == current_step) and show_gt_mesh and should_show_by_downsample
            else:
                for mesh_idx, mesh_handle in enumerate(self.mesh_handles):
                    if mesh_idx < len(self.mesh_step_mapping):
                        step = self.mesh_step_mapping[mesh_idx]
                        step_index = self.step_to_index.get(step, 0)
                        should_show_by_downsample = (step_index % downsample_factor == 0)
                        mesh_handle.visible = show_mesh and should_show_by_downsample
                for mesh_idx, gt_mesh_handle in enumerate(self.gt_mesh_handles):
                    if mesh_idx < len(self.gt_mesh_step_mapping):
                        step = self.gt_mesh_step_mapping[mesh_idx]
                        step_index = self.step_to_index.get(step, 0)
                        should_show_by_downsample = (step_index % downsample_factor == 0)
                        gt_mesh_handle.visible = show_gt_mesh and should_show_by_downsample

            time.sleep(1.0 / gui_framerate.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)
