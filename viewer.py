import os
import cv2
import torch
import numpy as np

from argparse import ArgumentParser

from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    get_combined_args
)
from scene import Scene
from gaussian_renderer import GaussianModel, render_surfel
from utils.system_utils import searchForMaxIteration


def orbit_camera(
    base_world_view,
    target,
    angle_x,
    angle_y
):
    """
    Generate an orbiting camera using Ref-Gaussian's
    world_view_transform convention.

    Ref-Gaussian stores:

        world_view_transform = (world_to_camera)^T

    Therefore we convert to a conventional camera-to-world
    matrix, modify the camera pose, and convert back.
    """

    # Ref-Gaussian's stored convention:
    #
    # inverse(world_view_transform)
    #     = (camera_to_world)^T
    #
    base_c2w = torch.inverse(
        base_world_view
    ).T

    target = torch.tensor(
        target,
        dtype=torch.float32,
        device=base_c2w.device
    )

    # Original camera position.
    camera_position = base_c2w[:3, 3]

    # Position relative to orbit target.
    offset = camera_position - target

    # ------------------------------------------------------------
    # Horizontal orbit around world Y axis.
    # ------------------------------------------------------------

    cos_x = np.cos(angle_x)
    sin_x = np.sin(angle_x)

    R_z = torch.tensor(
        [
            [cos_x, -sin_x, 0.0],
            [sin_x,  cos_x, 0.0],
            [0.0,    0.0,   1.0],
        ],
        dtype=torch.float32,
        device=base_c2w.device
    )

    # ------------------------------------------------------------
    # Vertical orbit around world X axis.
    # ------------------------------------------------------------

    cos_y = np.cos(angle_y)
    sin_y = np.sin(angle_y)

    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_y, -sin_y],
            [0.0, sin_y, cos_y],
        ],
        dtype=torch.float32,
        device=base_c2w.device
    )

    # ------------------------------------------------------------
    # Rotate camera position around target.
    # ------------------------------------------------------------

    orbit_rotation = R_z @ R_x

    new_offset = orbit_rotation @ offset
    new_camera_position = target + new_offset

    # ------------------------------------------------------------
    # Rotate camera orientation together with the orbit.
    # ------------------------------------------------------------

    new_rotation = (
        orbit_rotation
        @ base_c2w[:3, :3]
    )

    # ------------------------------------------------------------
    # Construct conventional camera-to-world matrix.
    # ------------------------------------------------------------

    new_c2w = torch.eye(
        4,
        dtype=torch.float32,
        device=base_c2w.device
    )

    new_c2w[:3, :3] = new_rotation
    new_c2w[:3, 3] = new_camera_position

    # ------------------------------------------------------------
    # Convert back to Ref-Gaussian convention.
    # ------------------------------------------------------------

    new_world_view = (
        torch.inverse(new_c2w).T
    )

    return new_world_view


def render_camera(
    view,
    gaussians,
    pipeline,
    background,
    op
):
    """
    Render one image using Ref-Gaussian's renderer.
    """

    view.refl_mask = None

    with torch.no_grad():
        rendering = render_surfel(
            view,
            gaussians,
            pipeline,
            background,
            srgb=op.srgb,
            opt=op
        )

    image = torch.clamp(
        rendering["render"],
        0.0,
        1.0
    )

    # CHW -> HWC
    image = image.permute(
        1, 2, 0
    )

    # GPU -> CPU
    image = image.cpu().numpy()

    # [0, 1] -> [0, 255]
    image = (
        image * 255.0
    ).astype(np.uint8)

    # RGB -> BGR for OpenCV
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


def main():

    parser = ArgumentParser(
        description="Interactive Ref-Gaussian viewer"
    )

    model = ModelParams(
        parser,
        sentinel=True
    )

    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    args = get_combined_args(parser)

    # ------------------------------------------------------------
    # Make dataset path absolute.
    # ------------------------------------------------------------

    args.source_path = os.path.abspath(
        args.source_path
    )

    # ------------------------------------------------------------
    # Find latest trained iteration.
    # ------------------------------------------------------------

    iteration = searchForMaxIteration(
        os.path.join(
            args.model_path,
            "point_cloud"
        )
    )

    print(
        f"Loading model iteration {iteration}"
    )

    # ------------------------------------------------------------
    # Load Gaussian model and cameras.
    # ------------------------------------------------------------

    gaussians = GaussianModel(
        args.sh_degree
    )

    scene = Scene(
        args,
        gaussians,
        load_iteration=iteration,
        shuffle=False
    )

    # ------------------------------------------------------------
    # Background.
    # ------------------------------------------------------------

    bg_color = (
        [1, 1, 1]
        if args.white_background
        else [0, 0, 0]
    )

    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device="cuda"
    )

    # ------------------------------------------------------------
    # Starting camera.
    # ------------------------------------------------------------

    base_view = scene.getTestCameras()[0]

    print(
        f"Rendering test camera: "
        f"{base_view.image_width}x"
        f"{base_view.image_height}"
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Save the original camera matrices separately.
    #
    # We must NEVER modify these.
    # ------------------------------------------------------------

    base_world_view = (
        base_view.world_view_transform.clone()
    )

    base_projection = (
        base_view.projection_matrix.clone()
    )

    # ------------------------------------------------------------
    # Orbit target.
    # ------------------------------------------------------------

    target = np.zeros(
        3,
        dtype=np.float32
    )

    angle_x = 0.0
    angle_y = 0.0

    # ------------------------------------------------------------
    # Mouse state.
    # ------------------------------------------------------------

    mouse_down = False

    last_x = 0
    last_y = 0

    camera_changed = True

    window_name = (
        "Ref-Gaussian Interactive Viewer"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    # ------------------------------------------------------------
    # Mouse callback.
    # ------------------------------------------------------------

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):
        nonlocal mouse_down
        nonlocal last_x
        nonlocal last_y
        nonlocal angle_x
        nonlocal angle_y
        nonlocal camera_changed

        if event == cv2.EVENT_LBUTTONDOWN:

            mouse_down = True

            last_x = x
            last_y = y

        elif event == cv2.EVENT_LBUTTONUP:

            mouse_down = False

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and mouse_down
        ):

            dx = x - last_x
            dy = y - last_y

            # Only change camera when the mouse
            # actually moves.
            if dx != 0 or dy != 0:

                angle_x += dx * 0.01
                angle_y += dy * 0.01

                # Prevent flipping upside down.
                angle_y = np.clip(
                    angle_y,
                    -1.4,
                    1.4
                )

                camera_changed = True

            last_x = x
            last_y = y

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print("Controls:")
    print("  Left mouse drag : orbit")
    print("  ESC             : exit")
    print()

    # ------------------------------------------------------------
    # Initial image.
    # ------------------------------------------------------------

    image = None

    # ------------------------------------------------------------
    # Main viewer loop.
    # ------------------------------------------------------------

    while True:

        # --------------------------------------------------------
        # Only render when the camera has changed.
        # --------------------------------------------------------

        if camera_changed:

            world_view = orbit_camera(
                base_world_view,
                target,
                angle_x,
                angle_y
            )

            # Use the existing Camera object for rendering,
            # but always start from the immutable base matrices.
            view = base_view

            view.world_view_transform = (
                world_view
            )

            view.projection_matrix = (
                base_projection
            )

            # Camera center follows Ref-Gaussian's convention.
            view.camera_center = torch.inverse(
                world_view
            )[3, :3]

            # Recompute full projection transform.
            view.full_proj_transform = (
                view.world_view_transform
                .unsqueeze(0)
                .bmm(
                    view.projection_matrix
                    .unsqueeze(0)
                )
                .squeeze(0)
            )

            image = render_camera(
                view,
                gaussians,
                pipeline,
                background,
                op
            )

            camera_changed = False

        # --------------------------------------------------------
        # Display the most recent render.
        # --------------------------------------------------------

        if image is not None:

            cv2.imshow(
                window_name,
                image
            )

        # waitKey is now reached continuously because
        # rendering only occurs after camera changes.
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()