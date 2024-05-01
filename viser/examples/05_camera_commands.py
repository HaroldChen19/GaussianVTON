# mypy: disable-error-code="assignment"
#
# Asymmetric properties are supported in Pyright, but not yet in mypy.
# - https://github.com/python/mypy/issues/3004
# - https://github.com/python/mypy/pull/11643
"""Camera commands

In addition to reads, camera parameters also support writes. These are synced to the
corresponding client automatically.
"""

import time

import numpy as onp

import viser
import viser.transforms as tf

server = viser.ViserServer()
num_frames = 20


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    """For each client that connects, we create a set of random frames + a click handler for each frame.

    When a frame is clicked, we move the camera to the corresponding frame.
    """

    rng = onp.random.default_rng(0)

    def make_frame(i: int) -> None:
        # Sample a random orientation + position.
        wxyz = rng.normal(size=4)
        wxyz /= onp.linalg.norm(wxyz)
        position = rng.uniform(-3.0, 3.0, size=(3,))

        # Create a coordinate frame and label.
        frame = client.add_frame(f"/frame_{i}", wxyz=wxyz, position=position)
        client.add_label(f"/frame_{i}/label", text=f"Frame {i}")

        # Move the camera when we click a frame.
        @frame.on_click
        def _(_):
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(onp.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(20):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 19.0
                )

                # We can atomically set the orientation and the position of the camera
                # together to prevent jitter that might happen if one was set before the
                # other.
                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                client.flush()  # Optional!
                time.sleep(1.0 / 60.0)

            # Mouse interactions should orbit around the frame origin.
            client.camera.look_at = frame.position

    for i in range(num_frames):
        make_frame(i)


while True:
    time.sleep(1.0)
