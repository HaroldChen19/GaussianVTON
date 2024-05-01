from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as onp
import trimesh
import yourdfpy

import viser

from .. import transforms as tf


class ViserUrdf:
    """Helper for rendering URDFs in Viser."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        urdf_path: Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        mesh_color_override: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=urdf_path.parent
            ),
        )
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name

        # Add coordinate frame for each joint.
        self._joint_frames: List[viser.SceneNodeHandle] = []
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.add_frame(
                    _viser_name_from_frame(
                        self._urdf, joint.child, self._root_node_name
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in urdf.scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = urdf.get_transform(
                link_name, urdf.scene.graph.transforms.parents[link_name]
            )
            name = _viser_name_from_frame(urdf, link_name, root_node_name)

            # Scale the mesh. (this will mutate it)
            mesh.apply_scale(self._scale)
            if mesh_color_override is None:
                target.add_mesh_trimesh(
                    name,
                    mesh,
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3] * scale,
                )
            else:
                target.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=mesh_color_override,
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3] * scale,
                )

    def update_cfg(self, configuration: onp.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        with self._target.atomic():
            for joint, frame_handle in zip(
                self._urdf.joint_map.values(), self._joint_frames
            ):
                assert isinstance(joint, yourdfpy.Joint)
                T_parent_child = self._urdf.get_transform(joint.child, joint.parent)
                frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
                frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(
        self,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for joint_name, joint in zip(
            self._urdf.actuated_joint_names, self._urdf.actuated_joints
        ):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            assert joint.limit is not None
            out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)


def _viser_name_from_frame(
    urdf: yourdfpy.URDF,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != urdf.scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = urdf.scene.graph.transforms.parents[frame_name]
    return root_node_name + "/" + "/".join(frames[::-1])
