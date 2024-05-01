# viser

|mypy| |nbsp| |pyright| |nbsp| |typescript| |nbsp| |versions|

`viser` is a library for interactive 3D visualization + Python, inspired by
tools like [Pangolin](https://github.com/stevenlovegrove/Pangolin),
[rviz](https://wiki.ros.org/rviz/), and
[meshcat](https://github.com/rdeits/meshcat).

As a standalone visualization tool, `viser` features include:

- Web interface for easy use on remote machines.
- Python API for sending 3D primitives to the browser.
- Python-configurable inputs: buttons, checkboxes, text inputs, sliders,
  dropdowns, gizmos.
- Support for multiple panels and view-synchronized connections.

The `viser.infra` backend can also be used to build custom web applications
(example:
[the Nerfstudio viewer](https://github.com/nerfstudio-project/nerfstudio)). It
supports:

- Websocket / HTTP server management, on a shared port.
- Asynchronous server/client communication infrastructure.
- Client state persistence logic.
- Typed serialization; synchronization between Python dataclass and TypeScript
  interfaces.

## Running examples

```bash
# Clone the repository.
git clone https://github.com/nerfstudio-project/viser.git

# Install the package.
# You can also install via pip: `pip install viser`.
cd ./viser
pip install -e .

# Run an example.
pip install -e .[examples]
python ./examples/4_gui.py
```

After an example script is running, you can connect by navigating to the printed
URL (default: `http://localhost:8080`).

<!-- prettier-ignore-start -->

.. toctree::
   :caption: Notes
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./conventions.md
   ./development.md

.. toctree::
   :caption: API (Core)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./server.md
   ./client_handles.md
   ./gui_handles.md
   ./scene_handles.md
   ./events.md
   ./icons.md


.. toctree::
   :caption: API (Additional)
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./infrastructure.md
   ./transforms.md

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 1
   :titlesonly:
   :glob:

   examples/*


.. |build| image:: https://github.com/nerfstudio-project/viser/workflows/build/badge.svg
   :alt: Build status icon
   :target: https://github.com/nerfstudio-project/viser
.. |mypy| image:: https://github.com/nerfstudio-project/viser/workflows/mypy/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |pyright| image:: https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |typescript| image:: https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg
   :alt: TypeScript status icon
   :target: https://github.com/nerfstudio-project/viser
.. |versions| image:: https://img.shields.io/pypi/pyversions/viser
   :alt: Version icon
   :target: https://pypi.org/project/viser/
.. |nbsp| unicode:: 0xA0
   :trim:

<!-- prettier-ignore-end -->
