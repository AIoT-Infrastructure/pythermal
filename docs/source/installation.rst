Installation
=============

Prerequisites
-------------

Before installing the Python package, you need to set up the thermal camera permissions and native runtime:

.. code-block:: bash

   cd pythermal
   ./setup.sh

This script will:

1. Install required system dependencies (cross-compiler, FFmpeg libraries)
2. Set up USB device permissions for the thermal camera
3. Compile the native thermal recorder (`pythermal-recorder`)

After running `setup.sh`, you may need to:

* Disconnect and reconnect your thermal camera
* Log out and log back in (or restart) for permissions to take effect

Install Python Package
----------------------

Install directly on an ARM Linux device (e.g., Jetson, OrangePi, Raspberry Pi):

.. code-block:: bash

   uv pip install pythermal

Or from source:

.. code-block:: bash

   git clone https://github.com/AIoT-Infrastructure/pythermal.git
   cd pythermal
   uv pip install .

Requirements
------------

* Python ≥ 3.9
* ARM Linux environment (Jetson / OrangePi / Raspberry Pi)
* NumPy, OpenCV (auto-installed via pip)
* Thermal camera connected via USB
* Proper USB permissions (set up via `setup.sh`)

