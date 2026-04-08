# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WPI verl plugin — registers the WPI checkpoint engine into verl's registry.

To use this plugin with verl, add the following to your training config:

    checkpoint_engine:
      backend: wpi
      custom_backend_module: wpi_verl_plugin
      engine_kwargs:
        wpi:
          buffer_id: verl-weight-buffer
          socket_dir: /run/wpi/sockets

This module is imported on every Ray worker via the ``custom_backend_module``
hook introduced in verl PR #5718. The import triggers the
``@CheckpointEngineRegistry.register("wpi")`` decorator, making the engine
available to the checkpoint engine framework.
"""

from wpi_verl_plugin.engine import WPICheckpointEngine  # noqa: F401 — triggers registration

__all__ = ["WPICheckpointEngine"]
