# WPI verl Plugin

This is the standalone Python package that natively integrates the **Weight Propagation Interface (WPI)** into the [verl](https://github.com/verl-project/verl) distributed reinforcement learning framework.

By using this plugin, `verl` can leverage WPI to perform high-throughput, zero-copy weight propagation between the actor/trainer processes and the rollout replicas using direct VRAM memory mapping and NVIDIA IPC.

## Installation

To install the plugin in your local environment, run the following from within this directory:

```bash
pip install -e .
```

### Running on a Distributed Ray Cluster

If you submit distributed jobs via the Ray cluster, you don't need to manually install this plugin onto every node! Simply configure Ray's `runtime_env` to automatically distribute the codebase:

Add the following to your `runtime-env.yaml`:

```yaml
py_modules:
  - "../weight-propagation-interface/consumer/wpi_verl_plugin"  # Adjust the path relative to where you run ray job submit
```

*(Note: In distributed scenarios, Ray workers rely on this `py_modules` property to automatically zip and load the plugin namespace so that the `CheckpointEngineRegistry` can dynamically discover it).*

## Configuration / Usage

After ensuring the package is accessible to the Python environment running your script, update your `verl` configuration (or OmegaConf overrides) to load and register the plugin.

Add the following flags to your execution command or config file:

```bash
# 1. Instruct verl to load the wpi backend natively
actor_rollout_ref.rollout.checkpoint_engine.backend=wpi 

# 2. Instruct verl to dynamically import the backend package (triggers the registry hook)
actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=wpi_verl_plugin 

# 3. Supply any specific WPI engine arguments
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.buffer_id=verl-weight-buffer 
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.socket_dir=/run/wpi/sockets
```

## Logging and Debugging

To configure verbose standard logging for the WPI plugin (which will print detailed information about gRPC requests, SCM_RIGHTS FD transfers, and mapped memory locations), you can set the `VERL_LOGGING_LEVEL` environment variable.

If running on a Ray cluster, add the environment variable to your `runtime-env.yaml`:

```yaml
env_vars:
  VERL_LOGGING_LEVEL: "DEBUG"  # or "INFO"
```
