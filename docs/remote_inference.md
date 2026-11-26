
# Remote policy serving

This page documents the general OpenPI policy server retained in Scalar-Pi. It lets the policy run on a separate machine from the robot. For the study's model settings and paired LIBERO evaluation, follow [Reproduction](reproduction.md).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=LIBERO
```

The `env` argument selects a default released checkpoint. Supported values are `DROID`, `ALOHA`, `ALOHA_SIM`, and `LIBERO`; `LIBERO` loads `pi05_libero`. You can also select a configuration and checkpoint explicitly, including a checkpoint you trained. For example, the DROID default is equivalent to:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

This starts a policy server for the selected configuration and checkpoint. It binds to `127.0.0.1:8000` by default. A remote client can connect through an SSH tunnel; `--host` and `--port` set a different listening address.

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).
