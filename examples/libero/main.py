# ruff: noqa: UP007 -- Optional annotations must work at runtime in the Python 3.8 simulator.
import collections
import dataclasses
import logging
import math
import pathlib
import sys
import time
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # Opt-in research records. Existing official runner defaults are unchanged.
    episode_log_path: Optional[str] = None
    checkpoint_id: str = "unspecified"
    checkpoint_dir: str = "unspecified"
    config: str = "unspecified"
    group: str = "unspecified"
    training_seed: Optional[int] = None
    reference: bool = False
    num_steps: Optional[int] = None
    paired_noise_seed: Optional[int] = None
    max_tasks: Optional[int] = None  # Diagnostic runs only; primary protocol uses every task.
    save_videos: bool = True


def eval_libero(args: Args) -> None:
    research = None
    if args.episode_log_path is not None or args.paired_noise_seed is not None:
        # The official simulator uses its own Python 3.8 environment. Only this
        # dependency-free subpackage is needed, not the model training environment.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
        from openpi.research import evaluation

        research = evaluation
    if args.episode_log_path is not None and pathlib.Path(args.episode_log_path).exists():
        raise FileExistsError(f"Refusing to append a second run to {args.episode_log_path}")
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.max_tasks is not None:
        if not 1 <= args.max_tasks <= num_tasks_in_suite:
            raise ValueError("max_tasks must be between 1 and the number of suite tasks")
        num_tasks_in_suite = args.max_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    if args.episode_log_path is not None:
        for key, expected in {
            "checkpoint_id": args.checkpoint_id,
            "config_name": args.config,
            "num_steps": args.num_steps,
        }.items():
            if server_metadata.get(key) != expected:
                raise ValueError(f"Server {key}={server_metadata.get(key)!r}, expected {expected!r}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            raise ValueError("Requested more trials than the provided official initial states")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            action_plan = collections.deque()
            t = 0
            control_steps = 0
            done = False
            success = False
            replay_images = []
            latencies = []
            query_noise_seeds = []
            failure_tags = []
            error_message = None
            action_horizon = None
            query_index = 0
            phase = "reset"
            episode_start = time.monotonic()

            try:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
            except Exception as exc:
                failure_tags.append("reset_error")
                error_message = f"{type(exc).__name__}: {exc}"
                logging.exception("Episode reset failed")

            logging.info(f"Starting episode {task_episodes + 1}...")
            while not failure_tags and t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        phase = "stabilization"
                        obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    phase = "preprocessing"
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        if args.paired_noise_seed is not None:
                            noise_seed = research.paired_noise_seed(
                                args.paired_noise_seed, f"{args.task_suite_name}/{task_id}", episode_idx, query_index
                            )
                            element["noise_seed"] = noise_seed
                            query_noise_seeds.append(noise_seed)

                        # Query model to get action
                        phase = "policy_inference"
                        query_start = time.monotonic()
                        response = client.infer(element)
                        timing = {"client_roundtrip_ms": (time.monotonic() - query_start) * 1000}
                        timing.update({k: float(v) for k, v in response.get("policy_timing", {}).items()})
                        latencies.append(timing)
                        query_index += 1
                        action_chunk = response["actions"]
                        phase = "action_validation"
                        if not np.isfinite(action_chunk).all():
                            raise ValueError("Policy returned non-finite actions")
                        if action_horizon is not None and len(action_chunk) != action_horizon:
                            raise ValueError("Policy action horizon changed within an episode")
                        action_horizon = len(action_chunk)
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    phase = "environment_step"
                    obs, _reward, done, _info = env.step(action.tolist())
                    control_steps += 1
                    if done:
                        success = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    failure_tags.append(f"{phase}_error")
                    error_message = f"{type(e).__name__}: {e}"
                    break

            if not success and not failure_tags:
                failure_tags.append("timeout")

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if success else "failure"
            task_segment = task_description.replace(" ", "_")
            if args.save_videos and replay_images:
                try:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                except Exception as exc:
                    failure_tags.append("video_error")
                    error_message = f"{type(exc).__name__}: {exc}"
                    logging.exception("Replay video write failed")

            if args.episode_log_path is not None:
                research.append_episode(
                    args.episode_log_path,
                    {
                        "schema_version": 1,
                        "protocol_id": research.PROTOCOL_ID,
                        "episode_id": research.episode_id(args.task_suite_name, task_id, episode_idx, args.seed),
                        "checkpoint_id": args.checkpoint_id,
                        "checkpoint_dir": args.checkpoint_dir,
                        "config": args.config,
                        "group": args.group,
                        "training_seed": args.training_seed,
                        "reference": args.reference,
                        "num_steps": args.num_steps,
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task_description": str(task_description),
                        "initial_state_id": episode_idx,
                        "rollout_seed": args.seed,
                        "paired_noise_seed": args.paired_noise_seed,
                        "query_noise_seeds": query_noise_seeds,
                        "success": success,
                        "steps": control_steps,
                        "num_steps_wait": args.num_steps_wait,
                        "max_steps": max_steps,
                        "action_horizon": action_horizon,
                        "replan_steps": args.replan_steps,
                        "policy_queries": query_index,
                        "episode_seconds": time.monotonic() - episode_start,
                        "latencies": latencies,
                        "failure_tags": failure_tags,
                        "error_message": error_message,
                        "server_metadata": server_metadata,
                    },
                )
                if any(tag != "timeout" for tag in failure_tags):
                    env.close()
                    raise RuntimeError(f"Research evaluation stopped after logged execution error: {error_message}")

            # Log current results
            logging.info(f"Success: {success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
