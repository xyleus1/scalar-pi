# ruff: noqa: UP006, UP007, UP035 -- Runtime annotations must support the Python 3.8 simulator.
import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import List, Optional

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
    require_provenance: bool = False
    attempt_id: str = "initial"
    task_shards: bool = False
    task_ids: Optional[List[int]] = None
    task_attempts: str = "{}"  # JSON mapping original task IDs to explicit attempt names.
    deadline_unix: Optional[float] = None
    task_budget_seconds: Optional[float] = None
    cleanup_seconds: float = 600.0
    metadata_timeout_seconds: float = 60.0
    inference_timeout_seconds: float = 120.0
    cold_inference_timeout_seconds: float = 900.0
    calibration_output: Optional[str] = None


class InvalidPolicyActionsError(ValueError):
    """A completed policy query returned unusable actions; this is a policy failure."""


def eval_libero(args: Args) -> None:
    research = None
    if args.episode_log_path is not None or args.paired_noise_seed is not None or args.calibration_output is not None:
        # The official simulator uses its own Python 3.8 environment. Only this
        # dependency-free subpackage is needed, not the model training environment.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
        from openpi.research import evaluation

        research = evaluation
    if args.calibration_output is not None:
        calibrate_scenes(args, research)
        return
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
    task_ids = list(range(num_tasks_in_suite)) if args.task_ids is None else args.task_ids
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(not 0 <= task < num_tasks_in_suite for task in task_ids)
    ):
        raise ValueError("task_ids must contain unique original suite task IDs")
    attempts = json.loads(args.task_attempts)
    if not isinstance(attempts, dict):
        raise ValueError("task_attempts must be a JSON object")
    if args.task_shards and args.episode_log_path is None:
        raise ValueError("Task shards require the canonical suite log path")
    deadline = (
        args.deadline_unix
        if args.deadline_unix is not None
        else float(os.environ.get("SCALAR_PI_DEADLINE_UNIX") or "inf")
    )
    if math.isnan(deadline) or deadline <= 0:
        raise ValueError("Invalid absolute rollout deadline")
    if not math.isfinite(args.cleanup_seconds) or args.cleanup_seconds < 600:
        raise ValueError("Reserve at least 600 seconds for cleanup")
    if args.task_budget_seconds is not None and (
        not math.isfinite(args.task_budget_seconds) or args.task_budget_seconds <= 0
    ):
        raise ValueError("Task time budget must be finite and positive")
    if math.isfinite(deadline) and args.task_budget_seconds is None:
        raise ValueError("Deadline-bound rollouts require a calibrated whole-task budget")

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

    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        metadata_timeout=args.metadata_timeout_seconds,
        inference_timeout=args.inference_timeout_seconds,
        cold_inference_timeout=args.cold_inference_timeout_seconds,
    )
    server_metadata = client.get_server_metadata()
    if args.episode_log_path is not None:
        for key, expected in {
            "checkpoint_id": args.checkpoint_id,
            "config_name": args.config,
            "num_steps": args.num_steps,
        }.items():
            if server_metadata.get(key) != expected:
                raise ValueError(f"Server {key}={server_metadata.get(key)!r}, expected {expected!r}")

    provenance_sha256 = None
    provenance = None
    if args.require_provenance:
        if args.episode_log_path is None:
            raise ValueError("Provenance requires an episode log")
        task_assets = []
        for task_id in range(num_tasks_in_suite):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            task_assets.append(
                {
                    "task_id": task_id,
                    "description": str(task.language),
                    "bddl_sha256": research.file_digest(bddl),
                    "initial_states_sha256": [
                        hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
                        for state in initial_states[: args.num_trials_per_task]
                    ],
                }
            )
        settings = {
            "suite": args.task_suite_name,
            "num_steps": args.num_steps,
            "rollout_seed": args.seed,
            "paired_noise_seed": args.paired_noise_seed,
            "replan_steps": args.replan_steps,
            "num_steps_wait": args.num_steps_wait,
            "render_size": LIBERO_ENV_RESOLUTION,
            "resize_size": args.resize_size,
            "max_steps": max_steps,
            "attempt_id": args.attempt_id,
        }
        provenance = research.suite_provenance(
            server_metadata=server_metadata,
            settings=settings,
            tasks=task_assets,
            simulator_paths={name: get_libero_path(name) for name in ("bddl_files", "init_states", "assets")},
        )
        research.validate_provenance(
            {
                **settings,
                "server_metadata": server_metadata,
                "reference": args.reference,
                "task_id": 0,
                "initial_state_id": 0,
                "_provenance": provenance,
                "provenance_sha256": research.content_digest(provenance),
            }
        )
        provenance_sha256 = research.content_digest(provenance)
        if not args.task_shards:
            research.write_provenance(args.episode_log_path, provenance)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        if time.time() + (args.task_budget_seconds or 0) + args.cleanup_seconds >= deadline:
            logging.info("Stopping at a task boundary before the lease deadline")
            return
        task_attempt = attempts.get(str(task_id), "initial")
        if (
            not isinstance(task_attempt, str)
            or task_attempt in {"", ".", ".."}
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in task_attempt)
        ):
            raise ValueError("Task attempt must be a safe identifier")
        episode_log_path = args.episode_log_path
        if args.task_shards:
            episode_log_path = research.task_log_path(
                {"episode_log_path": args.episode_log_path}, task_id, task_attempt
            )
            if (
                pathlib.Path(episode_log_path).exists()
                or pathlib.Path(episode_log_path).with_suffix(".provenance.json").exists()
            ):
                raise FileExistsError(f"Refusing to replace task attempt {episode_log_path}")
            if provenance is not None:
                research.write_provenance(episode_log_path, provenance)
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
            failure_kind = None
            artifact_errors = []
            error_message = None
            action_horizon = 10 if args.episode_log_path is not None else None
            query_index = 0
            phase = "reset"
            episode_start = time.monotonic()

            try:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
            except Exception as exc:
                failure_tags.append("reset_error")
                failure_kind = "infrastructure"
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
                    if args.save_videos:
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
                        if isinstance(response, dict):
                            timing.update({k: float(v) for k, v in response.get("policy_timing", {}).items()})
                        latencies.append(timing)
                        query_index += 1
                        phase = "action_validation"
                        if not isinstance(response, dict):
                            raise InvalidPolicyActionsError("Policy response is not an object")
                        if response.get("policy_error"):
                            error = response["policy_error"]
                            if isinstance(error, dict) and error.get("type") == "invalid_actions":
                                raise InvalidPolicyActionsError(str(error.get("message", "Invalid policy actions")))
                            phase = "policy_inference"
                            raise RuntimeError(f"Unclassified policy response error: {error}")
                        try:
                            action_chunk = np.asarray(response["actions"], dtype=float)
                        except (KeyError, TypeError, ValueError) as exc:
                            raise InvalidPolicyActionsError("Missing or nonnumeric action chunk") from exc
                        if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                            raise InvalidPolicyActionsError("Expected a [horizon,7] LIBERO action chunk")
                        if not np.isfinite(action_chunk).all():
                            raise InvalidPolicyActionsError("Policy returned non-finite actions")
                        if action_horizon is not None and len(action_chunk) != action_horizon:
                            raise InvalidPolicyActionsError("Policy action horizon differs from the declared protocol")
                        action_horizon = len(action_chunk)
                        if len(action_chunk) < args.replan_steps:
                            raise InvalidPolicyActionsError("Policy action chunk is shorter than the replan interval")
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
                    failure_kind = "policy" if isinstance(e, InvalidPolicyActionsError) else "infrastructure"
                    failure_tags.append("invalid_actions" if failure_kind == "policy" else f"{phase}_error")
                    error_message = f"{type(e).__name__}: {e}"
                    break

            if not success and not failure_tags:
                failure_tags.append("timeout")
                failure_kind = "policy"

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
                    artifact_errors.append({"kind": "video_error", "message": f"{type(exc).__name__}: {exc}"})
                    logging.exception("Replay video write failed")

            if args.episode_log_path is not None:
                research.append_episode(
                    episode_log_path,
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
                        "failure_kind": failure_kind,
                        "artifact_errors": artifact_errors,
                        "error_message": error_message,
                        "server_metadata": server_metadata,
                        "provenance_sha256": provenance_sha256,
                        "attempt_id": args.attempt_id,
                        "task_attempt_id": task_attempt,
                    },
                )
                if failure_kind == "infrastructure":
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


def calibrate_scenes(args, research):
    """Time native LIBERO components in every scene; these are never scored episodes."""
    output = pathlib.Path(args.calibration_output)
    if output.exists():
        raise FileExistsError(f"Calibration evidence already exists: {output}")
    if args.task_suite_name not in research.SUITES or args.resize_size != 224 or args.num_steps_wait != 10:
        raise ValueError("Calibration requires the native four-suite rendering protocol")
    if not math.isfinite(args.cleanup_seconds) or args.cleanup_seconds < 600:
        raise ValueError("Calibration requires at least 600 seconds of shutdown reserve")
    started = time.perf_counter()
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if suite.n_tasks != 10:
        raise ValueError("Calibration must cover all ten task scenes")
    deadline = (
        args.deadline_unix
        if args.deadline_unix is not None
        else float(os.environ.get("SCALAR_PI_DEADLINE_UNIX") or "inf")
    )
    if math.isnan(deadline) or deadline <= 0:
        raise ValueError("Invalid absolute calibration deadline")
    tasks, assets = [], []
    for task_id in range(10):
        if time.time() + args.cleanup_seconds >= deadline:
            raise TimeoutError("Calibration deadline reached; incomplete evidence was not published")
        task = suite.get_task(task_id)
        state = suite.get_task_init_states(task_id)[0]
        before = time.perf_counter()
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        creation = time.perf_counter() - before
        resets, steps, preprocessing = [], [], []
        try:

            def reset(env=env, state=state, resets=resets):
                tick = time.perf_counter()
                env.reset()
                observation = env.set_init_state(state)
                resets.append(time.perf_counter() - tick)
                return observation

            obs = reset()
            initial_observation = {key: np.asarray(value).copy() for key, value in obs.items()}
            initial_rng = np.random.get_state()
            for step in range(research.MAX_STEPS[args.task_suite_name] + args.num_steps_wait):
                if time.time() + args.cleanup_seconds >= deadline:
                    raise TimeoutError("Calibration deadline reached during a scene")
                tick = time.perf_counter()
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                steps.append(time.perf_counter() - tick)
                if step >= args.num_steps_wait:
                    tick = time.perf_counter()
                    for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                        image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(obs[camera][::-1, ::-1]), args.resize_size, args.resize_size
                            )
                        )
                    preprocessing.append(time.perf_counter() - tick)
                if done:
                    obs = reset()
        finally:
            env.close()
        # A standalone task process starts with this same explicit seed. Verify
        # constructor/reset behavior before allowing task-boundary recovery.
        np.random.seed(args.seed)
        standalone, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            standalone.reset()
            isolated = standalone.set_init_state(state)
            isolated_rng = np.random.get_state()
            if initial_observation.keys() != isolated.keys() or any(
                not np.array_equal(value, np.asarray(isolated[key])) for key, value in initial_observation.items()
            ):
                raise ValueError(f"Task {task_id} standalone initialization differs from contiguous execution")
            if any(not np.array_equal(left, right) for left, right in zip(initial_rng, isolated_rng)):  # noqa: B905 -- Python 3.8.
                raise ValueError(f"Task {task_id} standalone reset RNG differs")
        finally:
            standalone.close()
        tasks.append(
            {
                "task_id": task_id,
                "task_start_parity": True,
                "create_seconds": creation,
                "reset_seconds": max(resets),
                "settling_seconds": sum(steps[: args.num_steps_wait]),
                "step_seconds_p95": float(np.percentile(steps[args.num_steps_wait :], 95)),
                "preprocess_seconds_p95": float(np.percentile(preprocessing, 95)),
                "measured_steps": len(steps),
                "step_seconds": steps,
                "preprocess_seconds": preprocessing,
            }
        )
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        assets.append(
            {
                "task_id": task_id,
                "bddl_sha256": research.file_digest(bddl),
                "initial_states_sha256": [hashlib.sha256(np.asarray(state).tobytes()).hexdigest()],
            }
        )
    provenance_started = time.perf_counter()
    provenance = research.suite_provenance(
        server_metadata={},
        settings={"suite": args.task_suite_name},
        tasks=assets,
        simulator_paths={name: get_libero_path(name) for name in ("bddl_files", "init_states", "assets")},
    )
    report = {
        "schema_version": 1,
        "purpose": "libero_scene_timing",
        "valid_benchmark_evaluation": False,
        "suite": args.task_suite_name,
        "render_size": 256,
        "resize_size": 224,
        "num_steps_wait": 10,
        "max_steps": research.MAX_STEPS[args.task_suite_name],
        "tasks": tasks,
        "provenance": provenance,
        "process_seconds": time.perf_counter() - started,
        "provenance_seconds": time.perf_counter() - provenance_started,
        "scope": "Full native step counts with dummy actions; component timing, not policy outcomes",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


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
