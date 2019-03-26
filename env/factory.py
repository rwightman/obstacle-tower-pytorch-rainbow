from env.wrappers import ToTorchTensors, SkipFrames
from env.env_obt import CustomObstacleTowerEnv
from obstacle_tower_env import ObstacleTowerEnv


def create_env(
        env_filename,
        custom=True, skip_frames=4,
        docker=False, realtime=False,
        worker_id=0, device='cpu'
):

    if custom:
        env = CustomObstacleTowerEnv(
            env_filename,
            docker_training=docker, realtime_mode=realtime,
            worker_id=worker_id, timeout_wait=60)
    else:
        env = ObstacleTowerEnv(
            env_filename,
            docker_training=docker, realtime_mode=realtime,
            worker_id=worker_id, timeout_wait=60)
    if skip_frames > 1:
        env = SkipFrames(env, skip=skip_frames)
    env = ToTorchTensors(env, device=device)

    return env
