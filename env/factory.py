from env.wrappers import ToTorchTensors, ToTorchTensorsWithAug, SkipFrames
from env.env_obt import CustomObstacleTowerEnv
from obstacle_tower_env import ObstacleTowerEnv


def create_env(
        env_filename,
        custom=True, large=False, custom_reward=True,
        skip_frames=0, docker=False, realtime=False, random_aug=0.,
        worker_id=0, device='cpu'
):

    if custom:
        env = CustomObstacleTowerEnv(
            env_filename,
            mode='retro_large' if large else 'retro',
            custom_reward=custom_reward,
            docker_training=docker, realtime_mode=realtime,
            worker_id=worker_id, timeout_wait=60)
    else:
        env = ObstacleTowerEnv(
            env_filename,
            docker_training=docker, realtime_mode=realtime,
            worker_id=worker_id, timeout_wait=60)
    if skip_frames > 1:
        env = SkipFrames(env, skip=skip_frames)
    if random_aug > 0.:
        env = ToTorchTensorsWithAug(env, device=device, aug_prob=random_aug)
    else:
        env = ToTorchTensors(env, device=device)

    return env
