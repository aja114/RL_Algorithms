from gym import envs
import os


def get_env_list():
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    return env_ids


def get_data_location(training_name, algorithm):
    res_path = f'Data/{algorithm}_' + training_name

    if os.path.exists(f"{res_path}"):
        os.system(f"rm -f {res_path}/*")
    else:
        os.mkdir(f"{res_path}")

    return res_path


def prepare_data_directory(res_path):
    os.system(f"touch {res_path}/result.csv")

    with open(f"{res_path}/result.csv", 'w') as f:
        f.write("episodes,num_steps,reward\n")