import yaml
import os

def load_config(filename="config.yaml"):
    """
    从 src/config/ 目录中加载 YAML 配置文件。
    """
    # 获取当前文件所在目录（即 src/config）
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config