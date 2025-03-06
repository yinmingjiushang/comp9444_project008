import os
import shutil
import subprocess
import sys

# 项目根目录下的数据路径
DATA_DIR = "data/ScienceQA"
GIT_REPO_URL = "https://github.com/lupantech/ScienceQA.git"

def download_dataset():

    # 若数据不存在则下载
    if not os.path.exists(DATA_DIR):
        print(f"{DATA_DIR}未找到，开始从GitHub下载...")
        subprocess.run(["git", "clone", GIT_REPO_URL, DATA_DIR], check=True)
        print("ScienceQA数据集下载完成！")
    else:
        print("ScienceQA数据集已存在，无需下载。")

if __name__ == "__main__":
    download_dataset()
