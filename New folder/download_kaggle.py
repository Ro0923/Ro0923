import os
import shutil
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = r"E:\ViolenceDetection\cache"

path = kagglehub.dataset_download("ipythonx/k4testset")
print("Downloaded from:", path)

target_dir = r"E:\ViolenceDetection\k4testset"
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

print("Final dataset folder:", target_dir)
