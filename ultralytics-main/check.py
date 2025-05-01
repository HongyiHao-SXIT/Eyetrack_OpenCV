import importlib.metadata
import os

# 指定 requirements.txt 文件的路径
requirements_path = 'C:\\Users\\Lanyi\\Desktop\\Project\\Eyetrack_Furrysuit\\ultralytics-main\\requirements.txt'

# 检查文件是否存在
if not os.path.exists(requirements_path):
    print(f"错误: 未找到 {requirements_path} 文件。")
else:
    # 读取 requirements.txt 文件
    with open(requirements_path, 'r') as f:
        required_packages = f.read().splitlines()

    # 遍历每个包
    for package in required_packages:
        try:
            package_name, package_version = package.split('==')
            # 获取当前环境中该包的版本
            try:
                installed_version = importlib.metadata.version(package_name)
                if installed_version != package_version:
                    print(f"{package_name}: 要求版本 {package_version}，当前安装版本 {installed_version}")
            except importlib.metadata.PackageNotFoundError:
                print(f"{package}: 未安装")
        except ValueError:
            print(f"错误: {package} 格式不正确，应为 '包名==版本号'")   