import os
from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py as build
from datetime import datetime

# 检查是否安装了 OpenCV
def check_opencv_installed():
    try:
        import cv2
        print("✅ \033[1mOpenCV is already installed.\033[0m")
        if 'contrib' in cv2.getBuildInformation():
            print("   ✔️ Installed version: \033[1mopencv-contrib-python\033[0m\n")
        else:
            print("   ✔️ Installed version: \033[1mopencv-python\033[0m\n")
    except ImportError:
        print("\n\033[93m🔔 Note: OpenCV is not currently installed.\033[0m")
        print("\033[93mTo fully utilize all the features of func2stream, please consider installing one of the following packages:\033[0m")
        print("\n\033[92m  👉 pip install opencv-python\033[0m")
        print("\033[93m    or\033[0m")
        print("\033[92m  👉 pip install opencv-contrib-python\033[0m")
        print("\nFor more information, please visit:")
        print("\033[94mhttps://pypi.org/project/opencv-python/\033[0m")
        print("\033[94mhttps://pypi.org/project/opencv-contrib-python/\033[0m\n")

date_suffix = datetime.now().strftime("%y%m%d%H%M")

major_version = 0
minor_version = 0
patch_version = 0
base_version = f"{major_version}.{minor_version}.{patch_version}"
base_version_next = f"{major_version}.{minor_version}.{patch_version+1}"

if os.getenv('RELEASE_VERSION'):
    full_version = base_version
else:
    full_version = f"{base_version_next}.dev{date_suffix}"

class PostInstallCommand(Command):
    """Post-installation for installation mode."""
    description = "Run post-installation tasks"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("\n" + "="*50)
        print("🎉 \033[1mInstallation complete! Thank you for installing func2stream.\033[0m 🎉")
        print("🔄 Effortlessly transform functions into asynchronous elements for building high-performance pipelines.\n")
        # 显示 OpenCV 检查结果
        check_opencv_installed()
        print("\033[96m🌟 For more information and support, please visit our GitHub repository:\033[0m")
        print("\033[94mhttps://github.com/BICHENG/func2stream\033[0m")
        print("="*50 + "\n")

class CustomBuildCommand(build):
    """Custom build command to display a message after build."""
    def run(self):
        build.run(self)
        print("\n" + "="*50)
        print("🔧 \033[1mBuild complete! Thank you for building func2stream.\033[0m 🔧")
        print("🔄 Effortlessly transform functions into asynchronous elements for building high-performance pipelines.\n")
        print("\033[96m🌟 For more information and support, please visit our GitHub repository:\033[0m")
        print("\033[94mhttps://github.com/BICHENG/func2stream\033[0m")
        print("="*50 + "\n")

setup(
    name='func2stream',
    version=full_version,
    description='Effortlessly transform functions into asynchronous elements for building high-performance pipelines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='BI CHENG',
    url='https://github.com/BICHENG/func2stream',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MPL-2.0',
    cmdclass={
        'install': PostInstallCommand,
        'build_py': CustomBuildCommand,
    }
)
