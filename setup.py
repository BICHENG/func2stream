import os
from setuptools import setup, find_packages, Command
from datetime import datetime
from pathlib import Path

# 
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

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

# using read_text to avoid encoding issues:
# UnicodeDecodeError: 'gbk' codec can't decode byte ^^^ in position ^^^: illegal multibyte sequence
setup(
    name='func2stream',
    version=full_version,
    description='Effortlessly transform functions into asynchronous elements for building high-performance pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='BI CHENG',
    url='https://github.com/BICHENG/func2stream',
    packages=find_packages(),
    install_requires=[
        'loguru',
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
)
