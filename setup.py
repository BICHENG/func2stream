import os
from setuptools import setup, find_packages
from datetime import datetime

date_suffix = datetime.now().strftime("%Y%m%d") #YYYYMMDD
base_version = '0.0.1'
full_version = f"{base_version}.dev{date_suffix}"

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
        'opencv-python',
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
)