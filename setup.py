from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edf",
    version="0.2.0",
    author="Hyunwoo Ryu",
    author_email="tomato1mule@gmail.com",
    description="Standalone package of Equivariant Descriptor Fields (EDFs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomato1mule/edf",
    project_urls={
        "Bug Tracker": "https://github.com/tomato1mule/edf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    packages=['edf'],
    python_requires="<3.9",
    install_requires=[
        'numpy==1.23.5',
        'scipy==1.10.0',
        'torch==1.11.0',
        'torchvision==0.12.0',
        'torchaudio==0.11.0',
        'torch-sparse==0.6.13',
        'torch-cluster==1.6.0',
        'torch-scatter==2.0.9',
        'torch-spline-conv==1.2.1',
        'e3nn==0.4.4',
        'xitorch==0.3.0',
        'iopath',        # 0.1.9
        'fvcore',        # 0.1.5.post20220504
        'pytorch3d==0.7.2',
        'open3d==0.16.0',
        'pyyaml',        # 6.0
        'tqdm',          # 4.64.1
        'jupyter',       # 1.0.0
        'plotly==5.12.0',
        'dash==2.7.1',
        'dash_vtk==0.0.9',
        'dash_daq==0.5.0',
    ],
)