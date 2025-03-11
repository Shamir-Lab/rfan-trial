from setuptools import find_packages, setup

setup(
    name="ada-trial",
    version="0.0.0",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "click==8.0.1",
        "torch==1.10.0",
        "numpy==1.21.2",
        "scipy==1.7.1",
        "pandas==1.3.4",
        "pyreadr==0.4.2",
        "gpytorch==1.5.1",
        "seaborn==0.13.0",
        "hyperopt==0.2.5",
        "ray[tune]==1.7.0",
        "ray[default]==1.7.0",
        "protobuf==3.20.*",
        "matplotlib==3.4.3",
        "tensorboard==2.7.0",
        "torchvision==0.11.1",
        "scikit-learn==0.24.2",
        "pytorch-ignite==0.4.7",
		"statsmodels==0.14.0",
    ],
)
