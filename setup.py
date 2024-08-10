from setuptools import find_packages, setup

setup(
    name='self-compression-jax',
    packages=find_packages(),
    version='0.0.1',
    author='juvi21',
    url='https://github.com/juvi21/self-compression-jax',
    install_requires=[
        "jax>=0.4.31",
        "jaxlib>=0.4.31",
        "flax>=0.8.5",
        "optax>=0.2.3",
        "numpy>=2.0.1",
        "matplotlib>=3.9.1",
        "tqdm>=4.66.5",
    ],
    python_requires=">=3.10.12",
)