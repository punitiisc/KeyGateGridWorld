from setuptools import setup, find_packages

setup(
    name="key_gate_gridworld",
    version="0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "matplotlib"],
)

