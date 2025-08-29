from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """Return a list of requirements from a requirements.txt file."""
    requirements = []
    with open(file_path, "r") as f:
        requirements = f.read().splitlines()

    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements


setup(
    name="ml_project",
    version="0.0.1",
    author="sourav",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
