import os
import toml

from setuptools import setup, find_packages


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))


# NOTE: upstream wheeledlab_rl also pins gymnasium==1.0.0. That is left out on purpose:
# isaaclab requires gymnasium==1.2.1, and installing this package must not downgrade it.
INSTALL_REQUIRES = [
    "psutil",
    "rich",
    "av",
    "rsl-rl-lib>=2.3.0",
]

setup(
    name="witsense_ugv_rl",
    packages=find_packages(include=["witsense_ugv_rl", "witsense_ugv_rl.*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
