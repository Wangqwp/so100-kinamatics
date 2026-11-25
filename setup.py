from glob import glob

from setuptools import setup

package_name = "so100_trajectory"

setup(
    name=package_name,
    version="0.1.0",
    # temp / utils 是带 __init__.py 的普通 Python 包；其余节点模块平铺安装。
    packages=["temp", "utils"],
    py_modules=[
        "demo_trajectory_bus_publisher",
        "hole_insertion_geometry",
        "so100_kinematics",
        "trajectory_bus_node",
    ],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/calibration", glob("calibration/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SO100 user",
    maintainer_email="user@example.com",
    description="SO-100/101 policy EE trajectory and hole-insertion trajectory topics for ROS 2.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "trajectory_bus = trajectory_bus_node:main",
            "demo_trajectory_bus = demo_trajectory_bus_publisher:main",
        ],
    },
)
