from setuptools import setup, find_packages

setup(
    name="knots",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "fastapi",
        "uvicorn",
        "pytest",
    ],
)