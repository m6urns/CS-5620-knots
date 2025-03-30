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
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'collect_overhand_knot=scripts.collect_overhand_knot:main',
            'train_overhand_knot=scripts.train_overhand_knot:main',
            'run_overhand_classifier=scripts.run_overhand_classifier:main',
        ],
    },
)