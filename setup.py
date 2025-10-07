from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent

setup(
    name="adjepa",
    version="0.1.0",
    description="ADJEPA",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
    ],
    packages=find_packages(where="ADJEPA/src"),
    package_dir={"": "ADJEPA/src"},
    package_data={"": ["*.yaml", "*.yml", "*.json"]},
    entry_points={
        "console_scripts": [
            "adjepa = ADJEPA.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
