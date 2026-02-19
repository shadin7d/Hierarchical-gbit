from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hierarchical-gbit",
    version="2.0.0",
    author="shadin7d",
    description="Physics-based hierarchical cavity system simulation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shadin7d/hierarchical-gbit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
