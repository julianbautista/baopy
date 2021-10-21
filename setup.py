import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="baopy",
    version="0.0.1",
    author="Julian Bautista",
    author_email="bautista@cppm.in2p3.fr",
    description="Package for BAO fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/julianbautista/baopy",
    package_dir={"": "py"},
    packages=['baopy'],
    python_requires=">=3.6",
    install_requires=[
        "iminuit",
        "hankl",
    ]
)