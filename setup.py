import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ramo",  # The name of the package.
    version="0.0.1",  # The current release version.
    author="Willem Röpke",  # The full name of the author.
    author_email="willem.ropke@vub.be",  # Email address of the author.
    description="Algorithms for computing or learning equilibria in multi-objective games",  # Short tagline.
    long_description=long_description,  # Long description read from the readme file.
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),  # List of all python modules to be installed.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Information to filter the project on PyPi website.
    python_requires='>=3.6',  # Minimum version requirement of the package.
)
