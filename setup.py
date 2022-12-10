import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='mndynamics',
    version='0.1.0',
    # scripts=['itng'],
    author="Abolfazl Ziaeemehr",
    author_email="a.ziaeemehr@gmail.com",
    description="Modeling neural dynamcs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ziaeemehr/mndynamics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.1',
    install_requires=requirements,
    # include_package_data=True,
)
