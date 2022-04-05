import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="protego",
    version="0.0.10",
    author="Bradley Reeves",
    author_email="reevesbra@outlook.com",
    description="Detect SQL Injection Payloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reevesba/protego",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
