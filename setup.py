import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
with open("requirements.txt", "r") as fh:
    for line in fh:
        requirements.append(line.strip())

setuptools.setup(
    name="protego-sqli",
    version="1.0.0",
    author="Bradley Reeves",
    author_email="reevesbra@outlook.com",
    description="Detect SQL Injection Payloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reevesba/protego",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
