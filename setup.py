from setuptools import setup, find_packages

setup(
    name="chggen",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pymatgen==2023.9.10",
        "e3nn==0.5.1",
    ],
    entry_points={
        "console_scripts": [
            "chggen=chggen.main:main",  # Adjust this if your entry point is different
        ],
    },
    author="Peichen Zhong",
    description="A brief description of CHGGen",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhongpc/chggen",
    classifiers=[],
    python_requires=">=3.9",
)
