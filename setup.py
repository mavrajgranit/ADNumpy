import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ADNumpy', version='1.0',
    author='mavrajgranit',
    author_email="mavrajgranit@gmail.com",
    url="https://github.com/mavrajgranit/ADNumpy",
    description="Automatic Differentiation for Machine Learning in Numpy.",
    long_description="Automatic Differentiation for Machine Learning in Numpy.",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
