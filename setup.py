import setuptools

setuptools.setup(
    name="SCSegamba",
    version="0.25.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for image classification.",
    long_description="A longer description of your project.",
    long_description_content_type="text/markdown",
    url="https://github.com/your/project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'torchvision',
        'mmcv-full',
        'numpy',
        'six',
        'addict',
        'yapf',
        'einops',
        'timm',
        'transformers',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'opencv-python',
        'pyyaml',
        'tqdm',
    ]
)
