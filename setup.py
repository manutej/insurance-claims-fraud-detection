"""Setup script for Insurance Claims Data Ingestion Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "CLAUDE.md").read_text()

# Read requirements
requirements_path = this_directory / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="insurance-claims-pipeline",
    version="1.0.0",
    author="Insurance Claims Analysis Team",
    author_email="team@insurance-claims.com",
    description="Production-ready data ingestion pipeline for insurance fraud detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/insurance-claims/data-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
        ],
        "performance": [
            "pyarrow>=10.0.0",
            "fastparquet>=0.8.0",
        ],
        "monitoring": [
            "memory-profiler>=0.60.0",
            "psutil>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "claims-pipeline=main:cli",
        ],
    },
    package_data={
        "src": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="insurance, fraud detection, data ingestion, healthcare, claims processing",
    project_urls={
        "Bug Reports": "https://github.com/insurance-claims/data-pipeline/issues",
        "Source": "https://github.com/insurance-claims/data-pipeline",
        "Documentation": "https://insurance-claims-pipeline.readthedocs.io/",
    },
)
