"""
QuantLab: Quantitative Finance Laboratory

A comprehensive financial engineering toolkit with zero external dependencies
beyond NumPy. Provides production-quality implementations of:

- Options pricing: Black-Scholes, Binomial Tree, Monte Carlo, Heston
- Risk management: VaR, CVaR, backtesting
- Portfolio optimization: Markowitz, Black-Litterman, Risk Parity
- Volatility modeling: GARCH family, realized volatility estimators
- Yield curve: bootstrapping, Nelson-Siegel, Svensson, analytics
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantlab",
    version="1.0.0",
    author="QuantLab Contributors",
    author_email="quantlab@example.com",
    description="Quantitative Finance Laboratory — comprehensive financial engineering toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantlab/quantlab",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    keywords="quantitative finance, options pricing, risk management, portfolio optimization, "
             "volatility, yield curve, black-scholes, var, garch",
)
