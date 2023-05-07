# Satellite to Radar Nowcasting ⛈️

[![Python 3.10](https://img.shields.io/badge/python-3.10-brightgreen.svg)](#satellite-to-radar-nowcasting-⛈️)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains source files for the bachelor thesis titled:

**Satellite to Radar: Sequence to Sequence learning for precipitation nowcasting** at University of Twente.

## Repository Structure

`ml_variants` contains different models that were trained during the thesis.

`report` contains the source files for the final report of the thesis.

`sandbox` contains all jupyter notebooks used for experimenting.

```
.
├── ml_variants
│  ├── attention
│  └── conv_lstm
├── report
│  ├── images
│  └── main.tex
└── sandbox
```

## Installation

```bash
cd satellite-to-radar-nowcasting
pip install -r requirements.txt
```

## Running

```bash
zenml connect --url=https://zenml.infoplaza.com --username=mbruderer
zenml stack set infoplaza-local-training
python ml_variants/mnist/pipelines/training/run_pipeline.py
```
