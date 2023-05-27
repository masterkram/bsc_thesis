# Satellite to Radar Nowcasting ⛈️

[![Python 3.10](https://badgers.space/badge/python/3.10/green?icon=https://cdn.simpleicons.org/python/white)](#satellite-to-radar-nowcasting-⛈️)
[![Pytorch](https://badgers.space/badge/pytorch/2.0.1/orange?icon=https://cdn.simpleicons.org/pytorch/white)](https://github.com/psf/black)
[![Code style: black](https://badgers.space/badge/code_style/black/black)](https://github.com/pytorch/pytorch)


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


## Experiments

| Experiment Name | Experiment |  Trained Model |
| --------------- | ---- | ------------- |
|  `sat2rad_conv_lstm`  |  [view](https://mlflow.infoplaza.com/#/experiments/11?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All%20Runs&selectedColumns=attributes.%60Source%60,attributes.%60Models%60&isComparingRuns=false&compareRunCharts=dW5kZWZpbmVk)    |
