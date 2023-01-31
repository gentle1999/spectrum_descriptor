<!--
 * @Author: TMJ
 * @Date: 2022-12-27 09:08:35
 * @LastEditors: TMJ
 * @LastEditTime: 2023-01-31 10:12:33
 * @Description: 请填写简介
-->
# spectrum-descriptor

## Introduction

spectrum descriptor is a tool to extract and re-encode information from spectral image or spectral curves.

## Dependences

Third-party python packages are required for generating spectrum descriptor.

```
python>=3.7
torch>=1.0.0
numpy>=1.17.4
rdkit>=2020.09.1.0
pandas>=1.3.5
scikit-image>=0.19.1
scikit-learn>=1.0.2
matplotlib>=3.5.3
```

In order to run Jupyter Notebook for machine learning application demonstration, several machine learning, deep learning and visualazation third-party python packages are required.

```
seaborn>=0.12.1
tqdm>=4.64.1
```

## Usage

The core script to generate spectrum descriptor is in [utils](utils/__init__.py), where includes [img2spec.py](utils/img2spec.py) and [spec2des.py](utils/spec2des.py).

The [spec_notebook.ipynb](spec_notebook.ipynb) shows how to use and some test for it.

The [OCR_performance.ipynb](OCR_performance.ipynb) shows the performance of our OCR tool.