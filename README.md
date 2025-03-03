# RFAN

An implementation of Randomize First Augment Next (RFAN), a design framework for Phase-III clinical trials.

## Abstract
Randomized Controlled Trials (RCTs) are the gold standard for evaluating the effect of new medical treatments. Treatments must pass stringent regulatory conditions in order to be approved for widespread use, yet even after the regulatory barriers are crossed, real-world challenges might arise: Who should get the treatment? What is its true clinical utility? Are there discrepancies in the treatment effectiveness across diverse and under-served populations? We introduce two new objectives for future clinical trials that integrate regulatory constraints and treatment policy value for both the entire population and under-served populations, thus answering some of the questions above in advance. Designed to meet these objectives, we formulate Randomize First Augment Next (RFAN), a new framework for designing Phase III clinical trials. Our framework consists of a standard randomized component followed by an adaptive one, jointly meant to efficiently and safely acquire and assign patients into treatment arms during the trial. Then, we propose strategies for implementing RFAN based on causal, deep Bayesian active learning. Finally, we empirically evaluate the performance of our framework using synthetic and real-world semi-synthetic datasets.

## Installation
```.sh
$ conda env create -f environment.yml
$ conda activate ada-trial
$ pip install .
```

## Run
```.sh
$  python -m src.main
```

