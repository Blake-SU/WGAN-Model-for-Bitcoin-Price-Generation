# WGAN-Model-for-Bitcoin-Price-Generation

This repository contains a project for predicting Bitcoin price trends using a Wasserstein Generative Adversarial Network (WGAN) with Seq2Seq architecture.

## Project Overview

This project aims to predict future Bitcoin price trends by utilizing historical price data and technical indicators. We employ a WGAN architecture combined with Seq2Seq modeling to generate plausible future price scenarios.

## Data Collection

In this project, we collected Open-High-Low-Close-Volume (OHLCV) data for Bitcoin from GitHub repositories. We then generated 158 different technical indicators using TALib, including but not limited to CMO (Chande Momentum Oscillator), SMA (Simple Moving Average), and many others. These indicators serve as features that help capture different aspects of price movements and market behavior.

## Models

We implemented a Conditional GAN approach, taking into consideration that our data consists of multivariate time series with dimensions of Batch size × Sequence length × Feature size. To properly combine condition information with noise, we developed a custom encoder-decoder architecture.

Our approach includes:

1. **Seq2Seq Architecture**: We utilize LSTM networks as the backbone for both our Encoder and Decoder components. The Encoder compresses historical price data and technical indicators into a condition vector that can be combined with noise.

2. **Generator**: After training the Encoder, we feed it historical prices and technical indicators to produce a compressed condition vector. This condition vector is then combined with random noise to generate synthetic future price data. The Generator employs 1D CNN as its internal network architecture.

3. **Critic/Discriminator**: Drawing inspiration from WGAN (Wasserstein GAN) methodology, we implemented a critic to help improve our Generator's performance by providing more stable gradients during training.

## Results

In our experiments, we used a 60-day lookback window as a condition to generate 10,000 potential price paths for the next 30 days. The model successfully captured the fat-tailed distribution characteristic of Bitcoin returns, which is a crucial aspect of financial time series modeling.

## Requirements

The project requires Python 3.x and the following key libraries:

- finlab_crypto
- pandas
- numpy
- talib
- psutil
- tracemalloc
- torch
- scikit-learn
- matplotlib
- pytz
- tqdm
