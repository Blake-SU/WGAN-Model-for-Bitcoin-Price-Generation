# crypto-gan-seq2seq-project

This repository contains a project for generating cryptocurrency data using a combination of Seq2Seq models and Conditional GANs.

## Project Overview

- **Data Acquisition:**  
  Retrieves OHLCV data for BTCUSDT from Binance and applies technical analysis indicators using TALib.

- **Data Preprocessing:**  
  Processes data (e.g., log transform and differencing) and prepares it for training.

- **Seq2Seq Model:**  
  Implements an encoder-decoder architecture to model time series data.

- **Conditional GAN:**  
  Uses a CNN-based generator and an LSTM-based projection discriminator to generate simulated future log returns.

- **Prediction and Visualization:**  
  Generates simulated predictions, converts log returns to future prices, and plots predicted price bands compared to actual prices.

## Requirements

The project requires Python 3.x and the following key libraries:

- `finlab_crypto`
- `pandas`
- `numpy`
- `talib`
- `psutil`
- `tracemalloc`
- `torch`
- `scikit-learn`
- `matplotlib`
- `pytz`
- `tqdm`



