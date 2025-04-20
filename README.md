# Anomaly Detection on ROAD Dataset

This project implements a deep convolutional autoencoder to detect anomalies in LOFAR spectrograms from the Radio Observatory Anomaly Detection (ROAD) dataset.

- Input: (256, 256, 4) spectrograms (frequency, time, polarization)
- Output: Anomaly scores based on reconstruction error (MAE)

## File Structure

- `AnomalyDetectionROAD.ipynb` — Notebook for data exploration and prototyping
- `report.md` — Short write-up with results and insights
- `ROAD_dataset.h5`— Dataset file (download separately)
- images/ — Directory for saving images
- `README.md` — This file

## How to Run

1. Open the Notebook in Google Colab
2. Download the data set from: https://zenodo.org/records/8028045 and upload the ROAD_dataset.h5 file to Colab session.
3. Run All Cells

##  Installation & Dependencies
If running python locally, install the following libraries: 
`pip install tensorflow numpy matplotlib h5py`

This project was tested with:
- Python 3.10+
- TensorFlow 2.14+

## Evaluation Metrics

- MAE (Mean Absolute Error)
- SSIM (Structural Similarity Index)
- Thresholding for anomaly detection via manual tuning

## Notes
Trained only on unsupervised samples
