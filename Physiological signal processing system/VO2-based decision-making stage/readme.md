# VO<sub>2</sub>-based decision-making stage
This is a VO<sub>2</sub> memristor-based LSNN model for physiological signal processing implemented using the PyTorch-based SpikingJelly module. Before using, see `requirements.txt` for a list of required Python packages. Included .py files:
- `main_ecg.py`: Train and test a VO<sub>2</sub> memristor-based LSNN model on the provided ECG dataset
- `main_eeg.py`: Train and test a VO<sub>2</sub> memristor-based LSNN model on the provided EEG dataset
- `dataset.py`: Defines the ECG and EEG dataset classes
- `model.py`: Defines the VO<sub>2</sub> memristor-based LSNN model
- `surrogate.py`: Defines the surrogate function used to train the model
- `utils.py`: Miscellaneous functions
## Arrhythmia detection
Dataset used in the article is given in the folder `/data_ecg`. Run `main_ecg.py` to train the network. 
## Epilepsy detection
Dataset used in the article is given in the folders `/data_eeg_balanced_2530` (training set) and `/data_eeg_contiguous` (testing set). First, decompress the EEG dataset by running `decompress.py` bin these folders. Then, run `main_eeg.py` to train the network.
