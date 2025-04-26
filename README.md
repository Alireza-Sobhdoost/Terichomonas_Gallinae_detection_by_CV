# Terichomonas gallinae Image Classifier

A Windows-based Tkinter GUI that uses three deep-learning frameworks to detect Terichomonas gallinaes in microscope images:

1. **VAE + Color Heuristic**  
2. **GAN-based Anomaly Detector**  
3. **ResNet-50 + Focal Loss** (in the GUI)

---

## Overview

This project explores and compares three approaches for Terichomonas gallinae detection:

- **VAE + Color Heuristic**  
  Learns a latent representation of Terichomonas gallinae images and uses per-pixel color deviations to highlight anomalies.

- **GAN-based Anomaly Detector**  
  Trains a generator to reconstruct Terichomonas gallinae images and a discriminator to score realism; combines reconstruction error with color features.

- **ResNet-50 + Focal Loss**  
  Fine-tunes a ResNet-50 backbone with custom Focal Loss to handle class imbalance. This is the model powering the GUI.

---

## Installation

1. **Python 3.8+ on Windows**  
2. Install required packages via pip:  
   - torch  
   - torchvision  
   - pillow  
3. Verify Tkinter is available: launch the standard Python installer for Windows, which includes Tkinter by default.  

---

## Running the App

### Double-click

- Rename the script to `Terichomonas gallinae_classifier.pyw`  
- Associate `.pyw` with your Python executable  
- Double-click to launch the GUI

> _Print statements won’t appear when double-clicked; use pop-up dialogs for debugging._

### Command-line

- Open Command Prompt  
- Navigate to the script folder  
- Run `python Terichomonas gallinae_classifier.py` to see both console logs and the GUI

---

## Usage

1. Click **Load Image** and select a microscope photo.  
2. Click **Classify** to run inference.  
3. View the predicted label (Tterichomonas gallinaeed” or Tno_terichomonas gallinaeed”) and the confidence probability.

---

## Results

| Method                       | Terichomonas gallinae-only F1 | Mixed F1  |
|------------------------------|-------------:|----------:|
| VAE + Color Heuristic        |      0.8709  |      n/a  |
| GAN-based Anomaly Detector   |      0.8800  |     0.660 |
| ResNet-50 + Focal Loss (GUI) |      1  |      0.9032  |

---

## Troubleshooting

- **App won’t open**: confirm Python 3.8+ and required packages.  
- **`tkinter` missing**: reinstall Python with the Windows installer.  
- **No debug prints**: double-click mode hides stdout—use GUI pop-ups instead.

---

## Credits

**Author:** Sepehr Sobhdoost  
**Model file:** `classify_best_model3.pth`  

Feel free to open an issue or submit a PR with questions or improvements!
