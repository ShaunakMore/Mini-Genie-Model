
# Mini-Genie: An Action-Conditioned World Model

## 📌 Overview
**Mini-Genie** is a deep learning project inspired by Google DeepMind's groundbreaking work on generative interactive environments.  
This project implements a **foundational world model** that learns the dynamics of an environment to **predict future frames** based on the current state and a given action.

Built with **PyTorch**, Mini-Genie uses a **CNN-Transformer architecture** to process visual information and model temporal relationships.  
It serves as a practical exploration into the **core concepts behind world models** and **action-conditioned video prediction**.

---

## 🧠 Model Architecture
The model is composed of several key modules that work in concert to predict the next frame.

### 1. **ImageEncoder**
A **Convolutional Neural Network (CNN)** that processes the input image frame `(64x64x3)`.  
It downsamples the image through a series of convolutional layers to extract a **high-level feature representation**, which is then **flattened into a sequence of spatial tokens** for the Transformer.

### 2. **ActionEmbedding**
Converts discrete, categorical actions (e.g., `Turn Left`, `Move Forward`) into **dense vector embeddings**.  
This numerical representation of the action is concatenated with the image tokens for Transformer processing.

### 3. **Transformer**
The **core of the model's temporal reasoning**.  
It takes the sequence of image and action tokens, adds **positional embeddings**, and uses **self-attention mechanisms** to model how the action will influence the visual features of the environment.

### 4. **FrameDecoder**
A **transposed convolutional network (deconvolutional network)** that reconstructs the predicted next frame.  
It reshapes the processed tokens from the Transformer into a spatial format and upsamples them to generate the final output image.

### 5. **MiniGenie**
The main class that orchestrates:
1. Passing the initial frame & action to the encoder and action embedding.
2. Processing with the Transformer.
3. Decoding the predicted frame with the FrameDecoder.

---

## 📊 Results & Evaluation
The model was trained on `(initial_frame, action, next_frame)` tuples and evaluated on a held-out test set.  

**Performance Metrics:**
- **Mean Squared Error (MSE):** `0.000447`
- **Mean Absolute Error (MAE):** `0.001996`
- **Peak Signal-to-Noise Ratio (PSNR):** `33.50 dB`

The image at the top of this README shows a **side-by-side comparison** of the model's predictions against the actual next frames.

---

## ⚙️ Setup and Usage

### 1. **Prerequisites**
Ensure you have **Python 3** installed.  
Install the dependencies:
```
pip install -r requirements.txt
```

**requirements.txt**

```
torch
torchvision
numpy
matplotlib
tqdm
gymnasium
minigrid
opencv-python
Pillow
```

---

### 2. **Dataset**

The model expects a dataset split into three `.npy` files:

* `images.npy` — Initial frames
* `actions.npy` — Corresponding actions
* `next_frames.npy` — Resulting next frames

Place these files in the **root directory** of the project.

---

### 3. **Training**

To train from scratch, run the training cells in **`mini-genie.ipynb`**:

* Load and split the dataset.
* Initialize **MiniGenie**, loss function, and optimizer.
* Run the training loop for the chosen number of epochs.
* Save the best model weights to `mini-genie.pth`.

---

### 4. **Evaluation**

To evaluate:

* Load the saved weights from `mini-genie.pth`.
* Calculate MSE, MAE, and PSNR on the test set.
* Generate `model_evaluation_grid.png` to visualize predictions.

---

## 🚀 Future Work

* **Larger Dataset:** Improve generalization by training on more diverse data.
* **Higher Resolution:** Extend architecture to handle higher-resolution images.
* **Interactive Environment:** Create a playable loop where the model's output feeds back as input.

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.