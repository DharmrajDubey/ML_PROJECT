# Implementation of PCA with ANN Algorithm for Face Recognition

This project involves building a **Face Recognition System** using **Principal Component Analysis (PCA)** and an **Artificial Neural Network (ANN)** for classification. The PCA algorithm is used for feature extraction and dimensionality reduction, while ANN is used for training and prediction.

---

## 📁 Dataset

We use a facial image dataset available here:   PROVIDED BY INTERNSHIP STUDIO
🔗 [Face Dataset (GitHub)](https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip)

---

## 📌 Objectives

- Implement a complete PCA pipeline to generate eigenfaces.
- Use ANN to classify facial features extracted via PCA.
- Evaluate the system on accuracy and robustness.
- Test with both known and imposter face inputs.

---

## 🛠️ Libraries Used

- `NumPy` and `SciPy` – for matrix operations, eigen decomposition
- `OpenCV (cv2)` – for reading and processing image data
- `sklearn`, `matplotlib` (optional) – for training, testing, plotting
- `TensorFlow/Keras` or `PyTorch` – for building the ANN

---

## 📈 Training Steps

1. **Generate Face Database:** Convert each face image into a column vector.
2. **Calculate Mean Face Vector.**
3. **Mean Centering:** Subtract the mean vector from all image vectors.
4. **Covariance Matrix Calculation:** Use surrogate covariance (as suggested by Turk & Pentland).
5. **Eigenvalue & Eigenvector Decomposition.**
6. **Select Best Directions (Top K Eigenvectors).**
7. **Generate Eigenfaces:** Project mean-centered data to selected eigenvectors.
8. **Generate Face Signatures:** Create unique feature vectors per face.
9. **Train ANN:** Use backpropagation to classify these signatures.

---

## 🧪 Testing Steps

1. Convert test image to column vector.
2. Mean center using training mean face.
3. Project onto eigenfaces to get test signature.
4. Feed into the trained ANN to classify the face.

---

## 📊 Evaluation

- Use **60%** of data for training and **40%** for testing.
- Analyze performance by varying `k` (number of eigenfaces).
- Test with **imposters** (unknown faces) to check system robustness.
- Plot: **Accuracy vs. K-value** graph to observe the optimal feature vector dimension.

---

## 📎 Project Highlights

- Implements Turk and Pentland's Eigenfaces approach efficiently.
- Modular design for easy adjustments and experiments.
- Flexible ANN integration – any backend (TensorFlow, PyTorch) can be used.
- Supports adding new faces and re-training.

---

## 🚀 Future Enhancements

- Implement real-time face recognition via webcam.
- Integrate with GUI for user-friendly interface.
- Add dropout/layer normalization to ANN to boost generalization.

---

## 📄 Reference

Turk, M., & Pentland, A. (1991). **Eigenfaces for recognition**. Journal of Cognitive Neuroscience.

---

## 👨‍💻 Author

Dharmraj Dubey - Intern under INTERNSHIP STUDIO
This project was developed as part of an academic course on Machine Learning and Pattern Recognition.

