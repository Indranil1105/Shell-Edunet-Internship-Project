# Shell-Edunet-Internship-Project


This project implements an image classification model using **TensorFlow** and **EfficientNetV2** to classify images into 6 different types of trash: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`. The model is built using a high-level Keras API and supports visualization and interactive testing via **Gradio**.

---

## 🚀 Features

- Preprocessing of custom image dataset with `image_dataset_from_directory`
- Automatic train/validation/test split
- Image visualization for sanity checks
- Transfer learning using `EfficientNetV2B2`
- Performance optimization using caching and prefetching
- Class imbalance handling with `compute_class_weight`
- Model evaluation with `confusion_matrix` and `classification_report`
- Interactive web interface using `Gradio`

---

## 🗂 Dataset

- The dataset should be placed at the path specified in the script:
  ```
  C:\Users\Dell\Desktop\Shell Internship\Dataset\TrashType_Image_Dataset
  ```
- Expected structure:
  ```
  TrashType_Image_Dataset/
  ├── cardboard/
  ├── glass/
  ├── metal/
  ├── paper/
  ├── plastic/
  └── trash/
  ```

---

## 📦 Libraries Used

- `numpy`, `matplotlib`, `seaborn` — for numerical ops and visualization
- `tensorflow`, `keras` — for deep learning and model building
- `EfficientNetV2B2` — pre-trained model for transfer learning
- `sklearn` — for class weighting and evaluation
- `gradio` — to build a simple web-based interface for predictions

---

## 📊 Workflow

1. **Load Dataset:**
   - Training: 80% of images
   - Validation: 10%
   - Testing: 10%

2. **Visualize Data:**
   - Plot samples from each class to verify labeling.

3. **Prepare Model (not shown in full here):**
   - Uses a `Sequential` model with a `Rescaling` layer and `EfficientNetV2B2` as base.

4. **Evaluation:**
   - Performance is measured using classification report and confusion matrix.

---

## 🖼 Sample Visualization

The code includes functionality to show sample training images labeled by class:

```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(12):
    ax = plt.subplot(4, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
```

---

## 💡 Getting Started

1. Clone the repository.
2. Place your dataset in the correct path as mentioned above.
3. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn gradio
   ```
4. Run the notebook or Python script to train and visualize the model.




