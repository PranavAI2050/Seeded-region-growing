# 🌱 Seeded Region Growing Segmentation GUI

This is a web-based GUI built with **Streamlit** for performing **Seeded Region Growing** segmentation on hyperspectral or multispectral images. The app is accessible online and allows for flexible control over image preprocessing, seed selection, and segmentation constraints.

👉 **Access the GUI here:**  
[https://seeded-region-growing.streamlit.app/](https://seeded-region-growing.streamlit.app/)

---

## 🚀 How to Use the GUI

### 1. **Upload Your Image**
- Upload a `.mat` file containing the image cube.
- The image must be a 3D matrix where:
  - The first two dimensions represent **spatial size** (rows × columns).
  - The third dimension represents **spectral bands**.

### 2. **Preprocessing Options**
- **Normalization**: Select whether to normalize the image before segmentation.
- **Band Filtering**: Optionally select specific spectral bands to include/exclude.

### 3. **Seed Selection**
- Choose from different seed selection methods:
  - **Manual** (user-defined)
  - **Automatic** (random or based on clustering)

### 4. **Segmentation Parameters**
- **Similarity Metric**:
  - Euclidean Distance
  - Spectral Angle
- **Number of Seeds**: Set how many seeds to use for region growing.
- **Similarity Threshold**: Controls the strictness of region expansion.
- **Size Constraint (Optional)**: Enforces a minimum region size to reduce noise.

### 5. **Run Segmentation**
- Click the button to run the seeded region growing algorithm.
- The segmented image will be displayed on the interface with a unique color map.

---

## ⚠️ Notes on Image Requirements & Performance

- The GUI **requires input images in `.mat` format**.
- The algorithm was designed for **multispectral images with high spatial resolution**.
- Results may not be ideal for **hyperspectral images** with many bands but low spatial resolution (e.g., Indian Pines), which can lead to noisy or poorly segmented results.
- For best performance, use **multispectral images (.mat)** that:
  - Have a **high spatial resolution**
  - Contain a **limited number of spectral bands**

---

## 🧪 Example
You can try testing the segmentation using a sample `.mat` multispectral file structured as described. The GUI will display:
- Original image
- Segmented output (with and without size constraint)
- Optional ground truth comparison (if provided)

---

Feel free to contribute or report any issues via GitHub if applicable.
