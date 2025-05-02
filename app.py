import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, maximum_filter
from collections import deque
import matplotlib.colors as mcolors

# --- Utility Functions ---
def normalize_minmax(data):
    H, W, B = data.shape
    norm_data = np.zeros_like(data, dtype=np.float32)
    for b in range(B):
        band = data[:, :, b]
        min_val = band.min()
        max_val = band.max()
        if max_val - min_val != 0:
            norm_data[:, :, b] = (band - min_val) / (max_val - min_val)
    return norm_data

def normalize_zscore(data):
    H, W, B = data.shape
    norm_data = np.zeros_like(data, dtype=np.float32)
    for b in range(B):
        band = data[:, :, b]
        mean_val = np.mean(band)
        std_val = np.std(band)
        if std_val != 0:
            norm_data[:, :, b] = (band - mean_val) / std_val
    return norm_data

def estimate_snr_per_band(data):
    H, W, B = data.shape
    snr_values = np.zeros(B)
    for b in range(B):
        band = data[:, :, b].flatten()
        mean_val = np.mean(band)
        var_val = np.var(band)
        snr_values[b] = (mean_val ** 2) / var_val if var_val > 0 else 0
    return snr_values

def pca_reduce(data, n_components=3):
    H, W, B = data.shape
    reshaped = data.reshape(-1, B)
    centered = reshaped - np.mean(reshaped, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    reduced = centered @ eigvecs[:, idx[:n_components]]
    return reduced.reshape(H, W, n_components)

def kmeans_seeds(data, n_clusters=10, max_iter=100):
    H, W, C = data.shape
    X = data.reshape(-1, C)
    np.random.seed(0)
    centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(n_clusters)])
        if np.allclose(centers, new_centers): break
        centers = new_centers

    seeds = [divmod(np.argmin(np.linalg.norm(X - c, axis=1)), W) for c in centers]
    return seeds

def local_maxima_seeds(data, threshold=0.8, sigma=1.5, size=50):
    spectral_norm = np.linalg.norm(data, axis=2)
    smoothed = gaussian_filter(spectral_norm, sigma=sigma)
    neighborhood = maximum_filter(smoothed, size)
    maxima = (smoothed == neighborhood) & (smoothed > threshold * smoothed.max())
    return list(zip(*np.nonzero(maxima)))

def spectral_entropy(vec):
    norm = vec / (np.sum(vec) + 1e-6)
    return -np.sum(norm * np.log2(norm + 1e-6))

def entropy_seeds(data, top_n=20):
    H, W, B = data.shape
    entropy_map = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            entropy_map[y, x] = spectral_entropy(data[y, x, :])
    neighborhood = maximum_filter(entropy_map, 32)
    maxima = (entropy_map == neighborhood)
    return list(zip(*np.nonzero(maxima)))[:top_n]

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def spectral_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_theta, -1, 1))

def region_growing(image, seeds, similarity_func, threshold, use_pca=True):
    if use_pca: image = pca_reduce(image)
    H, W, D = image.shape
    visited = np.zeros((H, W), dtype=bool)
    regions = np.full((H, W), -1, dtype=int)
    region_id = 0

    for y, x in seeds:
        if visited[y, x]: continue
        queue = deque([(y, x)])
        visited[y, x] = True
        regions[y, x] = region_id
        seed_vec = image[y, x, :]

        while queue:
            cy, cx = queue.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        sim = similarity_func(seed_vec, image[ny, nx, :])
                        if sim < threshold:
                            visited[ny, nx] = True
                            regions[ny, nx] = region_id
                            queue.append((ny, nx))
        region_id += 1
    return regions

def region_growing_constrained(image, seeds, similarity_func, threshold, use_pca=True, max_region_size=10000):
    if use_pca: image = pca_reduce(image)
    H, W, D = image.shape
    visited = np.zeros((H, W), dtype=bool)
    regions = np.full((H, W), -1, dtype=int)
    region_id = 0

    for y, x in seeds:
        if visited[y, x]: continue
        queue = deque([(y, x)])
        visited[y, x] = True
        regions[y, x] = region_id
        seed_vec = image[y, x, :]
        region_size = 1

        while queue and region_size < max_region_size:
            cy, cx = queue.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        sim = similarity_func(seed_vec, image[ny, nx, :])
                        if sim < threshold:
                            visited[ny, nx] = True
                            regions[ny, nx] = region_id
                            queue.append((ny, nx))
                            region_size += 1
                            if region_size >= max_region_size:
                                break
        region_id += 1
    return regions

def plot_image(image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def plot_regions(region_map, title="Segmented Regions"):
    fig, ax = plt.subplots()
    n_labels = np.max(region_map) + 1
    cmap = mcolors.ListedColormap(plt.cm.hsv(np.linspace(0, 1, n_labels)))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_labels + 0.5), ncolors=n_labels)
    ax.imshow(region_map, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Region Growing Segmentation App")

uploaded_file = st.file_uploader("Upload .mat file containing hyperspectral image", type="mat")
if uploaded_file:
    mat = loadmat(uploaded_file)
    band_keys = [k for k in mat.keys() if not k.startswith('__')]
    data_key = st.selectbox("Select hyperspectral data key", band_keys)
    X = mat[data_key]

    st.subheader("Preprocessing Options")
    norm_type = st.radio("Normalization Type", ['Min-Max', 'Z-Score'])
    snr_threshold = st.slider("SNR Threshold for Band Selection", 0, 100, 20)

    snr = estimate_snr_per_band(X)
    good_band_indices = np.where(snr > snr_threshold)[0]
    X = X[:, :, good_band_indices]
    X = normalize_minmax(X) if norm_type == 'Min-Max' else normalize_zscore(X)

    st.subheader("Seed Selection Method")
    seed_method = st.selectbox("Choose Seed Selection", ['K-Means', 'Local Maxima', 'Spectral Entropy'])
    if seed_method == 'K-Means':
        n_clusters = st.slider("Number of Clusters", 1, 30, 10)
        seeds = kmeans_seeds(pca_reduce(X), n_clusters)
    elif seed_method == 'Local Maxima':
        threshold = st.slider("Local Maxima Threshold", 0.1, 1.0, 0.8)
        size = st.slider("Local Maxima Size", 5, 100, 50)
        seeds = local_maxima_seeds(X, threshold=threshold, size=size)
    else:
        top_n = st.slider("Top N Entropy Seeds", 5, 100, 20)
        seeds = entropy_seeds(X, top_n=top_n)

    band_index = st.slider("Band to Display", 0, X.shape[2] - 1, 0)
    plot_image(X[:, :, band_index], "Selected Band with Seeds")
    plot_image(np.linalg.norm(X, axis=2), "Spectral Norm")

    st.subheader("Region Growing Settings")
    method = st.radio("Region Growing Method", ['Standard', 'Constrained'])
    distance_metric = st.selectbox("Similarity Measure", ['Euclidean', 'Spectral Angle'])
    threshold = st.slider("Similarity Threshold", 0.0, 10.0, 0.5)
    use_pca = st.checkbox("Use PCA for Dimensionality Reduction", value=True)
    max_size = st.slider("Max Region Size (only for constrained)", 1000, 50000, 11000) if method == 'Constrained' else None

    sim_func = euclidean_distance if distance_metric == 'Euclidean' else spectral_angle
    result = region_growing_constrained(X, seeds, sim_func, threshold, use_pca, max_size) if method == 'Constrained' else region_growing(X, seeds, sim_func, threshold, use_pca)

    st.subheader("Segmentation Output")
    plot_regions(result, "Region Growing Output")

    # --- Ground Truth (Optional) ---
    st.subheader("Optional Ground Truth Upload and Comparison")
    gt_file = st.file_uploader("Upload Ground Truth (.mat)", type="mat", key="gt")
    if gt_file:
        gt_mat = loadmat(gt_file)
        gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]
        gt_key = st.selectbox("Select Ground Truth Key", gt_keys)
        ground_truth = gt_mat[gt_key]

        if ground_truth.shape != result.shape:
            st.error(f"Shape mismatch: Segmentation output shape {result.shape} vs Ground Truth shape {ground_truth.shape}")
        else:
            fig, ax = plt.subplots()
            n_labels_gt = np.max(ground_truth) + 1
            cmap_gt = mcolors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, n_labels_gt)))
            norm_gt = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_labels_gt + 0.5), ncolors=n_labels_gt)
            ax.imshow(ground_truth, cmap=cmap_gt, norm=norm_gt)
            ax.set_title("Ground Truth Labels")
            ax.axis('off')
            st.pyplot(fig)
