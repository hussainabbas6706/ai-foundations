import numpy as np


def dot_product_manual(a: list[float], b: list[float]) -> float:
    """Calculate dot product using pure Python loops."""
    if len(a) != len(b):
        raise ValueError("Vectors must be of identical length.")
    return sum(x * y for x, y in zip(a, b))


def magnitude_manual(v: list[float]) -> float:
    """Calculate Euclidean norm (L2 norm) using pure Python."""
    return sum(x**2 for x in v) ** 0.5


def cosine_similarity_manual(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity without external libraries."""
    dot_prod = dot_product_manual(a, b)
    mag_a = magnitude_manual(a)
    mag_b = magnitude_manual(b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_prod / (mag_a * mag_b)


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Vectorized cosine similarity calculation using NumPy."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


if __name__ == "__main__":
    # Test sample vectors
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    py_sim = cosine_similarity_manual(v1, v2)

    np_v1 = np.array(v1)
    np_v2 = np.array(v2)
    np_sim = cosine_similarity_numpy(np_v1, np_v2)

    print(f"Pure Python Cosine Similarity: {py_sim:.6f}")
    print(f"NumPy Vectorized Cosine Similarity: {np_sim:.6f}")

    assert np.isclose(py_sim, np_sim), "Implementations do not match!"