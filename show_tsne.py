import pandas as pd
import numpy as np
from e2i import EmbeddingsProjector

# Input CSV file
INPUT_CSV = "/media/david/datasets/clip_features.csv"

def parse_features(features_str):
    """
    Parse the CLIP features from a string representation.
    """
    return np.array([float(x) for x in features_str.strip("[]").split()])

def main():
    # Load the CSV
    df = pd.read_csv(INPUT_CSV)

    # Extract image URLs and feature vectors
    urls = df["image_path"].values  # Image paths
    vectors = np.array([parse_features(row) for row in df["CLIP_features"]])  # Feature vectors

    # Initialize EmbeddingsProjector
    image = EmbeddingsProjector()

    # Load URLs and vectors explicitly
    image.image_list = np.asarray(urls)
    image.data_vectors = np.asarray(vectors)

    # Calculate projection and create the image
    image.calculate_projection()
    image.create_image()

if __name__ == "__main__":
    main()
