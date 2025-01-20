import os
import pandas as pd
import torch
from PIL import Image
import clip

# Folders
RENDERED_IMAGES_FOLDER = "/media/david/datasets/blender_renders/"  # Folder containing rendered images
OUTPUT_CSV = "/media/david/datasets/clip_features.csv"  # Path to save CSV file

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def parse_filename(filename):
    """
    Parse the filename to extract parameters.
    Expected format: {object_name}_theta_{theta}_phi_{phi}_light_{light}.png
    Example: chair_theta_30_phi_45_light_1.png
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    name = parts[0]  # Object name, e.g., chair-1
    object_class = name.split('-')[0]  # Extract class from name, e.g., chair
    print(parts)
    params = {
        "name": name,  # Full object name
        "class": object_class,  # Extracted class
        "theta": float(parts[4]),  # Extract theta value
        "phi": float(parts[6]),  # Extract phi value
        "light": int(parts[8]),  # Extract light configuration
    }
    return params

def extract_clip_features(image_path):
    """
    Extract CLIP features from an image.
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy().flatten()

def main():
    # Prepare a DataFrame to store results
    results = []

    # Iterate over images
    for root, _, files in os.walk(RENDERED_IMAGES_FOLDER):
        for file in files:
            if file.endswith(".png"):  # Process only PNG images
                image_path = os.path.join(root, file)
                print(f"Processing: {file}")

                # Extract parameters from the filename
                params = parse_filename(file)

                # Extract CLIP features
                clip_features = extract_clip_features(image_path)

                # Combine parameters, features, and image path
                params["CLIP_features"] = clip_features
                params["image_path"] = image_path
                results.append(params)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
