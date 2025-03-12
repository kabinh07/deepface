from deepface import DeepFace

def run_deepface_inference(model_name="Facenet512"):
    """ Run DeepFace model inference and measure resource utilization """
    img1 = "data/image_1.jpg"  # Change to your image path
    img2 = "data/image_2.jpg"  # Change to your image path

    print(f"Running DeepFace model: {model_name}...\n")

    # Run DeepFace model
    result = DeepFace.verify(img1, img2, model_name=model_name, detector_backend="retinaface", enforce_detection=False)

    # Print results
    print("\n=== DeepFace Results ===")
    print(result)

if __name__ == "__main__":
    run_deepface_inference()
