def predict_single_image(image_path, model_path, index_to_label):
    try:
        # Load image
        with open(image_path, "rb") as f:
            content = f.read()
        img = load_image(content)
        if img is None:
            raise ValueError("Failed to load or preprocess image.")
        
        img = np.expand_dims(img, axis=0)  # Make it a batch of 1

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Predict
        pred = model.predict(img)
        pred_label_index = np.argmax(pred[0])
        pred_label = index_to_label[pred_label_index]

        print(f"Predicted label for '{image_path}': {pred_label}")
        return pred_label
    except Exception as e:
        print(f"Prediction error: {str(e)}", file=sys.stderr)
        return None

# Predict for a single image (for demonstration)
example_image = "/path/to/a/valid/image.jpg"  # Replace with actual path
predict_single_image(example_image, "simple_lung_cancer_model.h5", index_to_label)
