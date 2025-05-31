from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StringType
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def save_classification_report(true_labels, pred_labels, index_to_label, filename="classification_report.png"):
    report = classification_report(true_labels, pred_labels, target_names=[index_to_label[i] for i in sorted(index_to_label.keys())])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    plt.text(0.1, 0.5, report, fontsize=12, va='center', ha='left')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(true_labels, pred_labels, index_to_label, filename="confusion_matrix.png"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[index_to_label[i] for i in sorted(index_to_label.keys())], yticklabels=[index_to_label[i] for i in sorted(index_to_label.keys())])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def predict_single_image_bytes(content, model_path, index_to_label):
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - np.mean(img)) / np.std(img)
        img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256, 3)

        model = tf.keras.models.load_model(model_path)
        pred = model.predict(img)
        pred_index = np.argmax(pred[0])
        return index_to_label[pred_index]
    except Exception as e:
        print(f"Error during single prediction: {str(e)}", file=sys.stderr)
        return None


def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("LungCancerClassification") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
   
    spark.sparkContext.setLogLevel("ERROR")



    # Load HDFS paths
    def get_labeled_paths(spark, base_dir):
        try:
            df = spark.read.format("binaryFile").load(f"{base_dir}/*/*")
            extract_label = udf(lambda path: path.split("/")[-2].replace(".", "_"), StringType())
            df = df.withColumn("label", extract_label(df["path"]))
            return df.select("content", "path", "label")
        except Exception as e:
            print(f"Error loading data from {base_dir}: {str(e)}", file=sys.stderr)
            return None

    # Load and preprocess image
    def load_image(content):
        try:
            img = Image.open(io.BytesIO(content)).convert('RGB')
            img = img.resize((256, 256))
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - np.mean(img)) / np.std(img)
            return img
        except Exception as e:
            print(f"Error loading image: {str(e)}", file=sys.stderr)
            return None

    # Process images in small batches
    def process_in_batches(df, batch_size=100):
        results = []
        for row in df.toLocalIterator():
            img = load_image(row["content"])
            if img is not None:
                results.append((img, row["label_index"], row["path"]))
            if len(results) >= batch_size:
                yield results
                results = []
        if results:
            yield results

    try:
        print("Loading datasets...")
        train_df = get_labeled_paths(spark, "hdfs://namenode:8020/datasets/train")
        valid_df = get_labeled_paths(spark, "hdfs://namenode:8020/datasets/valid")
        test_df  = get_labeled_paths(spark, "hdfs://namenode:8020/datasets/test")

        print("Showing train_df after useing get_labeled_paths this function:")
        train_df.printSchema()
        train_df.show(1)


        if not all([train_df, valid_df, test_df]):
            raise ValueError("Could not load one or more datasets")
        

        print("processing Label...")

        # Combine labels from all splits
        train_labels = train_df.select("label")
        valid_labels = valid_df.select("label")
        test_labels  = test_df.select("label")

        combined_labels = train_labels.union(valid_labels).union(test_labels).distinct()
        combined_labels.show()
        print("Showing combined_labels after union:")
        combined_labels.printSchema()
        # Collect all unique labels
        all_labels = combined_labels.rdd.flatMap(lambda x: x).collect()
        print("Showing all_labels after collect:")
        print(all_labels)
        label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        # Safe UDF (returns -1 if label not found)
        label_index = udf(lambda x: label_to_index.get(x, -1), IntegerType())
        print("Showing label_to_index mapping:", label_to_index)
        
        print("Showing train_df after using label_to_index:",index_to_label)
        print("Showing index_label mapping:", label_index)
     
        # Apply and reassign
        train_df = train_df.withColumn("label_index", label_index(col("label")))
        valid_df = valid_df.withColumn("label_index", label_index(col("label")))
        test_df  = test_df.withColumn("label_index", label_index(col("label")))
        print("Showing train_df:")
        train_df.printSchema()
       
        # Create training dataset
        print("Creating training dataset...")
        train_data = []
        for batch in process_in_batches(train_df.limit(500)):  
            train_data.extend(batch)

        if not train_data:
            raise ValueError("No valid training data found")

        train_images = np.stack([x[0] for x in train_data])
        train_labels = np.array([x[1] for x in train_data])

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds = train_ds.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

        print(f"\nTotal images loaded into train_data: {len(train_data)}")
        print(f"Total images loaded into train_ds: {len(train_ds)}")
        print(f"Total images loaded into train_images: {len(train_images)}")
        print(f"Total images loaded into train_labels: {len(train_labels)}")
        print(f"Total images loaded into train_df: {train_df.count()}")
        print(f"Total images loaded into valid_df: {valid_df.count()}")
        print(f"Total images loaded into test_df: {test_df.count()}")
 

        print("\n--- Inspecting train_ds ---")

        # Get one batch to inspect
        for batch_images, batch_labels in train_ds.take(1):
            print(f"Batch image shape : {batch_images.shape}")  # e.g., (32, 256, 256, 3)
            print(f"Batch label shape : {batch_labels.shape}")  # e.g., (32,)
            print(f"Batch image dtype : {batch_images.dtype}")
            print(f"Batch label dtype : {batch_labels.dtype}")
            print(f"First 5 labels    : {batch_labels.numpy()[:5]}")
            break



        # Build and train model
        print("Training model...")
        model = models.Sequential([
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(label_to_index), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_ds, epochs=10, verbose=1)
        # Plot training accuracy and loss
        plt.figure(figsize=(10, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_metrics.png")  # Save to file
        plt.close()


        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

        # Save the model
        model.save("simple_lung_cancer_model.h5")
        print("Model saved as 'simple_lung_cancer_model.h5'")

        # --- PREDICTION & EVALUATION PART ---
        print("Evaluating on test dataset...")
        test_data = []
        for batch in process_in_batches(test_df.limit(200)):
            test_data.extend(batch)

        if not test_data:
            raise ValueError("No test images available")

        test_images = np.stack([x[0] for x in test_data])
        true_labels = np.array([x[1] for x in test_data])
        paths = [x[2] for x in test_data]

        predictions = model.predict(test_images)
        pred_labels = np.argmax(predictions, axis=1)

        # Generate report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=[index_to_label[i] for i in sorted(index_to_label.keys())]))
        save_classification_report(true_labels, pred_labels, index_to_label, filename="classification_report.png")


        print("Confusion Matrix:")
        print(confusion_matrix(true_labels, pred_labels))
        save_confusion_matrix(true_labels, pred_labels, index_to_label, filename="confusion_matrix.png")

        # Save predictions
        print("Saving predictions to CSV...")
        df_results = pd.DataFrame({
            "image_path": paths,
            "true_label": [index_to_label[l] for l in true_labels],
            "predicted_label": [index_to_label[l] for l in pred_labels]
        })

        df_results.to_csv("predictions.csv", index=False)
        print("Predictions saved to 'predictions.csv'")

        #Predict a single image from valid_df ---
        print("\nPredicting a single image from validation set...")
        single_row = valid_df.limit(1).collect()[0]
        single_content = single_row["content"]
        single_path = single_row["path"]
        true_label_index = single_row["label_index"]
        true_label = index_to_label[true_label_index]

        predicted_label = predict_single_image_bytes(single_content, "simple_lung_cancer_model.h5", index_to_label)

        if predicted_label:
            print(f"Image: {single_path}")
            print(f"True Label: {true_label}")
            print(f"Predicted Label: {predicted_label}")
        else:
            print("Failed to predict label for validation image.")

        print("\nPredicting a single image from test set...")
        single_row = test_df.limit(1).collect()[0]
        single_content = single_row["content"]
        single_path = single_row["path"]
        true_label_index = single_row["label_index"]
        true_label = index_to_label[true_label_index]

        predicted_label = predict_single_image_bytes(single_content, "simple_lung_cancer_model.h5", index_to_label)

        if predicted_label:
            print(f"Image: {single_path}")
            print(f"True Label: {true_label}")
            print(f"Predicted Label: {predicted_label}")
        else:
            print("Failed to predict label for validation image.")


        return 0
    

    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        return 1

    finally:
        spark.stop()

if __name__ == "__main__":
    sys.exit(main())
