# Install required libraries (for notebook execution only)
# !pip install kagglehub tensorflow opencv-python matplotlib seaborn scikit-learn tqdm wandb

import os  # For file and directory operations
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt  # For plotting graphs and images
import seaborn as sns  # For advanced statistical data visualization
import kagglehub  # For downloading datasets from Kaggle
import cv2  # For image processing
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # For evaluation metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image data augmentation and loading
from tensorflow.keras.models import Model, load_model  # For building and loading models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D  # For model layers
from tensorflow.keras.optimizers import Adam  # For optimizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard  # For training callbacks
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0  # For transfer learning models
import tensorflow as tf  # TensorFlow main library
from tqdm import tqdm  # For progress bars
import time  # For timing operations
import argparse  # For parsing command-line arguments
import datetime  # For timestamps
import pandas as pd  # For data manipulation and saving results
import matplotlib.cm as cm  # For color maps
# import wandb  # Uncomment for wandb integration

# Set up argument parser for command line usage
parser = argparse.ArgumentParser(description='Brain CT Classification')
parser.add_argument('--img_size', type=int, default=128, help='Image size')  # Image size (height, width)
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')  # Number of images per batch
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')  # Number of training epochs
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # Learning rate for optimizer
parser.add_argument('--model_type', type=str, default='custom', choices=['custom', 'resnet', 'efficientnet'], help='Model type')  # Model architecture
parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for tracking')  # Enable wandb
args = parser.parse_args()  # Parse command-line arguments

# Create directories for saving results
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Current timestamp for unique folder names
results_dir = f"results_{timestamp}"  # Results directory name
os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist
os.makedirs(os.path.join(results_dir, "model"), exist_ok=True)  # Directory for saving models
os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)  # Directory for plots
os.makedirs(os.path.join(results_dir, "misclassified"), exist_ok=True)  # Directory for misclassified images

# Wandb setup (optional)
if args.use_wandb:
    wandb.init(project="brain-ct-classification", name=f"{args.model_type}_{timestamp}")  # Initialize wandb run
    wandb.config.update(args)  # Log config

# Dataset setup
print("Setting up dataset...")
dataset_path = kagglehub.dataset_download("shuvokumarbasakbd/brain-ct-medical-imaging-colorized-dataset")  # Download dataset
base_path = os.path.join(dataset_path, "Computed Tomography (CT) of the Brain", "dataset")  # Base dataset path
train_dir = os.path.join(base_path, "train")  # Training data directory
test_dir = os.path.join(base_path, "test")  # Test data directory

# Parameters
img_size = (args.img_size, args.img_size)  # Image size tuple
batch_size = args.batch_size  # Batch size

# Data augmentation with more options
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.15,  # Shear transformation
    zoom_range=0.15,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    vertical_flip=False,  # No vertical flip (not suitable for medical images)
    brightness_range=[0.8, 1.2],  # Random brightness
    fill_mode='nearest',  # Fill mode for new pixels
    validation_split=0.2  # Split for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training'  # Training data
)
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation'  # Validation data
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for test data
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False  # No shuffling for test data
)

num_classes = len(train_generator.class_indices)  # Number of classes
class_labels = list(train_generator.class_indices.keys())  # List of class names
print(f"Found {num_classes} classes: {class_labels}")

# Model selection
def create_model(model_type='custom', img_size=(128, 128), num_classes=2):
    if model_type == 'resnet':
        # ResNet50V2 transfer learning model
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
        base_model.trainable = False  # Freeze base model layers
        
        inputs = Input(shape=(*img_size, 3))  # Input layer
        x = base_model(inputs)  # Pass input through base model
        x = GlobalAveragePooling2D()(x)  # Global average pooling
        x = Dense(512, activation='relu')(x)  # Dense layer
        x = BatchNormalization()(x)  # Batch normalization
        x = Dropout(0.5)(x)  # Dropout for regularization
        outputs = Dense(num_classes, activation='softmax')(x)  # Output layer
        model = Model(inputs, outputs)  # Build model
        
    elif model_type == 'efficientnet':
        # EfficientNetB0 transfer learning model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
        base_model.trainable = False  # Freeze base model layers
        
        inputs = Input(shape=(*img_size, 3))  # Input layer
        x = base_model(inputs)  # Pass input through base model
        x = GlobalAveragePooling2D()(x)  # Global average pooling
        x = Dense(256, activation='relu')(x)  # Dense layer
        x = BatchNormalization()(x)  # Batch normalization
        x = Dropout(0.5)(x)  # Dropout for regularization
        outputs = Dense(num_classes, activation='softmax')(x)  # Output layer
        model = Model(inputs, outputs)  # Build model
        
    else:  # custom model
        # Custom CNN architecture
        inputs = Input(shape=(*img_size, 3))  # Input layer
        x = Conv2D(32, (3,3), activation='relu', padding='same', name='conv1')(inputs)  # Conv layer 1
        x = BatchNormalization()(x)  # Batch normalization
        x = MaxPooling2D(2,2)(x)  # Max pooling
        
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2')(x)  # Conv layer 2
        x = BatchNormalization()(x)  # Batch normalization
        x = MaxPooling2D(2,2)(x)  # Max pooling
        
        x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv3')(x)  # Conv layer 3
        x = BatchNormalization()(x)  # Batch normalization
        x = MaxPooling2D(2,2)(x)  # Max pooling
        
        x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv4')(x)  # Conv layer 4
        x = BatchNormalization()(x)  # Batch normalization
        x = MaxPooling2D(2,2)(x)  # Max pooling
        
        x = Flatten()(x)  # Flatten feature maps
        x = Dense(512, activation='relu')(x)  # Dense layer
        x = Dropout(0.5)(x)  # Dropout for regularization
        x = Dense(256, activation='relu')(x)  # Dense layer
        x = Dropout(0.3)(x)  # Dropout for regularization
        outputs = Dense(num_classes, activation='softmax', name='output')(x)  # Output layer
        
        model = Model(inputs, outputs)  # Build model
    
    return model  # Return the model

# Create model
print(f"Creating {args.model_type} model...")
model = create_model(args.model_type, img_size, num_classes)  # Build model based on args
model.compile(
    optimizer=Adam(learning_rate=args.lr),  # Adam optimizer with specified learning rate
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]  # Metrics to monitor
)
model.summary()  # Print model summary

# Enhanced callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),  # Stop early if no improvement
    ModelCheckpoint(os.path.join(results_dir, 'model', 'best_model.h5'), save_best_only=True, monitor='val_loss'),  # Save best model
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),  # Reduce learning rate on plateau
    TensorBoard(log_dir=os.path.join(results_dir, 'logs'))  # TensorBoard logging
]

# Add WandB callback if enabled
# if args.use_wandb:
#    callbacks.append(wandb.keras.WandbCallback())

# Train
print("Starting training...")
start_time = time.time()  # Record start time

history = model.fit(
    train_generator,  # Training data
    validation_data=val_generator,  # Validation data
    epochs=args.epochs,  # Number of epochs
    callbacks=callbacks,  # Callbacks
    verbose=1  # Verbosity
)

training_time = time.time() - start_time  # Calculate training duration
print(f"Training completed in {training_time:.2f} seconds")

# Save training history
hist_df = pd.DataFrame(history.history)  # Convert history to DataFrame
hist_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)  # Save as CSV

# Plot training & validation curves
plt.figure(figsize=(12, 8))  # Set figure size
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])  # Training accuracy
plt.plot(history.history['val_accuracy'])  # Validation accuracy
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])  # Training loss
plt.plot(history.history['val_loss'])  # Validation loss
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'])  # Training precision
plt.plot(history.history['val_precision'])  # Validation precision
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(2, 2, 4)
plt.plot(history.history['recall'])  # Training recall
plt.plot(history.history['val_recall'])  # Validation recall
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()  # Adjust layout
plt.savefig(os.path.join(results_dir, 'visualizations', 'training_curves.png'))  # Save plot
plt.show()  # Show plot

# Load best model for evaluation
best_model = load_model(os.path.join(results_dir, 'model', 'best_model.h5'))  # Load best saved model

# Evaluation
print("Evaluating model on test set...")
test_results = best_model.evaluate(test_generator, verbose=1)  # Evaluate on test data
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")

# Predictions
pred_probs = best_model.predict(test_generator)  # Get predicted probabilities
pred_classes = np.argmax(pred_probs, axis=1)  # Get predicted class indices
true_classes = test_generator.classes  # True class indices

# Save results to file
with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
    f.write(f"Test Loss: {test_results[0]:.4f}\n")
    f.write(f"Test Accuracy: {test_results[1]:.4f}\n")
    f.write(f"Test Precision: {test_results[2]:.4f}\n")
    f.write(f"Test Recall: {test_results[3]:.4f}\n")

# Classification Report
report = classification_report(true_classes, pred_classes, target_names=class_labels)  # Generate report
print(report)
with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)  # Compute confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)  # Plot heatmap
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, 'visualizations', 'confusion_matrix.png'))  # Save plot
plt.show()

# ROC Curve and AUC (for multi-class using one-vs-rest)
plt.figure(figsize=(12, 10))
for i, class_name in enumerate(class_labels):
    # Create one-hot encoding for the current class
    true_class = np.zeros((len(true_classes), num_classes))
    for j in range(len(true_classes)):
        true_class[j, true_classes[j]] = 1
    
    fpr, tpr, _ = roc_curve(true_class[:, i], pred_probs[:, i])  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC
    
    plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')  # Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'visualizations', 'roc_curve.png'))  # Save plot
plt.show()

# Grad-CAM Visualization function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )  # Model for gradients

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)  # Forward pass
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])  # Use top predicted class if not specified
        class_channel = predictions[:, pred_index]  # Class score

    grads = tape.gradient(class_channel, conv_outputs)  # Compute gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Average gradients
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # Weighted sum
    heatmap = tf.squeeze(heatmap)  # Remove extra dimensions
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Normalize
    return heatmap.numpy()  # Return as numpy array

# Function to display prediction with confidence
def display_prediction(img_path, model, class_labels, img_size=(128, 128), last_conv_layer='conv3'):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)  # Load image
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and add batch dimension
    
    # Make prediction
    preds = model.predict(img_array)  # Predict probabilities
    pred_class = np.argmax(preds[0])  # Predicted class index
    confidence = preds[0][pred_class] * 100  # Confidence in percent
    
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_class)
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    heatmap = cv2.resize(heatmap, img_size)  # Resize to image size
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
    
    # Superimpose heatmap on original image
    orig_img = np.uint8(img)  # Convert PIL image to uint8 numpy array
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)  # Overlay heatmap
    
    # Display image and prediction
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    # Create bar chart of confidence scores for all classes
    bars = plt.bar(range(len(class_labels)), preds[0] * 100)
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.ylim(0, 100)
    plt.title("Confidence Scores (%)")
    
    # Highlight the predicted class
    bars[pred_class].set_color('red')
    
    plt.suptitle(f"Prediction: {class_labels[pred_class]} ({confidence:.2f}% confidence)")
    plt.tight_layout()
    return superimposed_img, preds[0]

# Find misclassified images
misclassified_indices = np.where(pred_classes != true_classes)[0]  # Indices where prediction != true label
print(f"Found {len(misclassified_indices)} misclassified images")

# Save and visualize misclassified images
if len(misclassified_indices) > 0:
    num_to_show = min(10, len(misclassified_indices))  # Show up to 10
    for i, idx in enumerate(misclassified_indices[:num_to_show]):
        img_path = os.path.join(test_dir, test_generator.filenames[idx])  # Path to misclassified image
        true_label = class_labels[true_classes[idx]]  # True label
        pred_label = class_labels[pred_classes[idx]]  # Predicted label
        
        # Display prediction
        superimposed_img, probs = display_prediction(
            img_path, 
            best_model, 
            class_labels, 
            img_size,
            'conv4' if args.model_type == 'custom' else 'top_activation'
        )
        plt.savefig(os.path.join(results_dir, 'misclassified', f'misclassified_{i}.png'))  # Save visualization
        plt.show()

# Visualize a sample of correctly classified images
correctly_classified_indices = np.where(pred_classes == true_classes)[0]  # Indices where prediction == true label
if len(correctly_classified_indices) > 0:
    num_to_show = min(5, len(correctly_classified_indices))  # Show up to 5
    sample_indices = np.random.choice(correctly_classified_indices, num_to_show, replace=False)  # Random sample
    
    for i, idx in enumerate(sample_indices):
        img_path = os.path.join(test_dir, test_generator.filenames[idx])  # Path to image
        display_prediction(
            img_path, 
            best_model, 
            class_labels, 
            img_size,
            'conv4' if args.model_type == 'custom' else 'top_activation'
        )
        plt.savefig(os.path.join(results_dir, 'visualizations', f'correct_pred_{i}.png'))  # Save visualization
        plt.show()

print(f"Analysis complete. Results saved to {results_dir}")
