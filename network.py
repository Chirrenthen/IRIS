import random
import numpy as np
from PIL import Image
import os
import time
import glob

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias1 = [0.0] * hidden_size
        self.weights2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias2 = [0.0] * output_size
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def forward(self, inputs):
        hidden = [self.sigmoid(sum(i * w for i, w in zip(inputs, weights)) + b) 
                  for weights, b in zip(self.weights1, self.bias1)]
        output = [self.sigmoid(sum(h * w for h, w in zip(hidden, weights)) + b) 
                  for weights, b in zip(self.weights2, self.bias2)]
        return output
    
    def predict(self, image_path):
        """Load image and predict digit"""
        try:
            image = Image.open(image_path).convert('L').resize((28, 28))
            inputs = np.array(image).flatten() / 255.0
            result = self.forward(inputs)
            return np.argmax(result)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def save_model(self, filepath='model_weights.npz'):
        """Save model weights to file"""
        np.savez_compressed(
            filepath,
            weights1=np.array(self.weights1),
            bias1=np.array(self.bias1),
            weights2=np.array(self.weights2),
            bias2=np.array(self.bias2)
        )
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model_weights.npz'):
        """Load model weights from file"""
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.weights1 = data['weights1'].astype(np.float64).tolist()
            self.bias1 = data['bias1'].astype(np.float64).tolist()
            self.weights2 = data['weights2'].astype(np.float64).tolist()
            self.bias2 = data['bias2'].astype(np.float64).tolist()
            print(f"Model loaded from {filepath}")
            return True
        return False


def retrain_from_feedback(nn, image_path, correct_label, learning_rate=0.5, iterations=3):
    """
    Retrain the network based on user feedback using backpropagation.
    Performs multiple training iterations for better learning.
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('L').resize((28, 28))
        inputs = np.array(image, dtype=np.float64).flatten() / 255.0

        pred_before = None
        
        # Perform multiple training iterations
        for iteration in range(iterations):
            # Convert to numpy arrays with explicit float64 dtype
            w1 = np.array(nn.weights1, dtype=np.float64)
            b1 = np.array(nn.bias1, dtype=np.float64)
            w2 = np.array(nn.weights2, dtype=np.float64)
            b2 = np.array(nn.bias2, dtype=np.float64)

            # Forward pass
            sigmoid = nn.sigmoid
            hidden_raw = w1.dot(inputs) + b1
            hidden = sigmoid(hidden_raw)
            out_raw = w2.dot(hidden) + b2
            output = sigmoid(out_raw)

            if iteration == 0:
                pred_before = int(np.argmax(output))

            # Create one-hot target
            target = np.zeros_like(output, dtype=np.float64)
            target[correct_label] = 1.0

            # Backpropagation
            error_out = target - output
            delta_out = error_out * output * (1.0 - output)

            # Update output layer
            w2 += learning_rate * np.outer(delta_out, hidden)
            b2 += learning_rate * delta_out

            # Update hidden layer
            error_hidden = w2.T.dot(delta_out)
            delta_hidden = error_hidden * hidden * (1.0 - hidden)
            w1 += learning_rate * np.outer(delta_hidden, inputs)
            b1 += learning_rate * delta_hidden

            # Save updated weights - ensure float type
            nn.weights1 = w1.tolist()
            nn.bias1 = b1.tolist()
            nn.weights2 = w2.tolist()
            nn.bias2 = b2.tolist()

        # Save training example to dataset
        save_to_dataset(inputs, correct_label, image)

        # Get new prediction after all iterations
        out_after = nn.forward(inputs)
        pred_after = int(np.argmax(out_after))

        return pred_before, pred_after

    except Exception as e:
        print(f"Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_to_dataset(inputs, label, image):
    """Save training example to dataset file"""
    dataset_path = 'training_dataset.npz'
    
    try:
        if os.path.exists(dataset_path):
            try:
                data = np.load(dataset_path)
                images = np.vstack([data['images'], inputs.reshape(1, -1)])
                labels = np.concatenate([data['labels'], np.array([label], dtype=np.int32)])
                data.close()
            except Exception as load_error:
                # If file is corrupted, start fresh
                print(f"Warning: Existing dataset corrupted, creating new one.")
                images = inputs.reshape(1, -1)
                labels = np.array([label], dtype=np.int32)
        else:
            images = inputs.reshape(1, -1)
            labels = np.array([label], dtype=np.int32)
        
        np.savez_compressed(dataset_path, images=images, labels=labels)
        
        # Also save as PNG for reference
        images_dir = 'dataset_images'
        os.makedirs(images_dir, exist_ok=True)
        filename = f"digit_{label}_{int(time.time()*1000)}.png"
        image.save(os.path.join(images_dir, filename))
        
    except Exception as e:
        print(f"Warning: Could not save to dataset: {e}")


def batch_train_from_dataset(nn, dataset_path='training_dataset.npz', epochs=5, learning_rate=0.1):
    """Train the network on saved dataset"""
    if not os.path.exists(dataset_path):
        print("No training dataset found.")
        return
    
    try:
        data = np.load(dataset_path)
        images = data['images']
        labels = data['labels']
        data.close()
        
        if len(labels) == 0:
            print("Dataset is empty!")
            return
        
        print(f"\nTraining on {len(labels)} examples for {epochs} epochs...")
        
        for epoch in range(epochs):
            correct = 0
            for img, label in zip(images, labels):
                # Convert to numpy arrays with proper dtype
                w1 = np.array(nn.weights1, dtype=np.float64)
                b1 = np.array(nn.bias1, dtype=np.float64)
                w2 = np.array(nn.weights2, dtype=np.float64)
                b2 = np.array(nn.bias2, dtype=np.float64)

                # Forward pass
                sigmoid = nn.sigmoid
                hidden_raw = w1.dot(img) + b1
                hidden = sigmoid(hidden_raw)
                out_raw = w2.dot(hidden) + b2
                output = sigmoid(out_raw)

                pred_before = int(np.argmax(output))

                # Create one-hot target
                target = np.zeros_like(output, dtype=np.float64)
                target[int(label)] = 1.0

                # Backpropagation
                error_out = target - output
                delta_out = error_out * output * (1.0 - output)

                # Update output layer
                w2 += learning_rate * np.outer(delta_out, hidden)
                b2 += learning_rate * delta_out

                # Update hidden layer
                error_hidden = w2.T.dot(delta_out)
                delta_hidden = error_hidden * hidden * (1.0 - hidden)
                w1 += learning_rate * np.outer(delta_hidden, img)
                b1 += learning_rate * delta_hidden

                # Save updated weights
                nn.weights1 = w1.tolist()
                nn.bias1 = b1.tolist()
                nn.weights2 = w2.tolist()
                nn.bias2 = b2.tolist()

                # Check prediction after update
                out_after = nn.forward(img)
                pred_after = int(np.argmax(out_after))
                
                if pred_after == int(label):
                    correct += 1
            
            accuracy = (correct / len(labels)) * 100
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.2f}%")
        
        print("Training complete!\n")
        
    except Exception as e:
        print(f"Error during batch training: {e}")
        import traceback
        traceback.print_exc()


def process_images_interactive(nn, image_folder, target_digit=None):
    """Process all images in folder with user feedback loop"""
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nFound {len(image_files)} images to process.\n")
    print("=" * 60)
    
    correct_count = 0
    trained_count = 0
    
    # AUTO-TRAINING MODE: If target_digit is specified
    if target_digit is not None:
        print(f"AUTO-TRAINING MODE: All images will be trained as digit '{target_digit}'")
        print("Processing images automatically...\n")
        
        for idx, img_path in enumerate(image_files, 1):
            filename = os.path.basename(img_path)
            
            # Get prediction
            prediction = nn.predict(img_path)
            
            if prediction is None:
                continue
            
            # Automatically train on this image
            pred_before, pred_after = retrain_from_feedback(nn, img_path, target_digit, learning_rate=0.5, iterations=5)
            
            # Check if correct after training
            if pred_after == target_digit:
                status = "✓"
                correct_count += 1
            else:
                status = "⚠"
            
            trained_count += 1
            
            # Show progress for each image
            print(f"[{idx}/{len(image_files)}] Trained: {trained_count} | Correct: {correct_count} | Accuracy: {(correct_count/trained_count*100):.1f}%")
        
        print("\n" + "=" * 60)
        print(f"AUTO-TRAINING COMPLETE!")
        print(f"Trained {trained_count} images as digit '{target_digit}'")
        print(f"Final Accuracy: {correct_count}/{trained_count} ({(correct_count/trained_count*100):.2f}%)")
        print("=" * 60)
        
        return correct_count, trained_count
    
    # MANUAL MODE: Ask for each image
    else:
        for idx, img_path in enumerate(image_files, 1):
            filename = os.path.basename(img_path)
            print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            
            # Get prediction
            prediction = nn.predict(img_path)
            
            if prediction is None:
                continue
            
            print(f"Neural Network Prediction: {prediction}")
            print("-" * 60)
            
            # Get user feedback
            while True:
                response = input("Is this correct? (y/n/skip/quit): ").strip().lower()
                
                if response == 'y' or response == 'yes':
                    correct_count += 1
                    print("✓ Correct!")
                    break
                
                elif response == 'n' or response == 'no':
                    correct_answer = input("Enter the correct digit (0-9): ").strip()
                    try:
                        correct_label = int(correct_answer)
                        if 0 <= correct_label <= 9:
                            print(f"Retraining with correct label: {correct_label}...")
                            pred_before, pred_after = retrain_from_feedback(nn, img_path, correct_label, learning_rate=0.5, iterations=5)
                            if pred_after == correct_label:
                                print(f"✓ Success! Prediction changed from {pred_before} to {pred_after}")
                            else:
                                print(f"⚠ Partial learning: Prediction changed from {pred_before} to {pred_after} (target was {correct_label})")
                            break
                        else:
                            print("Please enter a digit between 0-9")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                elif response == 'skip' or response == 's':
                    print("Skipped.")
                    break
                
                elif response == 'quit' or response == 'q':
                    print("\nExiting...")
                    return correct_count, idx
                
                else:
                    print("Invalid input. Please enter y/n/skip/quit")
        
        print("\n" + "=" * 60)
        print(f"Processing complete!")
        print(f"Accuracy: {correct_count}/{len(image_files)} ({(correct_count/len(image_files)*100):.2f}%)")
        print("=" * 60)
        
        return correct_count, len(image_files)


def main():
    """Main program loop"""
    print("\n" + "=" * 60)
    print("DIGIT RECOGNITION NEURAL NETWORK")
    print("=" * 60 + "\n")
    
    # Initialize network
    nn = SimpleNeuralNetwork(784, 128, 10)
    
    # Try to load existing model
    if nn.load_model():
        print("Loaded existing model.")
    else:
        print("Starting with new model.")
    
    while True:
        print("\n" + "-" * 60)
        print("MENU:")
        print("1. Process images from folder")
        print("2. Train on saved dataset")
        print("3. Test single image")
        print("4. Save model")
        print("5. Load model")
        print("6. Exit")
        print("-" * 60)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            # Ask for training mode
            print("\nSelect mode:")
            print("a. Auto-train (specify digit, train all images automatically)")
            print("b. Manual mode (review each image)")
            mode = input("Enter mode (a/b): ").strip().lower()
            
            if mode == 'a' or mode == 'auto':
                # Auto-training mode
                digit_input = input("Which digit are you training? (0-9): ").strip()
                try:
                    target_digit = int(digit_input)
                    if 0 <= target_digit <= 9:
                        folder_path = input("Enter folder path containing images: ").strip()
                        if os.path.isdir(folder_path):
                            process_images_interactive(nn, folder_path, target_digit=target_digit)
                        else:
                            print("Invalid folder path!")
                    else:
                        print("Please enter a digit between 0-9")
                except ValueError:
                    print("Invalid input!")
            
            elif mode == 'b' or mode == 'manual':
                # Manual mode
                folder_path = input("Enter folder path containing images: ").strip()
                if os.path.isdir(folder_path):
                    process_images_interactive(nn, folder_path, target_digit=None)
                else:
                    print("Invalid folder path!")
            else:
                print("Invalid mode selection!")
        
        elif choice == '2':
            epochs = input("Enter number of training epochs (default=5): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 5
            batch_train_from_dataset(nn, epochs=epochs)
        
        elif choice == '3':
            img_path = input("Enter image path: ").strip()
            if os.path.isfile(img_path):
                prediction = nn.predict(img_path)
                if prediction is not None:
                    print(f"\nPrediction: {prediction}")
                    feedback = input("Was this correct? (y/n): ").strip().lower()
                    if feedback == 'n' or feedback == 'no':
                        correct = input("Enter correct digit (0-9): ").strip()
                        try:
                            correct_label = int(correct)
                            if 0 <= correct_label <= 9:
                                pred_before, pred_after = retrain_from_feedback(nn, img_path, correct_label, learning_rate=0.5, iterations=5)
                                if pred_after == correct_label:
                                    print(f"✓ Network retrained successfully! ({pred_before} → {pred_after})")
                                else:
                                    print(f"⚠ Network updated but needs more training ({pred_before} → {pred_after}, target: {correct_label})")
                            else:
                                print("Invalid input!")
                        except ValueError:
                            print("Invalid input!")
            else:
                print("Invalid file path!")
        
        elif choice == '4':
            nn.save_model()
        
        elif choice == '5':
            nn.load_model()
        
        elif choice == '6':
            save = input("Save model before exiting? (y/n): ").strip().lower()
            if save == 'y' or save == 'yes':
                nn.save_model()
            print("\nThank you for using Digit Recognition Neural Network!")
            break
        
        else:
            print("Invalid choice! Please enter 1-6.")


if __name__ == "__main__":
    main()