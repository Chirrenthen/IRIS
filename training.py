import numpy as np
from PIL import Image
import os
import gzip
import urllib.request

def download_mnist():
    """Download MNIST dataset"""
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    print("Downloading MNIST dataset...")
    os.makedirs('mnist_data', exist_ok=True)
    
    for file in files:
        filepath = os.path.join('mnist_data', file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
    
    print("Download complete!\n")


def load_mnist_images(filename):
    """Load MNIST images from gz file"""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        
    return data


def load_mnist_labels(filename):
    """Load MNIST labels from gz file"""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def create_training_dataset(num_samples_per_digit=1000):
    """Create a balanced training dataset"""
    
    # Download if needed
    download_mnist()
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_images = load_mnist_images('mnist_data/train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels('mnist_data/train-labels-idx1-ubyte.gz')
    
    # Normalize images
    train_images = train_images.astype(np.float64) / 255.0
    
    print(f"Total MNIST images: {len(train_images)}\n")
    
    # Create balanced dataset
    selected_images = []
    selected_labels = []
    
    for digit in range(10):
        # Get indices for this digit
        digit_indices = np.where(train_labels == digit)[0]
        
        # Randomly select samples
        selected_indices = np.random.choice(digit_indices, 
                                           size=min(num_samples_per_digit, len(digit_indices)), 
                                           replace=False)
        
        selected_images.append(train_images[selected_indices])
        selected_labels.append(train_labels[selected_indices])
        
        print(f"Digit {digit}: Selected {len(selected_indices)} samples")
    
    # Combine all digits
    images = np.vstack(selected_images)
    labels = np.concatenate(selected_labels)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(images))
    images = images[shuffle_indices]
    labels = labels[shuffle_indices]
    
    print(f"\nTotal training samples: {len(images)}")
    
    # Save to npz file
    np.savez_compressed('training_dataset.npz', images=images, labels=labels)
    print("\n✓ Saved to 'training_dataset.npz'")
    
    # Also save some sample images for visual inspection
    save_sample_images(images, labels)
    
    return images, labels


def save_sample_images(images, labels, num_samples=5):
    """Save sample images for each digit"""
    print("\nSaving sample images...")
    os.makedirs('dataset_images', exist_ok=True)
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0][:num_samples]
        for idx in digit_indices:
            img_data = (images[idx] * 255).astype(np.uint8).reshape(28, 28)
            img = Image.fromarray(img_data, mode='L')
            img.save(f'dataset_images/sample_digit_{digit}_{idx}.png')
    
    print("✓ Sample images saved to 'dataset_images/' folder")


def main():
    print("=" * 60)
    print("MNIST DATASET CREATOR FOR NEURAL NETWORK")
    print("=" * 60 + "\n")
    
    # Ask user how many samples per digit
    while True:
        try:
            num_samples = input("How many samples per digit? (100-6000, recommended: 1000): ").strip()
            num_samples = int(num_samples) if num_samples else 1000
            
            if 100 <= num_samples <= 6000:
                break
            else:
                print("Please enter a number between 100 and 6000")
        except ValueError:
            print("Invalid input!")
    
    print(f"\nCreating dataset with {num_samples} samples per digit...")
    print("This will create a file with ~{} total training examples.\n".format(num_samples * 10))
    
    # Create the dataset
    images, labels = create_training_dataset(num_samples_per_digit=num_samples)
    
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run your neural network script (network.py)")
    print("2. Choose Menu Option 2: 'Train on saved dataset'")
    print("3. Train for 10-20 epochs for best results")
    print("\nThe model will learn to recognize handwritten digits!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()