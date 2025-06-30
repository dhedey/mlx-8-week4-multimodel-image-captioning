import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont
import argparse


def load_mnist_dataset(train=True):
    """Load the MNIST dataset."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ])
    
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")
    dataset = torchvision.datasets.MNIST(
        data_folder,
        download=True,
        transform=transform,
        train=train,
    )
    return dataset

random_transform = v2.Compose([
    v2.RandomResize(28, 40),
    v2.RandomRotation(30),
    v2.RandomResizedCrop(size = 28, scale = (28.0/40, 28.0/40)),
])

def nick_create_composite_image(dataset, num_digits=6, canvas_size=(256, 256), digit_size=28, show_individual=True, verbose=False):
    """
    Create a composite image with randomly positioned MNIST digits.
    
    Args:
        dataset: MNIST dataset
        num_digits: Number of digits to include
        canvas_size: Size of the output canvas (width, height)
        digit_size: Size of each digit (assumed square)
        show_individual: Whether to show/save individual images
    """
    
    # Randomly select digits from the dataset
    selected_indices = random.sample(range(len(dataset)), num_digits)
    selected_digits = []
    labels = []
    
    for idx in selected_indices:
        image, label = dataset[idx]
        image = random_transform(image)
        # Convert tensor to numpy array and scale to 0-255
        image_np = image.squeeze().numpy()  # Remove channel dimension
        image_np = (image_np * 255).astype(np.float32)
        selected_digits.append(image_np)
        labels.append(label)
    
    # Create canvas as numpy array (height, width) - black background
    canvas_array = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    
    # Create row-based positioning with randomness
    margin = digit_size // 2
    
    # Step 1: Create 3 rows at random y positions with vertical distance
    num_rows = 3
    row_margin = digit_size  # Minimum distance between rows
    available_height = canvas_size[1] - 2 * margin - (num_rows - 1) * row_margin
    row_height = available_height // num_rows
    
    # Check if we have enough space for at least one row
    if row_height < digit_size:
        if verbose:
            print(f"Warning: Canvas too small for proper row layout. Using simplified placement.")
        # Fallback to simple placement if canvas is too small
        placed_digits = []
        for i in range(min(num_digits, 3)):  # Place at most 3 digits
            x = margin + i * (canvas_size[0] - 2 * margin) // min(num_digits, 3)
            y = canvas_size[1] // 2
            if x + digit_size <= canvas_size[0] - margin and y + digit_size <= canvas_size[1] - margin:
                placed_digits.append((x, y, selected_digits[i], labels[i], 0))  # All in row 0 for fallback
    else:
        # Generate row centers with some randomness
        row_centers = []
        for i in range(num_rows):
            base_y = margin + i * (row_height + row_margin) + row_height // 2
            # Add some randomness to row position (but keep them separated)
            y_variation = min(row_height // 4, 20)  # Max 1/4 of row height or 20 pixels
            row_y = base_y + random.randint(-y_variation, y_variation)
            row_centers.append(row_y)
        
        # Step 2: Randomly distribute images onto rows (max 5 per row)
        max_per_row = 5
        row_assignments = []
        
        # Randomly assign all images to rows
        for i in range(num_digits):
            # Find rows that aren't full
            available_rows = [r for r in range(num_rows) if row_assignments.count(r) < max_per_row]
            if available_rows:
                row_assignments.append(random.choice(available_rows))
            else:
                # If all rows are full, just add to a random row (allowing overflow)
                row_assignments.append(random.randint(0, num_rows - 1))
        
        # Step 3: Place images along each row, only keeping successfully placed ones
        placed_digits = []  # List of (x, y, digit_array, label, row_idx) tuples
        
        # Group images by row
        images_per_row = [[] for _ in range(num_rows)]
        for i, row_idx in enumerate(row_assignments):
            images_per_row[row_idx].append(i)
        
        for row_idx, image_indices in enumerate(images_per_row):
            if not image_indices:
                continue
                
            row_center_y = row_centers[row_idx]
            num_in_row = len(image_indices)
            
            # Calculate x positions with spacing
            if num_in_row == 1:
                x_positions = [canvas_size[0] // 2]
            else:
                # Distribute across the width with margins
                available_width = canvas_size[0] - 2 * margin
                if available_width < digit_size * num_in_row:
                    # Not enough space for all digits in this row, skip some
                    max_fits = max(1, available_width // digit_size)
                    image_indices = image_indices[:max_fits]
                    num_in_row = len(image_indices)
                
                spacing = available_width // (num_in_row + 1)
                x_positions = [margin + spacing * (i + 1) for i in range(num_in_row)]
                
                # Add some randomness to x positions
                x_variation = min(spacing // 3, 30)  # Max 1/3 of spacing or 30 pixels
                x_positions = [x + random.randint(-x_variation, x_variation) 
                              for x in x_positions]
            
            # Assign positions to images in this row, validating each one
            for i, img_idx in enumerate(image_indices):
                x = max(margin, min(x_positions[i], canvas_size[0] - digit_size - margin))
                
                # Add y randomness around row center (but keep within row bounds)
                y_variation = min(row_height // 3, 15)  # Max 1/3 of row height or 15 pixels
                y = row_center_y + random.randint(-y_variation, y_variation)
                y = max(margin, min(y, canvas_size[1] - digit_size - margin))
                
                # Validate position is within canvas bounds
                if (x >= margin and y >= margin and 
                    x + digit_size <= canvas_size[0] - margin and 
                    y + digit_size <= canvas_size[1] - margin):
                    placed_digits.append((x, y, selected_digits[img_idx], labels[img_idx], row_idx))
                else:
                    if verbose:
                        print(f"Warning: Could not place digit {labels[img_idx]} at position ({x}, {y}), skipping.")
    
    # Sort placed digits by reading order (row by row, left to right)
    # This will determine the sequence label
    sorted_placed_digits = sorted(placed_digits, key=lambda item: (item[4], item[0]))  # Sort by row_idx, then x
    
    # Place digits on canvas by adding pixel values
    for x, y, digit_array, _, _ in sorted_placed_digits:  # Added extra _ for row_idx
        # Get the region bounds on the canvas
        x_end = min(x + digit_size, canvas_size[0])
        y_end = min(y + digit_size, canvas_size[1])
        
        # Get the corresponding region size in the digit image
        digit_width = x_end - x
        digit_height = y_end - y
        
        # Add the digit pixel values to the canvas region
        canvas_array[y:y_end, x:x_end] += digit_array[:digit_height, :digit_width]
    
    # Clip values to prevent overflow
    canvas_array = np.clip(canvas_array, 0, 255)

    # Generate sequence label based on sorted order of placed digits only (should be list of ints)
    sequence_label = [label for _, _, _, label, _ in sorted_placed_digits]  # Added extra _ for row_idx
    
    # Report if any digits were skipped
    placed_count = len(sorted_placed_digits)
    if placed_count < num_digits and verbose:
        print(f"Note: Only placed {placed_count} out of {num_digits} requested digits.")
    
    # Only show individual plot if requested
    if show_individual:
        plt.figure(figsize=(10, 8))
        plt.imshow(canvas_array, cmap='gray', vmin=0, vmax=255)
        plt.title(f'MNIST Composite Image\nSequence (top-left to bottom-right): {sequence_label}', 
                  fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print(f"Digit sequence: {sequence_label}")
    
    return canvas_array, sequence_label

def get_composite_image_and_sequence(dataset, min_digits = 1, max_digits = 6, canvas_size=(256, 256), digit_size=28):
    num_digits = random.randint(min_digits, max_digits)
    canvas, sequence = nick_create_composite_image(dataset, num_digits, canvas_size, digit_size, show_individual=False)
    return canvas, sequence

def generate_composite_batch(dataset, batch_size = 1, min_digits = 1, max_digits = 6, canvas_size=(256, 256), digit_size=28):
    """
    Generate a batch of composite images.

    Args:
        dataset: MNIST dataset
        min_digits: Minimum number of digits to include
        max_digits: Maximum number of digits to include
        canvas_size: Size of the output canvas (width, height)
        digit_size: Size of each digit (assumed square)
        show_individual: Whether to show/save individual images
    """
    canvas_list = []
    sequence_list = []
    for i in range(batch_size):
        num_digits = random.randint(min_digits, max_digits)
        canvas, sequence = nick_create_composite_image(
            dataset,
            num_digits,
            canvas_size,
            digit_size,
            show_individual=False,
        )
        canvas_list.append(canvas)
        sequence_list.append(sequence)
    return torch.tensor(canvas_list), torch.tensor(sequence_list)

def main():
    parser = argparse.ArgumentParser(description='Create composite MNIST images')
    parser.add_argument('--num_digits', type=int, default=6, 
                        help='Number of digits to include (default: 6)')
    parser.add_argument('--canvas_width', type=int, default=200, 
                        help='Canvas width in pixels (default: 200)')
    parser.add_argument('--canvas_height', type=int, default=200, 
                        help='Canvas height in pixels (default: 200)')
    parser.add_argument('--output', type=str, default='composite.png', 
                        help='Output filename (default: composite.png)')
    parser.add_argument('--count', type=int, default=1, 
                        help='Number of composite images to generate (default: 1)')
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading MNIST dataset...")
    dataset = load_mnist_dataset()
    print(f"Dataset loaded with {len(dataset)} images")
    
    # Generate composite images
    if args.count == 1:
        # Single image - show individually
        print(f"\nGenerating composite image...")
        canvas, sequence = nick_create_composite_image(
            dataset=dataset,
            num_digits=args.num_digits,
            canvas_size=(args.canvas_width, args.canvas_height),
            show_individual=True,
            verbose=True,
        )
        # Save single image
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Composite image saved as: {args.output}")
    else:
        # Multiple images - collect and display in grid
        images = []
        sequences = []
        
        print(f"\nGenerating {args.count} composite images...")
        for i in range(args.count):
            print(f"Creating image {i+1}/{args.count}...")
            canvas, sequence = nick_create_composite_image(
                dataset=dataset,
                num_digits=args.num_digits,
                canvas_size=(args.canvas_width, args.canvas_height),
                show_individual=False,
                verbose=True,
            )
            images.append(canvas)
            sequences.append(sequence)
        
        # Calculate grid dimensions
        cols = min(3, args.count)  # Max 3 columns
        rows = (args.count + cols - 1) // cols  # Ceiling division
        
        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if args.count == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Display images
        for i in range(args.count):
            axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f'Image {i+1}\nSequence: {sequences[i]}', fontsize=12)
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(args.count, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'MNIST Composite Images ({args.count} images)', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        # Save the grid
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Grid of {args.count} composite images saved as: {args.output}")


if __name__ == "__main__":
    main() 