import os
import shutil
from sklearn.model_selection import train_test_split

# Set the paths
original_dataset_path = "C:/Users/aryan/Documents/VSCode Python/478/project/256_ObjectCategories"  # Path to the original dataset
output_path = "C:/Users/aryan/Documents/VSCode Python/478/project/data2bUsed"  # Path for the split dataset

# Train-test split ratio
train_ratio = 0.8

# Create train and test directories
train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Iterate through each category folder
for category in os.listdir(original_dataset_path):
    category_path = os.path.join(original_dataset_path, category)
    if os.path.isdir(category_path):
        # List all images in the category folder
        images = os.listdir(category_path)
        
        # Split images into training and testing sets
        train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Create category subfolders in train and test directories
        train_category_path = os.path.join(train_path, category)
        test_category_path = os.path.join(test_path, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)
        
        # Move images to the train folder
        for img in train_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(train_category_path, img)
            shutil.copy(src, dst)
        
        # Move images to the test folder
        for img in test_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(test_category_path, img)
            shutil.copy(src, dst)

print("Dataset split into training and testing sets successfully!")
