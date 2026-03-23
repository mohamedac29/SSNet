import os
import shutil
import random

# Source directories
src_images_dir = '/home/usr/SSNet/data/DeepCrack/images'
src_masks_dir = '/home/usr/SSNet/data/DeepCrack/masks'

# Print the number of training images and labels
print(f"Number of training images: {len(os.listdir(src_images_dir))}")
print(f"Number of training labels: {len(os.listdir(src_masks_dir))}")

# Destination root directory
dst_root = '/home/usr/SSNet/data/DeepCrack'

# Subdirectories for train, validation, and test sets
train_dir = os.path.join(dst_root, 'train')
val_dir = os.path.join(dst_root, 'val')
test_dir = os.path.join(dst_root, 'test')

# Create destination directories if they don't exist
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'masks'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'masks'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'masks'), exist_ok=True)

# List image and mask files
images = sorted([f for f in os.listdir(src_images_dir) if (f.endswith('.png') or f.endswith('.jpg'))])
masks = sorted([f for f in os.listdir(src_masks_dir) if (f.endswith('.png') or f.endswith('.jpg'))])
assert len(images) == len(masks), "Number of images and masks do not match."

# Combine images and masks into pairs
data = list(zip(images, masks))

# Shuffle the data
random.seed(42)  # For reproducibility
random.shuffle(data)

# Split the data
train_data = data[:300]
val_data = data[300:400]
test_data = data[400:]

def copy_and_rename_files(file_list, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, prefix):
    img_paths = []
    mask_paths = []
    for idx, (img_file, mask_file) in enumerate(file_list, 1):
        # Copy and rename images
        src_img_path = os.path.join(src_img_dir, img_file)
        new_img_name = f"{prefix}_{idx}_image.png"
        dst_img_path = os.path.join(dst_img_dir, new_img_name)
        shutil.copy(src_img_path, dst_img_path)
        img_paths.append(dst_img_path)

        # Copy and rename masks
        src_mask_path = os.path.join(src_mask_dir, mask_file)
        new_mask_name = f"{prefix}_{idx}_mask.png"
        dst_mask_path = os.path.join(dst_mask_dir, new_mask_name)
        shutil.copy(src_mask_path, dst_mask_path)
        mask_paths.append(dst_mask_path)
    return img_paths, mask_paths

# Copy and rename train data
train_img_paths, train_mask_paths = copy_and_rename_files(
    train_data, src_images_dir, src_masks_dir,
    os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'), 'train'
)

# Copy and rename validation data
val_img_paths, val_mask_paths = copy_and_rename_files(
    val_data, src_images_dir, src_masks_dir,
    os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'), 'val'
)

test_img_paths, test_mask_paths = copy_and_rename_files(
    test_data, src_images_dir, src_masks_dir,
    os.path.join(test_dir, 'images'), os.path.join(test_dir, 'masks'), 'test'
)

# Write paths to .txt files
def write_paths_to_file(file_paths, file_name):
    with open(file_name, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')

# Write train paths
write_paths_to_file(train_img_paths + train_mask_paths, os.path.join(dst_root, 'deepcrack_train.txt'))

# Write validation paths
write_paths_to_file(val_img_paths + val_mask_paths, os.path.join(dst_root, 'deepcrack_val.txt'))

# Write validation paths
write_paths_to_file(test_img_paths + test_mask_paths, os.path.join(dst_root, 'deepcrack_test.txt'))

print("Files have been copied, renamed, and paths have been written to train.txt and val.txt")