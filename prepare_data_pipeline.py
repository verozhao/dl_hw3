import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, './datasets')
from voc import VOCSegmentation

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def analyze_dataset():
    train_dst = VOCSegmentation(root='./datasets', image_set='train', transform=None)
    
    num_classes = len(VOC_CLASSES)
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    
    for idx in range(len(train_dst)):
        mask = np.array(Image.open(train_dst.masks[idx]))
        for cls in np.unique(mask):
            if cls < num_classes:
                class_pixel_counts[cls] += np.sum(mask == cls)
    
    total_pixels = np.sum(class_pixel_counts)
    
    print(f"Number of categories: {num_classes}")
    print("\nClass distribution:")
    for i in range(num_classes):
        print(f"{VOC_CLASSES[i]:15s}: {class_pixel_counts[i]/total_pixels*100:5.2f}%")
    
    return train_dst

def visualize_samples(train_dst):
    class_samples = {}
    
    for idx in range(len(train_dst)):
        mask = np.array(Image.open(train_dst.masks[idx]))
        for cls in np.unique(mask):
            if cls < len(VOC_CLASSES) and cls not in class_samples:
                if np.sum(mask == cls) > mask.size * 0.02:
                    class_samples[cls] = idx
        if len(class_samples) == len(VOC_CLASSES):
            break
    
    fig, axes = plt.subplots(5, 10, figsize=(30, 15))
    axes = axes.flatten()
    
    plot_idx = 0
    for cls_idx in range(len(VOC_CLASSES)):
        if cls_idx in class_samples:
            img = Image.open(train_dst.images[class_samples[cls_idx]]).convert('RGB')
            mask = np.array(Image.open(train_dst.masks[class_samples[cls_idx]]))
            
            axes[plot_idx].imshow(img)
            axes[plot_idx].set_title(VOC_CLASSES[cls_idx])
            axes[plot_idx].axis('off')
            plot_idx += 1
            
            axes[plot_idx].imshow(train_dst.decode_target(mask))
            axes[plot_idx].set_title(f'{VOC_CLASSES[cls_idx]} (mask)')
            axes[plot_idx].axis('off')
            plot_idx += 1
    
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('./problem1_visualization.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    train_dst = analyze_dataset()
    visualize_samples(train_dst)