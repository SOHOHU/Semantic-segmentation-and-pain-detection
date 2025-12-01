
import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from deeplab import DeeplabV3
from utils.utils_metrics import fast_hist, per_class_iu

def calculate_sensitivity():
    # Configuration
    VOCdevkit_path = 'VOCdevkit'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    num_classes = 8
    name_classes = ["_background_","face", "ear", "eye", "mouth", "nose", "iris", "saliva"]
    
    # Load model
    print("Loading model...")
    # Try to find the model file
    model_path = "model_data/deeplab_xception.pth"
    backbone = "xception"
    
    if not os.path.exists(model_path):
         print("Error: Model file not found")
         return

    # Initialize Deeplab with specific parameters
    deeplab = DeeplabV3(model_path=model_path, num_classes=8, backbone=backbone) 
    print(f"Model loaded from {model_path} with backbone {backbone}")

    results = []
    
    print(f"Processing {len(image_ids)} images...")
    for image_id in tqdm(image_ids):
        # Load Image
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        try:
            image = Image.open(image_path)
        except:
            print(f"Error opening image: {image_path}")
            continue
            
        # Predict
        pred_image = deeplab.get_miou_png(image)
        pred = np.array(pred_image)
        
        # Load GT
        gt_path = os.path.join(gt_dir, image_id + ".png")
        try:
            label = np.array(Image.open(gt_path))
        except:
            print(f"Error opening GT: {gt_path}")
            continue
            
        # Check shapes
        if len(label.flatten()) != len(pred.flatten()):
            print(f"Shape mismatch for {image_id}")
            continue
            
        # Calculate IoU for this image
        hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
        ious = per_class_iu(hist)
        # Calculate mean IoU for this image (ignoring NaNs)
        miou = np.nanmean(ious)
        
        results.append({
            'image_id': image_id,
            'miou': miou,
            'ious': ious.tolist()
        })

    # Calculate global average (Baseline)
    all_mious = [r['miou'] for r in results]
    avg_miou = np.mean(all_mious)
    print(f"Global Average mIoU: {avg_miou*100:.2f}%")
    
    # Save to CSV
    output_file = 'sensitivity_analysis_results.csv'
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['image_id', 'mIoU'] + [f'IoU_{c}' for c in name_classes]
        writer.writerow(header)
        
        for r in results:
            row = [r['image_id'], r['miou']] + r['ious']
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    calculate_sensitivity()
