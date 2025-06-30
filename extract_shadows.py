import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import datetime

shadow_dataset = "Dataset_Shadows_20250630_164014"
no_shadow_dataset = "Dataset_No_Shadows_20250630_153804"

shadow_images_dir = os.path.join(shadow_dataset, "images")
no_shadow_images_dir = os.path.join(no_shadow_dataset, "images")
shadow_metadata_dir = os.path.join(shadow_dataset, "metadata")
no_shadow_metadata_dir = os.path.join(no_shadow_dataset, "metadata")

output_dir = f"Extracted_Shadows_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_images_dir = os.path.join(output_dir, "images")
output_metadata_dir = os.path.join(output_dir, "metadata")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_metadata_dir, exist_ok=True)

if not all(os.path.exists(d) for d in [shadow_images_dir, no_shadow_images_dir]):
    print("Error: One or both dataset directories not found!")
    print(f"Shadow dataset: {shadow_dataset}")
    print(f"No-shadow dataset: {no_shadow_dataset}")
    exit(1)

shadow_images = sorted([f for f in os.listdir(shadow_images_dir) if f.endswith('.png')])
no_shadow_images = sorted([f for f in os.listdir(no_shadow_images_dir) if f.endswith('.png')])

print(f"Found {len(shadow_images)} shadow images")
print(f"Found {len(no_shadow_images)} no-shadow images")

common_images = []
for img in shadow_images:
    if img in no_shadow_images:
        common_images.append(img)

extraction_stats = {
    "total_pairs": len(common_images),
    "successful_extractions": 0,
    "failed_extractions": 0,
    "differences_detected": []
}

for img_name in tqdm(common_images, desc="extracting shadows from image pairs"):
    shadow_img_path = os.path.join(shadow_images_dir, img_name)
    no_shadow_img_path = os.path.join(no_shadow_images_dir, img_name)
    
    try:
        shadow_img = cv2.imread(shadow_img_path)
        no_shadow_img = cv2.imread(no_shadow_img_path)
        shadow_diff = cv2.subtract(no_shadow_img, shadow_img)  
        total_diff = np.sum(shadow_diff)
        max_diff = np.max(shadow_diff)
        mean_diff = np.mean(shadow_diff) 
        diff_stats = {
            "image": img_name,
            "total_difference": int(total_diff),
            "max_difference": int(max_diff),
            "mean_difference": float(mean_diff),
            "has_significant_shadows": bool(max_diff > 10)
        }
        extraction_stats["differences_detected"].append(diff_stats)
        
        visualization = shadow_img.copy()
        shadow_gray = cv2.cvtColor(shadow_diff, cv2.COLOR_BGR2GRAY)
        shadow_threshold = 15
        _, shadow_mask = cv2.threshold(shadow_gray, shadow_threshold, 255, cv2.THRESH_BINARY)
        overlay = visualization.copy()
        overlay[shadow_mask > 0] = [0, 0, 255] 
        alpha = 0.4  
        visualization = cv2.addWeighted(visualization, 1-alpha, overlay, alpha, 0)
        
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (0, 255, 255), 2)  # Yellow contours
        
        # Add text overlay with shadow statistics
        text_y = 30
        cv2.putText(visualization, f"Max Shadow: {max_diff}", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, f"Mean Shadow: {mean_diff:.1f}", (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, f"Shadow Areas: {len(contours)}", (10, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        base_name = os.path.splitext(img_name)[0]
        viz_name = f"{base_name}_shadow_viz.png"
        diff_name = f"{base_name}_shadow_diff.png"
        
        output_viz_path = os.path.join(output_images_dir, viz_name)
        output_diff_path = os.path.join(output_images_dir, diff_name)
        
        cv2.imwrite(output_viz_path, visualization)
        cv2.imwrite(output_diff_path, shadow_diff)
        
        diff_stats.update({
            "shadow_areas_count": len(contours),
            "shadow_threshold_used": shadow_threshold,
            "visualization_files": {
                "highlighted": viz_name,
                "difference": diff_name
            }
        })
        
        metadata_file = f"{base_name}_metadata.json"
        
        shadow_metadata_path = os.path.join(shadow_metadata_dir, metadata_file)
        output_metadata_path = os.path.join(output_metadata_dir, metadata_file)
        
        if os.path.exists(shadow_metadata_path):
            with open(shadow_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata["shadow_extraction"] = {
                "method": "cv2.subtract(no_shadow, shadow)",
                "source_datasets": {
                    "shadow": shadow_dataset,
                    "no_shadow": no_shadow_dataset
                },
                "source_images": {
                    "shadow": shadow_img_path,
                    "no_shadow": no_shadow_img_path
                },
                "difference_statistics": diff_stats,
                "extraction_timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(output_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        extraction_stats["successful_extractions"] += 1
        
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        extraction_stats["failed_extractions"] += 1

significant_shadows = [d for d in extraction_stats["differences_detected"] if d["has_significant_shadows"]]

dataset_info = {
    "extraction_info": {
        "source_datasets": {
            "shadow": shadow_dataset,
            "no_shadow": no_shadow_dataset
        },
        "method": "cv2.subtract(no_shadow, shadow)",
        "extraction_timestamp": datetime.datetime.now().isoformat(),
        "statistics": extraction_stats,
        "summary": {
            "total_frame_pairs": len(common_images),
            "successful_extractions": extraction_stats["successful_extractions"],
            "frames_with_significant_shadows": len(significant_shadows),
            "shadow_detection_rate": f"{len(significant_shadows)/len(common_images)*100:.1f}%" if common_images else "0%"
        }
    }
}

with open(os.path.join(output_dir, "extraction_info.json"), 'w') as f:
    json.dump(dataset_info, f, indent=2)

with open(os.path.join(output_dir, "shadow_statistics.json"), 'w') as f:
    json.dump(extraction_stats, f, indent=2)

]