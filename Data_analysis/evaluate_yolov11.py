import os
import yaml
from ultralytics import YOLO

# Solve OpenMP duplicate library loading issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def evaluate_model():
    # Model path (modify according to your setup)
    model_path = r'C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\YOLOv11\runs\train\train-200epoch-v11n.yaml\weights\best.pt'
    
    # Dataset configuration file path
    data_config = r'C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\data.yaml'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    if not os.path.exists(data_config):
        print(f"Error: Dataset config file '{data_config}' not found")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print(f"Using dataset config: {data_config}")
    results = model.val(
        data=data_config,
        imgsz=640,
        batch=32,
        save_json=True
    )
    
    # Print evaluation metrics (revised)
    print("\nEvaluation Results:")
    print(f"mAP@.5:.95: {results.box.map:.4f}")
    print(f"mAP@.5: {results.box.map50:.4f}")
    print(f"Mean Precision: {results.box.mp:.4f}")
    print(f"Mean Recall: {results.box.mr:.4f}")
    
    # Print per-class metrics
    print("\nClass-wise Metrics:")
    for i, cls_name in enumerate(results.names):
        if i < len(results.box.p):
            print(f"{cls_name}:")
            print(f"  Precision: {results.box.p[i]:.4f}")
            print(f"  Recall: {results.box.r[i]:.4f}")
            print(f"  mAP@.5: {results.box.ap50[i]:.4f}")
            print(f"  mAP@.5:.95: {results.box.ap[i]:.4f}")
    
    # Save evaluation results to table
    save_evaluation_table(results)

def save_evaluation_table(results):
    """Save evaluation results to a table"""
    import pandas as pd
    
    # Extract class metrics
    cls_metrics = []
    for i, cls_name in enumerate(results.names):
        if i < len(results.box.p):
            cls_metrics.append({
                'Class': cls_name,
                'Precision': results.box.p[i],
                'Recall': results.box.r[i],
                'mAP@.5': results.box.ap50[i],
                'mAP@.5:.95': results.box.ap[i]
            })
    
    # Add global metrics
    global_metrics = {
        'Class': 'Global',
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'mAP@.5': results.box.map50,
        'mAP@.5:.95': results.box.map
    }
    cls_metrics.append(global_metrics)
    
    # Create DataFrame and save
    if cls_metrics:
        df = pd.DataFrame(cls_metrics)
        table_path = 'evaluation_results.csv'
        df.to_csv(table_path, index=False)
        print(f"Evaluation table saved to: {os.path.abspath(table_path)}")

if __name__ == "__main__":
    evaluate_model()