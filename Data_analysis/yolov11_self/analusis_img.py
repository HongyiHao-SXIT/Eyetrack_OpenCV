import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_statistics_plots(csv_path=r'C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\Data_analysis\evaluation_results.csv', save_dir='statistics_plots'):
    """
    Generate statistical images from evaluation results CSV.
    
    Parameters:
        csv_path: Path to the evaluation results CSV.
        save_dir: Directory to save the generated plots.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Read CSV data
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Data loaded: {csv_path}")
    print(df)
    
    # Extract data
    classes = df['Class'].tolist()
    if 'Global' in classes:
        global_idx = classes.index('Global')
        global_data = df.iloc[global_idx]
        class_data = df.iloc[:-1]  # Exclude the 'Global' row
    else:
        global_data = None
        class_data = df
    
    # 1. Plot class metrics bar chart
    plot_class_metrics(class_data, save_dir)
    
    # 2. Plot precision-recall curve
    if global_data is not None:
        plot_pr_curve(global_data, save_dir)
    
    # 3. Plot mAP comparison chart
    plot_map_comparison(class_data, save_dir)
    
    print(f"Statistics plots saved to: {os.path.abspath(save_dir)}")

def plot_class_metrics(data, save_dir):
    """Plot bar chart for class-wise metrics."""
    if data is None or len(data) == 0:
        return
    
    classes = data['Class'].tolist()
    precision = data['Precision'].tolist()
    recall = data['Recall'].tolist()
    map50 = data['mAP@.5'].tolist()
    map50_95 = data['mAP@.5:.95'].tolist()
    
    x = np.arange(len(classes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + 0.5*width, map50, width, label='mAP@.5', color='lightcoral')
    rects4 = ax.bar(x + 1.5*width, map50_95, width, label='mAP@.5:.95', color='orchid')
    
    # Add labels, title, and ticks
    ax.set_ylabel('Metric Value')
    ax.set_title('Class-wise Evaluation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='best')
    
    # Add data labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    
    fig.tight_layout()
    plt.savefig(f"{save_dir}/class_metrics_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(global_data, save_dir):
    """Plot precision-recall curve."""
    if global_data is None:
        return
    
    # Simulate PR curve data (replace with actual training data in practice)
    recall_values = np.linspace(0, global_data['Recall'], 50)
    precision_values = global_data['Precision'] * np.ones_like(recall_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, linewidth=2, color='blue')
    plt.fill_between(recall_values, precision_values, alpha=0.2, color='blue')
    
    # Add key point
    plt.scatter(global_data['Recall'], global_data['Precision'], s=100, color='red', label='Final Point')
    
    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR Curve)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.savefig(f"{save_dir}/pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_map_comparison(data, save_dir):
    """Plot mAP comparison chart."""
    if data is None or len(data) == 0:
        return
    
    classes = data['Class'].tolist()
    map50 = data['mAP@.5'].tolist()
    map50_95 = data['mAP@.5:.95'].tolist()
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, map50, width, label='mAP@.5', color='orange')
    rects2 = ax.bar(x + width/2, map50_95, width, label='mAP@.5:.95', color='purple')
    
    # Add labels, title, and ticks
    ax.set_ylabel('mAP Value')
    ax.set_title('mAP Metrics Comparison by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='best')
    
    # Add data labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    fig.tight_layout()
    plt.savefig(f"{save_dir}/map_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate statistics plots from the evaluation table
    generate_statistics_plots()