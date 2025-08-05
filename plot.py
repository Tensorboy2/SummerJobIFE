
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_history(history_files):
    """
    Plots the training and validation loss and IoU from one or more CSV files using seaborn.
    Args:
        history_files (list of str): List of CSV file paths containing training history.
    """
    all_histories = []
    for file in history_files:
        history = pd.read_csv(file)
        # Try to infer model name from file name
        model_name = os.path.basename(file).replace('_metrics.csv', '')
        history['model'] = model_name
        all_histories.append(history)
    df = pd.concat(all_histories, ignore_index=True)

    # Find all metric pairs (train_xxx, val_xxx)
    metrics = set()
    for col in df.columns:
        if col.startswith('train_'):
            metric = col.replace('train_', '')
            if f'val_{metric}' in df.columns:
                metrics.add(metric)

    # Create plots folder if it doesn't exist
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Loss metrics (keep as is)
    loss_metrics = [m for m in metrics if m in ['loss', 'bce', 'dice']]
    for metric in loss_metrics:
        plt.figure(figsize=(8, 6))
        metric_df = df.melt(
            id_vars=['epoch', 'model'],
            value_vars=[f'train_{metric}', f'val_{metric}'],
            var_name='type', value_name=metric
        )
        metric_df['type'] = metric_df['type'].str.replace(f'_{metric}', '')
        sns.lineplot(data=metric_df, x='epoch', y=metric, hue='model', style='type')
        plt.title(f'Training and Validation {metric.capitalize()} Comparison')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.grid(True)
        plt.xticks(metric_df['epoch'].unique())
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f'{metric}_comparison.pdf')
        plt.savefig(out_path)
        plt.close()

    # Precision vs Recall plot per model
    for model in df['model'].unique():
        plt.figure(figsize=(8, 6))
        for split, marker in zip(['train', 'val'], ['o', 's']):
            plt.plot(df[df['model'] == model][f'{split}_recall'],
                     df[df['model'] == model][f'{split}_precision'],
                     label=f'{split.capitalize()}', marker=marker)
        plt.title(f'Precision vs Recall for {model}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f'{model}_precision_vs_recall.pdf')
        plt.savefig(out_path)
        plt.close()

    # TP, FP, TN, FN plot (all models, all splits)
    plt.figure(figsize=(10, 7))
    confusion_metrics = ['tp', 'fp', 'tn', 'fn']
    colors = ['b', 'r', 'g', 'm']
    for i, cm in enumerate(confusion_metrics):
        for model in df['model'].unique():
            for split, style in zip(['train', 'val'], ['-', '--']):
                label = f'{model} {split.upper()} {cm.upper()}'
                plt.plot(df[df['model'] == model][f'{split}_{cm}'],
                         label=label, color=colors[i], linestyle=style)
    plt.title('TP, FP, TN, FN over Epochs (All Models)')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(plot_dir, 'confusion_matrix_elements.pdf')
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    # Example usage: add more files as needed
    files = [
        'results/convnextv2_locked_metrics.csv',
        'results/convnextv2_open_metrics.csv',
        'results/unet_2_metrics.csv',  # Uncomment/add more as needed
    ]
    plot_training_history(files)