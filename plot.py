
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

    # Melt for seaborn
    loss_df = df.melt(id_vars=['epoch', 'model'], value_vars=['train_loss', 'val_loss'],
                     var_name='type', value_name='loss')
    iou_df = df.melt(id_vars=['epoch', 'model'], value_vars=['train_iou', 'val_iou'],
                    var_name='type', value_name='iou')

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=loss_df, x='epoch', y='loss', hue='model', style='type')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    sns.lineplot(data=iou_df, x='epoch', y='iou', hue='model', style='type')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: add more files as needed
    files = [
        'convnextv2_metrics.csv',
        'convnextv2_full_metrics.csv',
        'unet_metrics.csv',  # Uncomment/add more as needed
    ]
    plot_training_history(files)