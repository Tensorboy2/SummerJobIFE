import torch
import matplotlib.pyplot as plt




def plot_BCE(path=None,show=False):
    '''
    Plot of pre-training BCE loss over epochs
    '''
    loss = torch.load(path)
    train_loss = torch.tensor(loss['train']).numpy()
    val_loss = torch.tensor(loss['val']).numpy()

    plt.style.use('seaborn-v0_8-paper')
    figsize = (6, 4)
    plt.figure(figsize=figsize)

    plt.plot(train_loss, label='Train', color='tab:blue', linewidth=2)
    plt.plot(val_loss, label='Validation', color='tab:red', linewidth=2, linestyle='--')


    plt.xlabel('Epoch',fontsize=12)
    plt.ylabel('BCE', fontsize=12)

    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, which='both', linestyle=':', linewidth=0.6)
    plt.tight_layout()
    plt.savefig(path+'_plot.pdf')

    if show:
        plt.show()
    plt.close()

def plot_seg_metrics(path=None, show=False):
    '''
    Plot validation metrics (Precision, Recall, Dice, IoU) from segmentation training
    '''
    metrics = torch.load(path)

    # Set style and figure
    plt.style.use('seaborn-v0_8-paper')
    figsize = (6, 4)
    plt.figure(figsize=figsize)

    # Metric plotting with better colors, line styles
    # plt.plot(torch.tensor(metrics['precision']).numpy(), label='Precision', color='tab:blue', linewidth=2)
    # plt.plot(torch.tensor(metrics['recall']).numpy(), label='Recall', color='tab:orange', linewidth=2, linestyle='--')
    print(max(torch.tensor(metrics['dice']).numpy()))
    plt.plot(torch.tensor(metrics['dice']).numpy(), label='Dice', color='tab:green', linewidth=2, linestyle='-.')
    # plt.plot(torch.tensor(metrics['iou']).numpy(), label='IoU', color='tab:red', linewidth=2, linestyle=':')

    # Axis labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Validation Segmentation Metrics', fontsize=14)

    # Axis ticks and grid
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    # Legend
    plt.legend(frameon=False, fontsize=10)

    # Layout and save
    plt.tight_layout()
    plt.savefig(path + '_seg_metrics_plot.pdf')
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':

    path = 'loss_mae.pt'
    plot_BCE(path=path,show=False)

    path = 'loss_segmentation.pt'
    plot_BCE(path=path)

    path = 'validation_metrics_segmentation.pt'
    plot_seg_metrics(path=path)