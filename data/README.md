# Data/ 

## Content
- **/dataset**
- **/gz_data**
- **/processed**
- **/utils**

## /dataset
The **dataset.py** script contains the PyTorch specialized dataset object for loading samples for segmentation.
The script contains a **get_dataloader** function, which returns train- and validation dataloader. The function requiers the confic to contain:
```python
config = {
    #...
    'val_ratio':0.2,
    'batch_size':16,
    #...
}
```

## /gz_data
The **gz_data** folder contains five datasets in **tfrecord.gz** format (Gziped TensorFlow records). These sets contain Sentinel-2 images of PV systems with a binary mask of solar panels for segmentation.

There is also **gz_data.py** for some visualization of these images in the different bands and mask.

## /utils
utils contains a conversion script **convert_tfrecors_pt.py**. This converts the **tfrecord.gz** files int PyTorch **.pt** format with a dictionary like structure. The name is **00000.pt**:
```python
{
    'image': torch.tensor, # (256,256,12)
    'mask': torch.tensor, # (256,256,1)
}
```

## /processed
**processed** contains the contains the results of **convert_tfrecors_pt.py**.


