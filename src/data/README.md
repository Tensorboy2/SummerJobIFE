# Data pipeline

## Steps:
1. Fetch Sentinel 2 images of PV systems (floating and land mounted). Two alternatives:
    - From Google earth engine, fetch observed coordinates of PV systems. Store as **.pt** for easy PyTorch training.
    - From Yang et al. download tfrecords in gz format (leads to limitations in model development).
2. Images must be labeled as PV or FPV and/or no PV. (Optional to add segmentation labels)
3. Once Images is labeled, then can be trained on classification and/or segmentation.

