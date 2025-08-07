# SummerJobIFE
## By Sigurd Vargdal


## Setup:
We recommend using wsl and miniconda as environment for this repo.
Steps:
first fetch it from the url with **wget**:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Execute the installer, follow the steps in the installer:
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
Activate it:
```bash
source ~/.bashrc
```
Check:
```bash
conda --version
```

Fetch the repo:
```bash
git clone https://github.com/Tensorboy2/SummerJobIFE.git && cd SummerJobIFE
```



Dependencies:
```bash
pip install -r requirements.txt
```


Yang dataset can be downloaded at: [https://zenodo.org/records/10939100](https://zenodo.org/records/10939100)
Convert to **.pt** using 
```bash 
python3 src/data/utils/convert_tfrecors_pt.py
```
