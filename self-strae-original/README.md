# Self-StrAE

Code for the Self-StrAE model as described in:

EMNLP 2023 - [StrAE: Autoencoding for Pre-Trained Embeddings using Explicit Structure](https://aclanthology.org/2023.emnlp-main.469/)

SemEval 2024 - [Self-StrAE at SemEval-2024 Task 1: Making Self-Structuring AutoEncoders Learn More With Less](https://aclanthology.org/2024.semeval-1.18/)


The code base mirrors that of [Banyan](https://github.com/exlab-research/Banyan) a significantly more performant version of this architecture described in our ICML 2025 paper [(available here)](https://arxiv.org/abs/2407.17771).



## Install Dependencies:
For quick setup on CPU you can just run:

`pip install -r requirements_cpu.txt`

Most likely, you'll want to use a GPU and getting DGL (one of the core packages) to install smoothly with pip is a little tricky. A much easier option is to create a conda environment and use that. To do so you can run:

`conda env create -f requirements.yaml`

Now you should be good to go! As a quick sanity check you can navigate to the src directory and run the following:

`python train.py --train_path ../data/small_train.txt --dev_path ../data/small_dev.txt --batch_size 64`

## Get Data + Tokenisers: 
Navigate to the scripts directory

We support the following languages:
- Afrikaans: af
- Amharic: am
- Arabic: ar
- English: en
- Spanish: es
- Hausa: ha
- Hindi: hi
- Indonesian: id
- Marathi: mr
- Telugu: te

Most of the test datasets are already in the data directory. However, to replicate the results in the paper you will need to get the pre-training corpora we used. Run the following to do so, you can edit the list in the file if you are only interested in a subset of languages:

`python get_data.py`

You will also want to download the relevant tokenisers from the BPEmb package (again edit the list if you only want a subset), to do so run:

`python get_tok.py`



## (Optional) Create Checkpoints Folder:
Before we get to training you might first want to create a checkpoint directory. If you would like to, navigate to the root and run 

`mkdir checkpoints` 

This won't be tracked by git if you call it checkpoints 

## Training: 
Navigate to the src directory.

By default the language is set to English, and hyperparameters follow the defaults in the paper. To train and English model (assuming you've followed the previous steps). Run the following:

`python train.py --save_path ../checkpoints/(name of your checkpoint).pt`

--save_path is an optional arg, and is not enabled by default

The default loss is contrastive. If you want to use the CECO loss (more performant) use:

python train.py --loss_type CECO --lr 1e-3

This is a higher learning rate than what's required for contrastive only training (1e-4 enabled by default). 


## Citation:

If you make use of this code or find it helpful, please consider citing our papers:

<pre> 
inproceedings{opper2023strae,
  title={StrAE: Autoencoding for Pre-Trained Embeddings using Explicit Structure},
  author={Opper, Mattia and Prokhorov, Victor and Siddharth, N},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={7544--7560},
  year={2023}
}
</pre>

<pre> 
@article{opper2024self,
  title={Self-StrAE at SemEval-2024 Task 1: Making Self-Structuring AutoEncoders Learn More With Less},
  author={Opper, Mattia and Siddharth, N},
  journal={arXiv preprint arXiv:2404.01860},
  year={2024}
}
</pre>

## License:

Self-StrAE is available under MIT License










