# Banyan

Code for the ICML 2025 paper - [Banyan: Improved Representation Learning with Explicit Structure](https://arxiv.org/abs/2407.17771). 

If you are just interested in the implementation of the model, it is defined in:

- src/models.py (contains the core Banyan code)
- src/funcs.py (contains the diagonal message passing functions)
- src/model_utils.py (contains helper functions for the running encoder portion of the model)


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

Finally, to run the retrieval eval for English, we need to get the relevant datasets from the BEIR package:

`python get_beir.py`

## (Optional) Create Checkpoints Folder:
Before we get to training you might first want to create a checkpoint directory. This will also be necessary to run the English classification and retrieval eval used in the paper. If you would like to, navigate to the root and run 

`mkdir checkpoints` 

This won't be tracked by git if you call it checkpoints 

## Training: 
Navigate to the src directory.

By default the language is set to English, and hyperparameters follow the defaults in the paper. To train and English model (assuming you've followed the previous steps). Run the following:

`python train.py --save_path ../checkpoints/(name of your checkpoint).pt`

--save_path is an optional arg, and is not enabled by default

Assuming that you have downloaded the appropriate corpora, you can train on any of the languages listed above by specifying its corresponding code using the --lang flag. For example, if you wanted to train a model on Afrikaans:

`python train.py --lang af`

Note that to replicate the results in the paper, we used a slightly lower number of epochs (8) because we found that the model converges to good sentence representations pretty quickly. So to fully reproduce things you should run:

`python train.py --lang af --epochs 8`

Otherwise we did no language specific hyperparameter tuning, and results can probably be significantly improved if you do so!

## Classification and Retrieval Eval (English):

Lexical and STS/STR evaluations will be run automatically during training. However, if you want to run our extra classification and retrieval evals for English you'll need a pre-trained model you saved somewhere. 

For retrieval eval, navigate to the src directory and use retrieval.py. The script expects you to specify a path to your saved model. So for example if you had saved it under en.pt in the checkpoints directory, you would run:

`python retrieval.py ../checkpoints/en.pt`

Similarly, for classification eval you can run:

`python classification.py ../checkpoints/en.pt` 

## Citation:

If you make use of this code or find it helpful, please consider citing our paper:

<pre> 
@article{opper2024banyan,
  title={Banyan: Improved Representation Learning with Explicit Structure},
  author={Opper, Mattia and Siddharth, N},
  journal={arXiv preprint arXiv:2407.17771},
  year={2024}
}
</pre>

## License:

Banyan is available under Apache 2.0 










