# FIFA ⚽

This is official code for our BMVC 2022 paper:<br>
[Fill in Fabrics: Body-Aware Self-Supervised Inpainting for Image-Based Virtual Try-On](https://arxiv.org/abs/2210.00918)
<br>

![attention](https://github.com/hasibzunair/fifa-tryon/blob/main/media/pipeline.png)

## 1. Specification of dependencies

This code requires Python 3.8.12. Run commands below to make your environment.
```python
conda create --name fifa python=3.8
conda activate fifa
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c conda-forge jupyterlab
pip install opencv-python, matplotlib, sklearn, tqdm, pycocotools, tensorboard, PyWavelets, tensorboardX
```
Or, do below to install the required packages.
```python
conda update conda
conda env create -f environment.yml
conda activate fifa 
```

## 2a. Training code

To train on VITON dataset, you will need to download [vgg19-dcbb9e9d.pth](https://github.com/hasibzunair/fifa-tryon/releases/download/v1.0-models/vgg19-dcbb9e9d.pth) and keep it inside the folder `train_src/models/`.

To train FIFA on VITON dataset, download the VITON training and test datasets from [here](https://github.com/hasibzunair/fifa-tryon/releases/download/v1.0-data/acgpn_data.zip). Make a folder named `datasets/acgpn_data` and put them there. Data directory tree will look like:

```
datasets/
    acgpn_data/
        try_on_training/
        try_on_testing/
```

VITON dataset is presented in VITON, containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.

Now, from your terminal, run the following to train FIFA in two stages. First, the Fabricator is trained using:
```python
python train_fabricator.py --name fabricator
```

Then, VTON pipeline is trained using
```python
python train.py --name fifa
```

A Colab training notebook is available with a subset of the dataset, see [notebook](https://github.com/hasibzunair/fifa-tryon/blob/main/train_src/notebooks/train_colab.ipynb).

After training, see `checkpoints` folder for the model weights.

## 2b. Evaluation code
To evaluate the performance of FIFA on the VITON test set, run `test.ipynb` inside the folder `test_src/notebooks` which shows visualizations as well as SSIM and FID scores.

SSIM and FID scores computed using [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

## 3. Pre-trained models

Pre-trained models are available in [GitHub Releases](https://github.com/hasibzunair/fifa-tryon/releases/tag/v1.0-models).

## 4. Demo

A hugging face spaces and colab demo is available [here](https://github.com/hasibzunair/fifa-demo).

## 5. Citation
```bibtex
@article{zunair2022fill,
  title={Fill in Fabrics: Body-Aware Self-Supervised Inpainting for Image-Based Virtual Try-On},
  author={Zunair, Hasib and Gobeil, Yan and Mercier, Samuel and Hamza, A Ben},
  journal={arXiv preprint arXiv:2210.00918},
  year={2022}
}
```

## Future works
* [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset was recently released and use by [HR-VITON](https://github.com/sangyun884/HR-VITON). One direction is to build Decathlon VTON dataset in the format of VITON-HD.
* Our approach uses human parsing (e.g. semantic segmentation) during inference only, a very recent method [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON) eliminates this step for faster inference.
* A possible approach to improve the method is to train the model with person images without background.

## Acknowledgements
This work is built on top of CVPR 2020 paper *Towards Photo-Realistic Virtual Try-On by Adaptively
Generating↔Preserving Image Content* ([Paper](https://arxiv.org/pdf/2003.05863.pdf), [Code](https://github.com/switchablenorms/DeepFashion_Try_On)). Thanks to the authors!