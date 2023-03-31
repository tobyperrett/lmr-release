# Use Your Head: Improving Long-Tail Video Recognition

This repo contains implementations of the Long-tail Mixed Reconstruction (LMR) method from [Use Your Head: Improving Long-Tail Video Recognition](https://tobyperrett.github.io/lmr/use_your_head.pdf), to appear at CVPR 2023. 

It is forked from the [Motionformer](https://github.com/facebookresearch/Motionformer) codebase, so follow/cite that for installation instructions/dataset setup etc.. Alternatively, as the SlowFast codebase (which Motionformer is based on) is difficult to modify, if you want to implement LMR as part of your own method or do a comparison, there is also a simple demo for you to use. It just runs on the cpu with dummy data and no memory bank stuff, and it's here (remember implement the sampling strategies, described below in the models and in the paper):

```
tools/simple_example.py
```

# Models

LMR uses the cRT setup. First, instance-balanced sampling is used to train a model from scratch with standard cross-entropy training (i.e. no LMR). This is the initialisation. Next, the initialisation is fine-tuned under class-balanced sampling with LMR. We provide both the intialisations and the fine-tuned models:

[Link to motionformer checkpoints](https://www.dropbox.com/scl/fo/gekwzzdizgrmz5clacg9x/h?dl=0&rlkey=sxlm9z1nwodchfvhmzxmilv5x)

To run inference on the Epic Kitchens model, update the paths/num_gpus and batch sizes (we used 8x32GB GPUs) in the config yamls and check the slurm submit scripts, then run

```
sbatch run_single_node_test.sh /path/to/config /path/to/model
```

The "verb_lt" output is [average_class_acc, average_head_acc, average_tail_acc, average_few_shot_acc]. Similar for SSv2-LT.

# LT-Dataset splits

EPIC Kitchens just uses the default dataset as it is a natural long tail. We provide splits for SSv2-LT and VideoLT-LT here:

[Link to splits](https://www.dropbox.com/scl/fo/gekwzzdizgrmz5clacg9x/h?dl=0&rlkey=sxlm9z1nwodchfvhmzxmilv5x)

Note that the Framestack codebase is different from Motionformer and relies on pre-extracted ResNet features, so is not provided here.

# Citation

If you find this helpful, please cite our paper:

```BibTeX
@inproceedings{perrett2023,
   title={Use Your Head: Improving Long-Tail Video Recognition}, 
   author={Perrett, Toby and Sinha, Saptarshi and Burghardt, Tilo and Mirhemdi, Majid and Damen, Dima},
   year={2023},
   booktitle={Computer Vision and Pattern Recognition},
}
```

And also cite the [EPIC](https://epic-kitchens.github.io/), [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) and [VideoLT](https://videolt.github.io/) works.