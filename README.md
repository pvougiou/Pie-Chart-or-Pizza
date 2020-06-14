# Pie-Chart-or-Pizza

This repository is part of our work in which we seek to learn to identify images of charts as they are posted on social media. We are adapting the VGGNet architecture proposed by [Simonyan and Zisserman](https://arxiv.org/abs/1409.1556) to the requirements of our chart identification problem. We train our system on a dataset consisting of chart images from the [ReVision](https://dl.acm.org/citation.cfm?id=2047247) corpus and general-purpose images from [ILSVRC-2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

You can use one of our pre-trained models to identify whether an image on the web displays a data visualisation or not. In case it does, our model will try to predict the depicted chart type.

## DataTweet+
We used crowdsourcing to build a new realistic dataset of data visualisations, consisting of 3000 image tweets that have been posted by the Twitter accounts of some major news agencies. Each image in the corpus has been labelled with the relevant chart type (or types, where applicable).

In a Unix shell environment execute: `sh download_DataTweet.sh` in order to download and uncompress the resulting corpus in its corresponding folder (i.e. `DataTweet`).

## BibTeX
Please cite the following paper should you use this repository in your work.
```
@Article{Vougiouklis_Carr_Simperl_2020,
  author       = {Vougiouklis, Pavlos and Carr, Leslie and Simperl, Elena},
  journal      = {Proceedings of the International AAAI Conference on Web and Social Media},
  title        = {Pie Chart or Pizza: Identifying Chart Types and Their Virality on Twitter},
  year         = {2020},
  month        = {May},
  number       = {1},
  pages        = {694-704},
  volume       = {14},
  url          = {https://www.aaai.org/ojs/index.php/ICWSM/article/view/7335},
}
```

## License
This project is licensed under the terms of the Apache 2.0 License.
