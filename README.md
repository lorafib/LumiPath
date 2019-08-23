# LumiPath

Implementation of our MICCAI'19 paper "LumiPath - Towards Real-time Physically-based Rendering on Embedded Devices". The conference preprint can be accessed on arXiv here: https://arxiv.org/abs/1903.03837 .

## Overview
With the increasing computational power of today’s workstations, real-time physically-based rendering is within reach, rapidly
gaining attention across a variety of domains. These have expeditiously applied to medicine, where it is a powerful tool for intuitive 3D data visualization. Embedded devices such as optical see-through head-mounted displays (OST HMDs) have been a trend for medical augmented reality. However, leveraging the obvious benefits of physically-based rendering remains challenging on these devices because of limited computational power, memory usage, and power consumption. We navigate the compromise between device limitations and image quality to
achieve reasonable rendering results by introducing a novel light field that can be sampled in real-time on embedded devices. We demonstrate its applications in medicine and discuss limitations of the proposed method. 

Check out our [**demo video**](https://youtu.be/9a_nFwE29b0).

![Population of and rendering from the plenoptic function](https://raw.githubusercontent.com/lorafib/LumiPath/master/readme_images/LumiPath_Methods.jpg)

## Brief Method Overview
In order to allow for physically-based rendering on embedded devices, our prototype consists of a two-step algorithm. First, we compute all values of a reformulated plenoptic function and save the outcome as texture, which trades off hardware resources for rendering quality (see Fig. 1). Second, we transform the computationally expensive rendering task into a fast data query and interpolation task using this new representation (see Fig. 2). Additionally, we rely on a neural network that performs post-rendering correction in order to resolve
artifacts and vastly enhance image quality. Exemplary results are shown in Fig. 3.

![Exemplary results](https://raw.githubusercontent.com/lorafib/LumiPath/master/readme_images/LumiPath_Results.jpg)

## Reference

We hope this proves useful for augmented reality and visualization research in medicine and beyond. If you use our work, we would kindly ask you to reference our MICCAI article:
```
@inproceedings{LumiPath2019,
  author       = {Laura Fink and Sing Chun Lee and Jie Ying Wu and Xingtong Liu and Tianyu Song and Yordanka Stoyanova and Marc Stamminger and Nassir Navab and Mathias Unberath},
  title        = {{LumiPath--Towards Real-time Physically-based Rendering on Embedded Devices}},
  date         = {2019},
  booktitle    = {Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  publisher    = {Springer},
}
```
