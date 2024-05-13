# MIE_SIE_BSC
Source code of research article “Brain tumor segmentation in MRI with multi-modality spatial information enhancement and boundary shape correction”
If this code is helpful for your research, please cite "Zhiqin Zhu, Ziyu Wang, Guanqiu Qi, Neal Mazur, Pan Yang, Yu Liu, Brain tumor segmentation in MRI with multi-modality spatial information enhancement and boundary shape correction, Pattern Recognition Volume 153, September 2024, 110553".

@article{ZHU2024110553,
title = {Brain tumor segmentation in MRI with multi-modality spatial information enhancement and boundary shape correction},
journal = {Pattern Recognition},
volume = {153},
pages = {110553},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110553},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324003042},
author = {Zhiqin Zhu and Ziyu Wang and Guanqiu Qi and Neal Mazur and Pan Yang and Yu Liu},
keywords = {Brain tumor segmentation, Multi-modality MRI, Spatial information enhancement, Boundary shape correction},
abstract = {Brain tumor segmentation is currently of a priori guiding significance in medical research and clinical diagnosis. Brain tumor segmentation techniques can accurately partition different tumor areas on multi-modality images captured by magnetic resonance imaging (MRI). Due to the unpredictable pathological process of brain tumor generation and growth, brain tumor images often show irregular shapes and uneven internal gray levels. Existing neural network-based segmentation methods with an encoding/decoding structure can perform image segmentation to some extent. However, they ignore issues such as differences in multi-modality information, loss of spatial information, and under-utilization of boundary information, thereby limiting the further improvement of segmentation accuracy. This paper proposes a multimodal spatial information enhancement and boundary shape correction method consisting of a modality information extraction (MIE) module, a spatial information enhancement (SIE) module, and a boundary shape correction (BSC) module. The above three modules act on the input, backbone, and loss functions of deep convolutional networks (DCNN), respectively, and compose an end-to-end 3D brain tumor segmentation model. The three proposed modules can solve the low utilization rate of effective modality information, the insufficient spatial information acquisition ability, and the improper segmentation of key boundary positions can be solved. The proposed method was validated on BraTS2017, 2018, and 2019 datasets. Comparative experimental results confirmed the effectiveness and superiority of the proposed method over state-of-the-art segmentation methods.}
}
