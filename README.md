# GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting

[Haodong Chen](https://haroldchen19.github.io/)<sup>😎</sup>, [Yongle Huang](https://github.com/KyleHuang9)<sup>😎</sup>, [Haojian Huang](https://github.com/JethroJames)<sup>🥳</sup>, Xiangsheng Ge<sup>😎</sup>, [Dian Shao](https://scholar.google.com/citations?hl=en&user=amxDSLoAAAAJ&view_op=list_works&sortby=pubdate)<sup>😎🤩</sup>

<sup>😎</sup>Northwestern Polytechnical University, <sup>🥳</sup>The University of Hong Kong; <sup>🤩</sup>Corresponding Author

<p align="center">
  <a href='https://arxiv.org/abs/2405.07472'>
  <img src='https://img.shields.io/badge/Arxiv-2405.07472-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://arxiv.org/pdf/2405.07472'>
  <img src='https://img.shields.io/badge/Paper-PDF-purple?style=flat&logo=arXiv&logoColor=yellow'></a> 
  <a href='https://haroldchen19.github.io/gsvton/'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
</p>

## Abstract

> The increasing prominence of e-commerce has underscored the importance of Virtual Try-On (VTON).
However, previous studies predominantly focus on the 2D realm and rely heavily on extensive data for training.
Research on  3D VTON primarily centers on garment-body shape compatibility, a topic extensively covered in 2D VTON.
Thanks to advances in 3D scene editing, a 2D diffusion model has now been adapted for 3D editing via multi-viewpoint editing.
In this work, we propose GaussianVTON, an innovative 3D VTON pipeline integrating Gaussian Splatting (GS) editing with 2D VTON.
To facilitate a seamless transition from 2D to 3D VTON, we propose, for the first time,
the use of only images as editing prompts for 3D editing. To further address issues,
e.g., face blurring, garment inaccuracy, and degraded viewpoint quality during editing, we devise a three-stage refinement strategy to gradually mitigate potential issues.
Furthermore, we introduce a new editing strategy termed Edit Recall Reconstruction (ERR) to tackle the limitations of
previous editing strategies in leading to complex geometric changes.
Our comprehensive experiments demonstrate the superiority of GaussianVTON, offering a novel
perspective on 3D VTON while also establishing a novel starting point for image-prompting 3D scene editing.

## News

- **[2024/05/14]** Upload paper and release partial code.

Full code coming soon!


## Citation

If you find this work useful, please consider citing our paper:

```bash
@misc{chen2024gaussianvton,
      title={GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting}, 
      author={Haodong Chen and Yongle Huang and Haojian Huang and Xiangsheng Ge and Dian Shao},
      year={2024},
      eprint={2405.07472},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
