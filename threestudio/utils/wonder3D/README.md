# Wonder3D
Single Image to 3D using Cross-Domain Diffusion
## [Paper](https://arxiv.org/abs/2310.15008) | [Project page](https://www.xxlong.site/Wonder3D/) | [Hugging Face Demo](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo) | [Colab from @camenduru](https://github.com/camenduru/Wonder3D-colab)

![](assets/fig_teaser.png)

Wonder3D reconstructs highly-detailed textured meshes from a single-view image in only 2 ∼ 3 minutes. Wonder3D first generates consistent multi-view normal maps with corresponding color images via a cross-domain diffusion model, and then leverages a novel normal fusion method to achieve fast and high-quality reconstruction.

## Share your reconstructions!
If you get any interesting reconstructions and would like to share with others, welcome to upload the input image and reconstructed mesh to this [onedrive repo](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EvIYSlYlkwNJpQkEnlD7uHsB-C6zU1oSNjqAGz_K7VGG2Q). 


Data structure:
```
{yourname}/{scenename}-input.png  # the input image
{yourname}/{scenename}-screenshot.png  # a front view screenshot of the reconstructed mesh
{yourname}/{scenename}-mesh.obj  # the reconstructed mesh, .obj or .ply

# example: 
# create a folder name `xxlong`, then upload the files to the folder
xxlong/apple-input.png
xxlong/apple-screenshot.png
xxlong/apple-mesh.obj
```

## Collaborations
Our overarching mission is to enhance the speed, affordability, and quality of 3D AIGC, making the creation of 3D content accessible to all. While significant progress has been achieved in the recent years, we acknowledge there is still a substantial journey ahead. We enthusiastically invite you to engage in discussions and explore potential collaborations in any capacity. <span style="color:red">**If you're interested in connecting or partnering with us, please don't hesitate to reach out via email (xxlong@connect.hku.hk)**</span> .

## More features

The repo is still being under construction, thanks for your patience. 
- [x] Local gradio demo.
- [ ] Detailed tutorial.
- [ ] GUI demo for mesh reconstruction
- [ ] Windows support
- [ ] Docker support

## Schedule
- [x] Inference code and pretrained models.
- [x] Huggingface demo.
- [ ] New model trained on the whole Objaverse dataset.


### Preparation for inference
1. Install packages in `requirements.txt`(Linux system).
```angular2html
conda create -n wonder3d
conda activate wonder3d
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Windows System. Thank @fefespn for his help! (Not tested. May refer to issues [#12](https://github.com/xxlong0/Wonder3D/issues/12) [#29](https://github.com/xxlong0/Wonder3D/issues/29) )
```angular2html
conda create --name venv_wonder3d -y python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install fire diffusers==0.19.3 transformers bitsandbytes accelerate gradio rembg segment_anything
pip install einops omegaconf pytorch-lightning==1.9.5 torch_efficient_distloss nerfacc==0.3.3 PyMCubes trimesh
pip install https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp38-cp38-manylinux2014_x86_64.whl
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

2. Download the [checkpoints](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EgSHPyJAtaJFpV_BjXM3zXwB-UMIrT4v-sQwGgw-coPtIA) and into the root folder.

### Inference
1. Make sure you have the following models.
```bash
Wonder3D
|-- ckpts
    |-- unet
    |-- scheduler.bin
    ...
```
2. Predict foreground mask as the alpha channel. We use [Clipdrop](https://clipdrop.co/remove-background) to segment the foreground object interactively. 
You may also use `rembg` to remove the backgrounds.
```bash
# !pip install rembg
import rembg
result = rembg.remove(result)
result.show()
```
3. Run Wonder3d to produce multiview-consistent normal maps and color images. Then you can check the results in the folder `./outputs`. (we use `rembg` to remove backgrounds of the results, but the segmentations are not always perfect. May consider using [Clipdrop](https://clipdrop.co/remove-background) to get masks for the generated normal maps and color images, since the quality of masks will significantly influence the reconstructed mesh quality.) 
```bash
accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
            --config mvdiffusion-joint-ortho-6views.yaml
```
or 
```bash
bash run_test.sh
```

#### Interactive inference: run your local gradio demo
```bash
python gradio_app.py
```

4. Mesh Extraction

#### Instant-NSR Mesh Extraction

```bash
cd ./instant-nsr-pl
bash run.sh output_folder_path scene_name
```
Our generated normals and color images are defined in orthographic views, so the reconstructed mesh is also in orthographic camera space. If you use MeshLab to view the meshes, you can click `Toggle Orthographic Camera` in `View` tab.

#### NeuS-based Mesh Extraction

Since there are many complaints about the Windows setup of instant-nsr-pl, we provide the NeuS-based reconstruction, which may get rid of the requirement problems. 

NeuS consumes less GPU memory and favors smooth surfaces without parameters tuning. However, NeuS consumes more times and its texture may be less sharp. If you are not sensitive to time, we recommend NeuS for optimization due to its robustness.

```bash
cd ./NeuS
bash run.sh output_folder_path scene_name 
```

## Common questions
Q: The evelation and azimuth degrees of the generated views?

A: Unlike that the prior works such as Zero123, SyncDreamer and One2345 adopt object world system, our views are defined in the camera system of the input image. The six views are in the plane with 0 elevation degree in the camera system of the input image. Therefore we don't need to estimate an elevation degree for input image. The azimuth degrees of the six views are 0, 45, 90, 180, -90, -45 respectively.

Q: The focal length of the generated views?

A: We assume the input images are captured by orthographic camera, so the generated views are also in orthographic space. This design enables our model to keep strong generlaization on unreal images, but sometimes it may suffer from focal lens distortions on real-captured images.
## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [stable diffusion](https://github.com/CompVis/stable-diffusion)
- [zero123](https://github.com/cvlab-columbia/zero123)
- [NeuS](https://github.com/Totoro97/NeuS)
- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@misc{long2023wonder3d,
      title={Wonder3D: Single Image to 3D using Cross-Domain Diffusion}, 
      author={Xiaoxiao Long and Yuan-Chen Guo and Cheng Lin and Yuan Liu and Zhiyang Dou and Lingjie Liu and Yuexin Ma and Song-Hai Zhang and Marc Habermann and Christian Theobalt and Wenping Wang},
      year={2023},
      eprint={2310.15008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
