<h1 align="center"> <img src="assets/logo.png" style="vertical-align: -15px;" :height="40px" width="40px">  V-RGBX: Video Editing with Accurate Controls    over Intrinsic Properties  
</h1>

<p align="center">
<a href="https://arxiv.org/abs/2512.11799"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://aleafy.github.io/vrgbx/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://www.youtube.com/watch?v=j5yGqvB-BP0"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/Aleafy/V-RGBX" />
</a>
</p>


<!-- <p align="center">
  <b><a href="https://kszpxxzmc.github.io/">Ye Fang</a></b><sup>1,2</sup>,
  <b><a href="https://wutong16.github.io/">Tong Wu✉️</a></b><sup>3</sup>,
  <b><a href="https://valentin.deschaintre.fr/">Valentin Deschaintre</a></b><sup>1</sup>,
  <b><a href="https://www.duygu-ceylan.com/">Duygu Ceylan</a></b><sup>1</sup>,
  <b><a href="https://iliyan.com/">Iliyan Georgiev</a></b><sup>1</sup>,
</p>
<p align="center">
  <b><a href="https://paulchhuang.wixsite.com/chhuang">Chun-Hao Paul Huang</a></b><sup>1</sup>,
  <b><a href="https://yiweihu.netlify.app/">Yiwei Hu</a></b><sup>1</sup>
  <b><a href="https://xuelin-chen.github.io/">Xuelin Chen</a></b><sup>1</sup>,
  <b><a href="https://tuanfeng.github.io/">Tuanfeng Yang Wang✉️</a></b><sup>1</sup>
</p> -->
<p align="center" style="margin-bottom: 0px;">
  <b><a href="https://kszpxxzmc.github.io/">Ye Fang</a></b><sup>1,2</sup>,
  <b><a href="https://wutong16.github.io/">Tong Wu✉️</a></b><sup>3</sup>,
  <b><a href="https://valentin.deschaintre.fr/">Valentin Deschaintre</a></b><sup>1</sup>,
  <b><a href="https://www.duygu-ceylan.com/">Duygu Ceylan</a></b><sup>1</sup>,
  <b><a href="https://iliyan.com/">Iliyan Georgiev</a></b><sup>1</sup>,
</p>

<p align="center" style="margin-top: 0;">
  <b><a href="https://paulchhuang.wixsite.com/chhuang">Chun-Hao Paul Huang</a></b><sup>1</sup>,
  <b><a href="https://yiweihu.netlify.app/">Yiwei Hu</a></b><sup>1</sup>
  <b><a href="https://xuelin-chen.github.io/">Xuelin Chen</a></b><sup>1</sup>,
  <b><a href="https://tuanfeng.github.io/">Tuanfeng Yang Wang✉️</a></b><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>Adobe Research &nbsp; <sup>2</sup>Fudan University &nbsp; <sup>3</sup>Stanford University 
</p>


<p align="center">
  <b>TLDR:</b> V-RGBX enables physically grounded video editing by decomposing videos into intrinsic properties and propagating keyframe edits over time, producing photorealistic and precisely controlled results.
</p>

<!-- <p align="center">
  <img src="assets/teaser.gif" alt="SpaceTimePilot Teaser Video" width="600"/>
</p> -->

<!-- [Ye Fang](https://kszpxxzmc.github.io), [Tong Wu✉️](https://wutong16.github.io), [Valentin Deschaintre](https://valentin.deschaintre.fr/), [Duygu Ceylan](https://www.duygu-ceylan.com/), [Iliyan Georgiev](https://iliyan.com/), [Chun-Hao Paul Huang](https://paulchhuang.wixsite.com/chhuang), [Yiwei Hu](https://yiweihu.netlify.app/), [Xuelin Chen](https://xuelin-chen.github.io/), [Tuanfeng Yang Wang✉️](https://tuanfeng.github.io/) -->


[**Paper**](https://arxiv.org/pdf/2512.11799) | [**Project page**](https://aleafy.github.io/vrgbx/) | [**Video**](https://www.youtube.com/watch?v=j5yGqvB-BP0) |
[**Huggingface**](https://huggingface.co/aleafy/V-RGBX/tree/main)


<details><summary>Click for the full abstract of V-RGBX</summary>

> Large-scale video generation models have shown remarkable potential in modeling photorealistic appearance and lighting interactions in real-world scenes. However, a closed-loop framework that jointly understands intrinsic scene properties (e.g., albedo, normal, material, and irradiance), leverages them for video synthesis, and supports editable intrinsic representations remains unexplored. We present V-RGBX, the first end-to-end framework for intrinsic-aware video editing. V-RGBX unifies three key capabilities: (1) video inverse rendering into intrinsic channels, (2) photorealistic video synthesis from these intrinsic representations, and (3) keyframe-based video editing conditioned on intrinsic channels. At the core of V-RGBX is an interleaved conditioning mechanism that enables intuitive, physically grounded video editing through user-selected keyframes, supporting flexible manipulation of any intrinsic modality. Extensive qualitative and quantitative results show that V-RGBX produces temporally consistent, photorealistic videos while propagating keyframe edits across sequences in a physically plausible manner. We demonstrate its effectiveness in diverse applications, including object appearance editing and scene-level relighting, surpassing the performance of prior methods.
</details> 


<!-- This repository is the official implementation of V-RGBX. It is a **training-free framework** that enables 
zero-shot illumination control of any given video sequences or foreground sequences.

<details><summary>Click for the full abstract of V-RGBX</summary>

> Recent advancements</details> -->


![Teaser Image](./assets/teaser.png)


*This work was partially done while Ye was an intern at Adobe Research.


## 🔥 News
<!-- - 🚀 [Jan 12, 2026] We release the **V-RGBX model weights** and **inference code**, including  
  [pretrained inverse & forward renderers](#🔑-model-weights) and  
  the [intrinsic-aware video editing pipeline](#💡-inference). -->

<!--- 🚀 **[Jan 1x, 2026]** We release the **[V-RGBX model weights](#🔑-model-weights)** and **[inference code](#💡-inference)**, including **inverse rendering**, **forward rendering**, and **intrinsic-aware video editing**.  [[Model Weights](#🔑-model-weights)] · [[Inference](#💡-inference)]-->

- 🚀 **[Jan 15, 2026]** We release the **V-RGBX model weights** and **inference code**, including **inverse rendering**, **forward rendering**, and **intrinsic-aware video editing**.  [[Model Weights](#🔑-model-weights)] · [[Inference](#💡-inference)]
- 🚀 **[Dec 15, 2025]** The [paper](https://arxiv.org/abs/2512.11799) and [project page](https://aleafy.github.io/vrgbx/) are released!

<!-- 📚 Gallery -->

<!-- 🚀 Method Overview -->

## 🌟 Highlights

- 🔥 The **first end-to-end intrinsic-aware video editing framework**, enabling physically grounded control over **albedo, normal, material, and irradiance**.
- 🔥 A **unified RGB → X → RGB pipeline** that supports **keyframe-based edit propagation** across time via inverse and forward rendering.
- 🔥 **Interleaved intrinsic conditioning** with **temporal-aware embeddings** ensures stable and temporally consistent video generation under complex multi-attribute edits.

## 📦 Installations
#### 1. Clone the repository
```bash
git clone https://github.com/Aleafy/V-RGBX.git
cd V-RGBX
```
#### 2. Create Conda environment
```bash
conda create -n vrgbx python=3.10 
conda activate vrgbx
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```


## 🔑 Model Weights

The model weights are available on [Hugging Face](https://huggingface.co/aleafy/V-RGBX/tree/main).
We provide 2 checkpoints:

| Checkpoints | Description |
|------------|-------------|
| **[aleafy/vrgbx_inverse_renderer](https://huggingface.co/aleafy/V-RGBX/tree/main)** | Decomposes an input RGB video into intrinsic channels (albedo, normal, material, irradiance). |
| **[aleafy/vrgbx_forward_renderer](https://huggingface.co/aleafy/V-RGBX/tree/main)** | Renders a photorealistic RGB video from intrinsic channels and propagates keyframe edits over time. |


<!-- | Checkpoints                                                                                                                | Description                                                           |
| -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [aleafy/vrgbx_inverse_renderer](https://huggingface.co/aleafy/V-RGBX/tree/main)                     | Inverse Renderer                                               |
| [aleafy/vrgbx_forward_renderer](https://huggingface.co/aleafy/V-RGBX/tree/main) | Forward Renderer              | -->


You can download V-RGBX model weights by running the following command:

```bash
python vrgbx/utils/download_weights.py --repo_id aleafy/V-RGBX
```

The pretrained backbone (built on WAN) can be downloaded with:
```bash
python vrgbx/utils/download_weights.py --repo_id Wan-AI/Wan2.1-T2V-1.3B
```


Expected project directory:
<pre>
V-RGBX/                              # Project root for the V-RGBX framework
├── assets/                          # Media resources(logos, figures, etc)
├── examples/                        # Example videos, intrinsics, and reference images
├── models/                          # Model weights directory
    ├── V-RGBX/                      # V-RGBX intrinsic rendering models
    │   ├── vrgbx_forward_renderer.safetensors   
    │   └── vrgbx_inverse_renderer.safetensors   
    └── Wan-AI/                      # Pretrained backbone (Wan)
        └── Wan2.1-T2V-1.3B/          
└── vrgbx/                           # Core V-RGBX codebase 
</pre>
<!--<pre>
V-RGBX/
├── assets/
├── examples/
├── vrgbx/
└── models/
    └── V-RGBX/
        ├── vrgbx_forward_renderer.safetensors
        └── vrgbx_inverse_renderer.safetensors
    └── Wan-AI/
        └── Wan2.1-T2V-1.3B/
</pre>-->


<!-- ## 🛠️ Usage -->


## 💡 Inference 
### 1. Perform intrinsic-aware video editing with disentangled property control:
```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_CreativeLoft \
    --task solid_color \
    --edit_type albedo
```
This command automatically resolves all required inputs by `video_name`, applies the specified intrinsic edit, and re-renders the edited result to RGB.

**Arguments**

- `video_name` : Video sequence name. All required RGB videos and reference images are automatically inferred from the dataset structure.
- `task` : A short tag used for file naming and auto path inference, e.g. `texture`, `material`, `shadow`, `light_color`, `normal`.
- `edit_type` : Intrinsic layer to edit, e.g. `albedo`, `irradiance`, `material`, or `normal`.


**Use your own video**

Put your files in the same structure:
<pre>
examples/
├── input_videos/
│   └── {your_video_name}.mp4
└── edit_images/
    ├── {your_video_name}_{your_task}_edit_ref.png   # edited RGB reference
    └── {your_video_name}_{your_task}_edit_x.png     # edited intrinsic (for --edit_type)
</pre>

Running command:
```bash
python vrgbx_edit_inference.py \
    --video_name <your_video_name> \
    --task <your_task> \
    --edit_type <your_edit_type>
```

<details><summary><b>🪄Click for more example bash commands of V-RGBX editing</b></summary>

```bash
python vrgbx_edit_inference.py \
    --video_name AdobeStock_GradientShadow \
    --task texture \
    --edit_type albedo
```

```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_Lounge \
    --task texture \
    --edit_type albedo
```

```bash
python vrgbx_edit_inference.py \
    --video_name Captured_PoolTable \
    --task texture \
    --edit_type albedo
```

```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_Kitchenette \
    --task light_color \
    --edit_type irradiance
```

```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_Studio \
    --task shadow \
    --edit_type irradiance
```

```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_CreativeLoft \
    --task shadow \
    --edit_type irradiance
```

```bash
python vrgbx_edit_inference.py \
    --video_name Evermotion_SingleWallKitchen \
    --task normal \
    --edit_type normal \
    --drop_type irradiance
```

```bash
python vrgbx_edit_inference.py \ 
    --video_path examples/input_videos/Evermotion_Lounge.mp4 \ 
    --ref_rgb_path examples/edit_images/Evermotion_Lounge_multiple_edit_ref.png \
    --edit_type irradiance --drop_type albedo \
    --edit_x_path examples/edit_images/Evermotion_Lounge_multiple_edit_irradiance.png --task multiple
```
</details>

### 2. Perform inverse rendering to extract disentangled intrinsic layers:
```bash
python vrgbx_inverse_rendering.py \
    --video_path examples/input_videos/Evermotion_CreativeLoft.mp4 \
    --save_dir output/inverse_rendering \
    --channels albedo normal material irradiance
```
This command decomposes the input video into intrinsic representations (e.g., albedo, shading, geometry, material) and saves them for later intrinsic-aware editing.

You can also try other cases in `examples/input_videos/` or use your own videos (recommended: 49 frames at 832×480 for better results).

### 3. Perform forward rendering to compose an RGB video from intrinsic inputs:
The forward renderer reconstructs an RGB video from multiple intrinsic layers, including **albedo**, **normal**, **material**, and **irradiance**.

**Without a reference image** (pure intrinsic-driven rendering):
```bash
python vrgbx_forward_rendering.py \
    --albedo_path examples/input_intrinsics/Evermotion_Banquet_Albedo.mp4 \
    --normal_path examples/input_intrinsics/Evermotion_Banquet_Normal.mp4 \
    --material_path examples/input_intrinsics/Evermotion_Banquet_Material.mp4 \
    --irradiance_path examples/input_intrinsics/Evermotion_Banquet_Irradiance.mp4
```

**With a reference RGB image** (to anchor global appearance and color tone):
```bash
python vrgbx_forward_rendering.py \
    --albedo_path examples/input_intrinsics/Evermotion_Banquet_Albedo.mp4 \
    --normal_path examples/input_intrinsics/Evermotion_Banquet_Normal.mp4 \
    --material_path examples/input_intrinsics/Evermotion_Banquet_Material.mp4 \
    --irradiance_path examples/input_intrinsics/Evermotion_Banquet_Irradiance.mp4 \
    --use_reference \
    --ref_rgb_path examples/input_intrinsics/Evermotion_Banquet_Ref.png
```
`Note`:
- The first mode reconstructs RGB solely from intrinsic layers.
- The second mode additionally uses a reference RGB image to provide global color and appearance guidance, improving visual fidelity.
- Due to intrinsic sampling mechanism, intrinsic channels do not need to be all provided — partial inputs are supported.



<!-- ### 🏋️‍♂️ Train  -->


<!-- ## ✨ Updates




## 📚 Dataset

## 🏋️‍♂️ Training -->

<!-- 🔥 We will release the code and models soon! -->



<!-- ## 🛠️ Usage

### Installation

### Inference -->



## 📝 Todo
- [x] Open-source V-RGBX models & weights
- [x] Intrinsic-conditioned video editing inference
- [x] Inverse rendering (RGB → X) inference
- [x] Forward rendering (X → RGB) inference
- [ ] Inverse Renderer training code
- [ ] Forward Renderer training code


<!-- 📣 Disclaimer-->
<!-- 📄 License -->
<!-- ## 📚 Acknowledgements
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
- []() -->


## ❤️ Acknowledgments

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main): A modular diffusion framework for training and inference across mainstream diffusion models (e.g., FLUX and Wan), which provides the codebase used in our V-RGBX implementation.
- [WAN-Video](https://github.com/Wan-Video/Wan2.1): A large-scale open video diffusion foundation model. We leverage its pretrained video generation capability as the base model for high-quality synthesis in our experiments.
- [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer): An influential line of work that bridges physically-based rendering and diffusion models, motivating our forward/inverse rendering formulation for intrinsic-aware video generation.
- [RGB↔X](https://github.com/zheng95z/rgbx): A seminal framework for intrinsic image decomposition and editing, laying the foundation for disentangled representations (e.g., albedo, normal, material, illumination).

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@misc{fang2025vrgbxvideoeditingaccurate,
      title={V-RGBX: Video Editing with Accurate Controls over Intrinsic Properties}, 
      author={Ye Fang and Tong Wu and Valentin Deschaintre and Duygu Ceylan and Iliyan Georgiev and Chun-Hao Paul Huang and Yiwei Hu and Xuelin Chen and Tuanfeng Yang Wang},
      year={2025},
      eprint={2512.11799},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.11799}, 
}
```

