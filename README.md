# DiffuseIR: Diffusion Models For Isotropic Reconstruction of 3D Microscopic Images
Implementation of ['DiffuseIR: Diffusion Models For Isotropic Reconstruction of 3D Microscopic Images'](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_31).

The paper has been accepted by **MICCAI 2023** 🎉.

## Workflow
1. Slice the 3D image along the XY high-resolution plane into 128x128 patches as the pre-training dataset, saved in `cfg:gt_path`.
2. Train the diffusion model to fit the high-resolution data distribution.
3. Pad the low-resolution images from the XZ/YZ planes of the 3D image as inpainting inputs, saved in `cfg:gt_path`.
4. Run the inference script test_cremi.sh.

## Training
Download data from [https://cremi.org](https://cremi.org/)

Please refer to [guided-diffusion](https://github.com/openai/guided-diffusion) for training steps:
```bash
git clone https://github.com/openai/guided-diffusion
cd guided-diffusion

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS
```

## Inference
```bash
bash test_cremi.sh
```