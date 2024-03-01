{
  "name": "sample",
  "path": "share/Stable-diffusion/mistoonAnime_v20.safetensors",  # 将Checkpoint指定为相对于 /animatediff-cli/data 的路径
  "vae_path": "share/VAE/vae-ft-mse-840000-ema-pruned.ckpt",       # 将vae指定为相对于 /animatediff-cli/data 的路径
  "motion_module": "models/motion-module/mm_sd_v14.ckpt",         # 将motion module指定为相对于 /animatediff-cli/data 的路径
  "compile": false,
  "seed": [
    341774366206100,
    -1,
    -1         # -1表示随机。 如果在此设置中指定“--repeats 3”，则第一个将为 341774366206100，第二个和第三个将是随机的。
  ],
  "scheduler": "ddim",      # # "ddim","euler","euler_a","k_dpmpp_2m", etc...
  "steps": 40,
  "guidance_scale": 20,     # cfg scale
  "clip_skip": 2,
  "prompt_fixed_ratio": 0.5,
  "head_prompt": "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo",
  "prompt_map": {        # “FRAME”：“PROMPT”格式/例如, 第 32 帧的提示是“head_prompt”+prompt_map[“32”]+“tail_prompt”
    "0": "smile standing,((spider webs:1.0))",
    "32": "(((walking))),((spider webs:1.0))",
    "64": "(((running))),((spider webs:2.0)),wide angle lens, fish eye effect",
    "96": "(((sitting))),((spider webs:1.0))"
  },
  "tail_prompt": "clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear",
  "n_prompt": [
    "(worst quality, low quality:1.4),nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,(pink body:1.4),7 arms,8 arms,4 arms"
  ],
  "lora_map": {             # "PATH_TO_LORA": STRENGTH format
    "share/Lora/muffet_v2.safetensors": 1.0,                     # Specify lora as a path relative to /animatediff-cli/data
    "share/Lora/add_detail.safetensors": 1.0                     # Lora support is limited. Not all formats can be used!!!
  },
  "motion_lora_map": {      # "PATH_TO_LORA": STRENGTH format
    "models/motion_lora/v2_lora_RollingAnticlockwise.ckpt": 0.5,   # Currently, the officially distributed lora seems to work only for v2 motion modules (mm_sd_v15_v2.ckpt).
    "models/motion_lora/v2_lora_ZoomIn.ckpt": 0.5
  },
  "ip_adapter_map": {       # config for ip-adapter
      # enable/disable (important)
      "enable": true,
      # Specify input image directory relative to /animatediff-cli/data (important! No need to specify frames in the config file. The effect on generation is exactly the same logic as the placement of the prompt)
      "input_image_dir": "ip_adapter_image/test",
    "prompt_fixed_ratio": 0.5,
      # save input image or not
      "save_input_image": true,
      # Ratio of image prompt vs text prompt (important). Even if you want to emphasize only the image prompt in 1.0, do not leave prompt/neg prompt empty, but specify a general text such as "best quality".
      "scale": 0.5,
      # IP-Adapter or IP-Adapter Plus or IP-Adapter Plus Face (important) It would be a completely different outcome. Not always PLUS a superior result.
      "is_plus_face": true,
    "is_plus": true
  },
  "img2img_map": {
      # enable/disable
      "enable": true,
      # Directory where the initial image is placed
      "init_img_dir": "..\\stylize\\2023-10-27T19-43-01-sample-mistoonanime_v20\\00_img2img",
    "save_init_image": true,
      # The smaller the value, the closer the result will be to the initial image.
      "denoising_strength": 0.7
  },
  "region_map": {
      # setting for region 0. You can also add regions if necessary.
      # The region added at the back will be drawn at the front.
      "0": {
          # enable/disable
          "enable": true,
          # If you want to draw a separate object for each region, enter a value of 0.1 or higher.
          "crop_generation_rate": 0.1,
          # Directory where mask images are placed
          "mask_dir": "..\\stylize\\2023-10-27T19-43-01-sample-mistoonanime_v20\\r_fg_00_2023-10-27T19-44-08\\00_mask",
      "save_mask": true,
          # If true, the initial image will be drawn as is (inpaint)
          "is_init_img": false,
          # conditions for region 0
          "condition": {
              # text prompt for region 0
              "prompt_fixed_ratio": 0.5,
        "head_prompt": "",
        "prompt_map": {
          "0": "(masterpiece, best quality:1.2), solo, 1girl, kusanagi motoko, looking at viewer, jacket, leotard, thighhighs, gloves, cleavage"
        },
        "tail_prompt": "",
              # image prompt(ip adapter) for region 0
              # It is not possible to change lora for each region, but you can do something similar using an ip adapter.
              "ip_adapter_map": {
          "enable": true,
          "input_image_dir": "..\\stylize\\2023-10-27T19-43-01-sample-mistoonanime_v20\\r_fg_00_2023-10-27T19-44-08\\00_ipadapter",
          "prompt_fixed_ratio": 0.5,
          "save_input_image": true,
          "resized_to_square": false
        }
      }
    },
      # setting for background
      "background": {
          # If true, the initial image will be drawn as is (inpaint)
          "is_init_img": true,
      "hint": "background's condition refers to the one in root"
    }
  },
  "controlnet_map": {       # config for controlnet(for generation)
    "input_image_dir": "controlnet_image/test",    # Specify input image directory relative to /animatediff-cli/data (important! Please refer to the directory structure of sample. No need to specify frames in the config file.)
    "max_samples_on_vram": 200,    # If you specify a large number of images for controlnet and vram will not be enough, reduce this value. 0 means that everything should be placed in cpu.
    "max_models_on_vram": 3,       # Number of controlnet models to be placed in vram
    "save_detectmap": true,        # save preprocessed image or not
    "preprocess_on_gpu": true,      # run preprocess on gpu or not (It probably does not affect vram usage at peak, so it should always set true.)
    "is_loop": true,                # Whether controlnet effects consider loop

    "controlnet_tile": {    # config for controlnet_tile
      "enable": true,              # enable/disable (important)
      "use_preprocessor": true,      # Whether to use a preprocessor for each controlnet type
      "preprocessor": {     # If not specified, the default preprocessor is selected.(Most of the time the default should be fine.)
        # none/blur/tile_resample/upernet_seg/ or key in controlnet_aux.processor.MODELS
        # https: //github.com/patrickvonplaten/controlnet_aux/blob/2fd027162e7aef8c18d0a9b5a344727d37f4f13d/src/controlnet_aux/processor.py#L20
        "type": "tile_resample",
        "param": {
          "down_sampling_rate": 2.0
        }
      },
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,    # control weight (important)
      "control_guidance_start": 0.0,       # starting control step
      "control_guidance_end": 1.0,         # ending control step
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]    # list of influences on neighboring frames (important)
    },                                              # This means that there is an impact of 0.5 on both neighboring frames and 0.4 on the one next to it. Try lengthening, shortening, or changing the values inside.
    "controlnet_ip2p": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_lineart_anime": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_openpose": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_softedge": {
      "enable": true,
      "use_preprocessor": true,
      "preprocessor": {
        "type": "softedge_pidsafe",
        "param": {}
      },
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_ref": {
      "enable": false,            # enable/disable (important)
        "ref_image": "ref_image/ref_sample.png",     # path to reference image.
        "attention_auto_machine_weight": 1.0,
      "gn_auto_machine_weight": 1.0,
      "style_fidelity": 0.5,                # control weight-like parameter(important)
        "reference_attn": true,               # [attn=true , adain=false
      ] means "reference_only""reference_adain": false,
      "scale_pattern": [
        0.5
      ]                 # Pattern for applying controlnet_ref to frames
    }                                         # ex. [
      0.5
    ] means [
      0.5,
      0.5,
      0.5,
      0.5,
      0.5 ....
    ]. All frames are affected by 50%
                                              # ex. [
      1,
      0
    ] means [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1 ....
    ]. Only even frames are affected by 100%.
  },
  "upscale_config": {       # config for tile-upscale
    "scheduler": "ddim",
    "steps": 20,
    "strength": 0.5,
    "guidance_scale": 10,
    "controlnet_tile": {    # config for controlnet tile
      "enable": true,       # enable/disable (important)
      "controlnet_conditioning_scale": 1.0,     # control weight (important)
      "guess_mode": false,
      "control_guidance_start": 0.0,      # starting control step
      "control_guidance_end": 1.0         # ending control step
    },
    "controlnet_line_anime": {  # config for controlnet line anime
      "enable": false,
      "controlnet_conditioning_scale": 1.0,
      "guess_mode": false,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    },
    "controlnet_ip2p": {  # config for controlnet ip2p
      "enable": false,
      "controlnet_conditioning_scale": 0.5,
      "guess_mode": false,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    },
    "controlnet_ref": {   # config for controlnet ref
      "enable": false,             # enable/disable (important)
      "use_frame_as_ref_image": false,   # use original frames as ref_image for each upscale (important)
      "use_1st_frame_as_ref_image": false,   # use 1st original frame as ref_image for all upscale (important)
      "ref_image": "ref_image/path_to_your_ref_img.jpg",   # use specified image file as ref_image for all upscale (important)
      "attention_auto_machine_weight": 1.0,
      "gn_auto_machine_weight": 1.0,
      "style_fidelity": 0.25,       # control weight-like parameter(important)
      "reference_attn": true,       # [attn=true , adain=false
      ] means "reference_only""reference_adain": false
    }
  },
  "output": {   # output format
    "format": "gif",   # gif/mp4/webm
    "fps": 8,
    "encode_param": {
      "crf": 10
    }
  }
}