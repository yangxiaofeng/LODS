import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("lods-embedding-guidance")
class StableDiffusionLODSGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True


        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }


        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)


        self.submodules = SubModules(pipe=pipe)
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)


        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        )


        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )


        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )


        self.pipe.scheduler = self.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        # self.learnable_embedding = torch.zeros((10,77, 768)).to(self.device)
        # self.notlearnable_null = torch.zeros((10,77, 768)).to(self.device)

        self.learnable_embedding = torch.nn.Parameter(torch.zeros((1,77, 1024)).clone()[None].to(self.device))
        # self.learnable_embedding = torch.nn.Parameter(torch.zeros((77, 1024)).to(self.device))
        self.learnable_embedding.requires_grad = True
        self.changed = False
        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe


    @property
    def unet(self):
        return self.submodules.pipe.unet


    @property
    def vae(self):
        return self.submodules.pipe.vae


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_embedding(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        learnable_embedding = self.learnable_embedding.repeat(B,  1, 1)
        text_embeddings_vd = torch.cat([text_embeddings_vd[0:B],learnable_embedding])



        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred_pretrain = self.forward_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd
                )




        (
            noise_pred_cond,
            noise_pred_uncon,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain =  noise_pred_cond + (1- self.cfg.guidance_scale)/self.cfg.guidance_scale*noise_pred_uncon - noise/self.cfg.guidance_scale


        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain)
        return grad

    def train_embedding(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]

        if (self.changed == False):
            self.learnable_embedding.data = text_embeddings_vd[B].detach()
            self.changed = True

        latents = latents.detach().repeat(1, 1, 1, 1)
        learnable_embedding = self.learnable_embedding.repeat(B,  1, 1)
        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        noise_pred_text = self.forward_unet(
            self.unet,
            latents_noisy,
            torch.cat([t]),
            encoder_hidden_states=learnable_embedding,
        )
        return F.mse_loss(noise_pred_text.float(), noise.detach().float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=True,
        )
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        loss_embedding = self.train_embedding(latents,text_embeddings_vd, text_embeddings, camera_condition)
        grad = self.compute_grad_embedding(
            latents,text_embeddings_vd, text_embeddings, camera_condition
        )

        grad = torch.nan_to_num(grad)


        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size



        return {
            "loss_sds": loss_sds,
            "loss_embedding": loss_embedding,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
