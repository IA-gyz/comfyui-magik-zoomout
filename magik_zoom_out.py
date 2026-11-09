r"""
Magik Zoom Out (V3)
====================

A single ComfyUI node that reproduces the graph in custnode.json:

    IMAGE -> MagikDownPad -> VAEEncode -> ApplyCosmosReferenceLatent(model) -> KSampler -> LATENT
                                   \_______________ latent_image ________________/

The source image is downscaled and centered on a black canvas the same size as
the original ("zoomed out"), VAE-encoded, and that latent is used BOTH as the
KSampler's starting latent_image AND as a Cosmos temporal reference latent
patched onto the model. The reference latent keeps the sample anchored to the
original composition while the model denoises/fills in the padded canvas,
producing the zoom-out result.

`sampling_steps` splits this into that many chained KSampler passes instead of
one: each pass's own per-step downscale is `downscale ** (1 / sampling_steps)`.
Every pass - including pass 2+ - downscales-and-pads in real pixel space with
genuine black padding, VAE-encodes, samples, then (if it isn't the last pass)
VAE-decodes the result back to pixel space to feed the next pass's
downscale-and-pad. This costs an extra decode+encode round-trip per pass, but
it's necessary: padding directly in latent space with a zero-valued latent
does NOT decode back to black (zero-latent maps to whatever color the VAE's
latent origin corresponds to, not black), so the model has no learned "fill
this in" signal there and leaves that region untouched instead of generating
into it. With sampling_steps=1 this reduces exactly to the single-pass
behavior.

Each pass's decoded output also gets a per-channel mean/std color match back
to that pass's own pre-padding input (_match_color_stats). Every pass is a
near-full (denoise=1.0) regeneration, and diffusion models tend to have a
small color/saturation bias relative to their own input; since pass i+1
references pass i's OWN output as both latent_image and the Cosmos reference
latent, that bias compounds multiplicatively across passes, showing up as
increasingly "burnt"/oversaturated colors by sampling_steps=3+ without this
correction.

The reference-latent patch (_apply_reference_latent / _extra_conds_with_reference /
_diffusion_reference_wrapper) is a self-contained port of the technique used by
ApplyCosmosReferenceLatent in Mirumo0u0/ComfyUI-Cosmos-Reference, so this node has
no dependency on that package being installed. It only applies to Anima-family
models; on any other model type the patch is skipped and this node behaves like a
plain VAEEncode -> KSampler chain on the padded image.

`vae_decode` optionally decodes the final LATENT back to an IMAGE using the same
`vae` input already used for encoding. When disabled, the `image` output is a
1x1 placeholder - this only skips the FINAL decode though; with sampling_steps > 1
the intermediate per-pass decodes still happen regardless, since later passes
need them.

`first_pass_padded_image` is always just the very first pass's pixel-space
downscale-and-pad result (before its VAE encode). With sampling_steps > 1 it
does NOT reflect the input to later passes, which each downscale-and-pad their
own freshly-decoded image.
"""

import logging

import torch
import torch.nn.functional as F

import comfy.conds
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher

import latent_preview

from comfy_api.latest import io

REF_LATENT_WRAPPER_KEY = "magik_zoom_out_temporal_reference"
REF_LATENT_COND_KEY = "ref_latents"
# Only a single reference latent is ever used, but ApplyCosmosReferenceLatent's
# ref_latents dict is keyed by name - "ref_latent_1" matches its own
# single-slot naming convention. Named here rather than inlined so it's clear
# this is a fixed single-slot key, not evidence of multi-reference support.
REF_LATENT_KEY = "ref_latent_1"


class MagikZoomOut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS

        return io.Schema(
            node_id="Magik.ZoomOut",
            display_name="Magik Zoom Out",
            category="Magik/Image/Process",
            description=(
                "Downscales and pads an image onto a black canvas, VAE-encodes it, and samples it "
                "with the same latent used both as the starting latent_image and as a Cosmos temporal "
                "reference latent on the model, for an Anima 'zoom out' style generation. Optionally "
                "chains this across multiple KSampler passes (Sampling Steps) and can VAE-decode the "
                "final result to an image."
            ),
            search_aliases=["zoom out", "cosmos reference", "anima zoom out", "outpaint", "downpad sampler"],
            inputs=[
                io.Model.Input("model", tooltip="Model to sample with. Reference-latent conditioning is only applied to Anima-family models."),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Conditioning.Input("positive", tooltip="The conditioning describing the attributes you want to include in the image."),
                io.Conditioning.Input("negative", tooltip="The conditioning describing the attributes you want to exclude from the image."),
                io.Float.Input(
                    "downscale",
                    default=0.95,
                    min=0.01,
                    max=1.00,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Scale factor applied to the image before it's centered on a black canvas the same size as the original.",
                ),
                io.Int.Input(
                    "sampling_steps",
                    default=1,
                    min=1,
                    max=99,
                    tooltip="Number of chained KSampler passes. Each pass's own downscale is 'Downscale' ** (1 / 'Sampling Steps'), so the compounded downscale across all passes equals the 'Downscale' value. Each pass costs its own VAE decode/encode round-trip.",
                ),
                io.Combo.Input(
                    "multiple_of",
                    options=["1", "2", "4", "8", "16", "32", "64"],
                    default="64",
                    tooltip="Forces the downscaled dimensions to a multiple of this value.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                io.Int.Input("steps", default=12, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input(
                    "sampler_name",
                    options=samplers,
                    default="euler_ancestral" if "euler_ancestral" in samplers else samplers[0],
                ),
                io.Combo.Input(
                    "scheduler",
                    options=schedulers,
                    default="beta57" if "beta57" in schedulers else schedulers[0],
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Amount of denoising applied. Lower values keep more of the padded starting latent.",
                ),
                io.Boolean.Input(
                    "vae_decode",
                    default=True,
                    label_on="decode",
                    label_off="skip",
                    tooltip="Decode the final LATENT back to an image using the vae input. When off, the image output is a 1x1 placeholder. Only affects the final decode - intermediate passes still decode/encode regardless.",
                ),
            ],
            outputs=[
                io.Latent.Output("LATENT", display_name="LATENT"),
                io.Image.Output("first_pass_padded_image", display_name="first_pass_padded_image"),
                io.Image.Output("image", display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, model: ModelPatcher, vae, image: torch.Tensor, positive, negative,
                downscale: float, sampling_steps: int, multiple_of: str,
                seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str,
                denoise: float, vae_decode: bool):
        multiple_of = int(multiple_of)
        # Each of the `sampling_steps` passes multiplies the size by this
        # factor; since the passes compound multiplicatively,
        # downscale ** (1/sampling_steps) is the factor that makes the
        # compounded total across all passes equal exactly `downscale`.
        per_step_downscale = downscale ** (1.0 / sampling_steps)

        current_image = image
        first_pass_padded_image = None
        current_latent = None
        decoded_image = None

        for i in range(sampling_steps):
            # Every pass downscales-and-pads in real pixel space (genuine
            # black padding) before encoding, exactly like the first pass -
            # padding with a zero-valued LATENT instead (skipping the VAE
            # round-trip) does NOT decode back to black, so the model has no
            # learned "fill this in" signal there and leaves it untouched.
            padded_image = cls._downscale_and_pad_image(current_image, per_step_downscale, multiple_of)
            if i == 0:
                first_pass_padded_image = padded_image

            # Reference color statistics for THIS pass, taken from its own
            # input before padding (so the black border doesn't skew them).
            # Used below to correct this pass's own output.
            step_ref_mean = current_image.mean(dim=(0, 1, 2), keepdim=True)
            step_ref_std = current_image.std(dim=(0, 1, 2), keepdim=True)

            current_latent = {"samples": vae.encode(padded_image)}

            patched_model = cls._apply_reference_latent(model, current_latent)

            current_latent = cls._ksample(
                patched_model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, current_latent, denoise,
            )

            is_last_pass = (i == sampling_steps - 1)
            if not is_last_pass or vae_decode:
                # Needed as pixel-space input to the next pass's downscale-and-pad,
                # or (on the last pass) as the final `image` output.
                decoded_image = cls._vae_decode(vae, current_latent["samples"])
                # Each denoise=1.0 pass is close to a full regeneration, and
                # diffusion models tend to have a small color/saturation bias
                # relative to their own input. Since pass i+1 references pass
                # i's OWN output, that bias compounds multiplicatively across
                # passes - invisible at 1-2 passes, visibly "burnt" by 3+.
                # Anchoring each pass's output back to its own input (rather
                # than always to the very first frame, which would fight
                # legitimately different newly-revealed content) breaks that
                # compounding without constraining how the composition
                # evolves pass to pass.
                decoded_image = cls._match_color_stats(decoded_image, step_ref_mean, step_ref_std)
                current_image = decoded_image

        result = current_latent

        if vae_decode:
            final_image = decoded_image
        else:
            final_image = torch.zeros((1, 1, 1, 3), dtype=image.dtype, device=image.device)

        return io.NodeOutput(result, first_pass_padded_image, final_image)

    # ------------------------------------------------------------------
    # Same handling as core VAEDecode.decode(): unwrap nested tensors, and
    # for temporal/video VAEs (Anima/Cosmos) collapse the 5D [B,T,H,W,C]
    # decode output into a standard 4D [B*T,H,W,C] IMAGE batch. Also clamps
    # to the [0,1] range ComfyUI's IMAGE type expects - vae.decode() isn't
    # guaranteed to stay within that range (highlights especially can spike
    # slightly above 1.0), and feeding that straight back into the next
    # pass's vae.encode() without clamping compounds pass over pass, showing
    # up as increasingly "burnt"/oversaturated colors by sampling_steps=3+.
    # ------------------------------------------------------------------
    @staticmethod
    def _vae_decode(vae, latent_samples: torch.Tensor) -> torch.Tensor:
        if latent_samples.is_nested:
            latent_samples = latent_samples.unbind()[0]

        decoded_image = vae.decode(latent_samples)
        if len(decoded_image.shape) == 5:  # Combine batches
            decoded_image = decoded_image.reshape(-1, decoded_image.shape[-3], decoded_image.shape[-2], decoded_image.shape[-1])

        decoded_image = decoded_image.clamp(0.0, 1.0)

        return decoded_image

    # ------------------------------------------------------------------
    # Per-channel mean/std color transfer (standard "Reinhard-style" color
    # matching). Rescales `image` so its per-channel mean and std match
    # ref_mean/ref_std, correcting color/exposure drift while leaving
    # composition and detail untouched. ref_mean/ref_std should be
    # broadcastable against image, e.g. shape (1,1,1,C) from
    # image.mean(dim=(0,1,2), keepdim=True).
    # ------------------------------------------------------------------
    @staticmethod
    def _match_color_stats(image: torch.Tensor, ref_mean: torch.Tensor, ref_std: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mean = image.mean(dim=(0, 1, 2), keepdim=True)
        std = image.std(dim=(0, 1, 2), keepdim=True)

        normalized = (image - mean) / (std + eps)
        matched = normalized * ref_std + ref_mean

        return matched.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 1 (MagikDownPad): downscale the image, center it on a black canvas.
    # IMAGE tensors are channel-last [B,H,W,C], so this permutes for interpolate.
    # ------------------------------------------------------------------
    @staticmethod
    def _downscale_and_pad_image(image: torch.Tensor, downscale: float, multiple_of: int) -> torch.Tensor:
        b, h, w, c = image.shape

        new_h, new_w = MagikZoomOut._scaled_padded_size(h, w, downscale, multiple_of)

        img_permuted = image.permute(0, 3, 1, 2)
        resized_img = F.interpolate(img_permuted, size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized_img = resized_img.permute(0, 2, 3, 1)

        canvas = torch.zeros_like(image)

        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2

        canvas[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized_img

        return canvas

    @staticmethod
    def _scaled_padded_size(h: int, w: int, downscale: float, multiple_of: int) -> tuple[int, int]:
        # A downscale of 1.0 means "no change" - short-circuit so an input
        # whose dims aren't already a multiple_of multiple isn't shrunk just
        # to satisfy the alignment grid (h,w are returned as-is, unmodified).
        if downscale >= 1.0:
            return h, w

        target_w = w * downscale
        target_h = h * downscale

        # Round to the NEAREST multiple_of grid line rather than flooring.
        # Flooring can cause a small requested shrink to collapse to a full
        # multiple_of chunk (e.g. w=128, multiple_of=64, downscale=0.99 ->
        # target_w=126 -> floor gives 64, a 50% drop instead of ~1%).
        # Rounding halves the worst case to +/- multiple_of/2.
        new_w = round(target_w / multiple_of) * multiple_of
        new_h = round(target_h / multiple_of) * multiple_of

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # A downscale must never enlarge past the original size. Rounding up
        # can occasionally push past it (target close to a grid line just
        # above), so fall back to flooring in that case, which is
        # guaranteed <= the original dimension.
        if new_w > w:
            new_w = max(1, (int(w) // multiple_of) * multiple_of)
        if new_h > h:
            new_h = max(1, (int(h) // multiple_of) * multiple_of)

        if new_h < multiple_of or new_w < multiple_of:
            logging.warning(
                f"Magik Zoom Out: downscaled size collapsed to {new_h}x{new_w} (multiple_of={multiple_of}). "
                "Consider a smaller multiple_of, a higher downscale, or fewer sampling_steps."
            )

        return new_h, new_w

    # ------------------------------------------------------------------
    # Step 2 (ApplyCosmosReferenceLatent): patch the model so the reference
    # latent is appended along the temporal dim during denoising, then
    # sliced back off afterward. Ported from Mirumo0u0/ComfyUI-Cosmos-Reference.
    # ------------------------------------------------------------------
    @classmethod
    def _apply_reference_latent(cls, model: ModelPatcher, latent: dict) -> ModelPatcher:
        m = model.clone()

        if not isinstance(m.model, Anima):
            logging.warning(
                "Magik Zoom Out: model is not an Anima-family model, skipping Cosmos reference-latent patch."
            )
            return m

        extra_conds = m.get_model_object("extra_conds")
        process_latent_in = m.get_model_object("process_latent_in")
        ref_latents = {REF_LATENT_KEY: latent}

        m.add_object_patch(
            "extra_conds",
            cls._extra_conds_with_reference(extra_conds, process_latent_in, ref_latents),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            REF_LATENT_WRAPPER_KEY,
            cls._diffusion_reference_wrapper,
        )

        return m

    @staticmethod
    def _extra_conds_with_reference(extra_conds, process_latent_in, ref_latents: dict):
        def _wrapped(**kwargs):
            out = extra_conds(**kwargs)
            latents = [process_latent_in(l["samples"]) for l in ref_latents.values()]
            out[REF_LATENT_COND_KEY] = comfy.conds.CONDList(latents)
            return out

        return _wrapped

    @staticmethod
    def _diffusion_reference_wrapper(executor, *args, **kwargs):
        x = args[0]
        x_temporal_dim = x.shape[2]
        ref_latents = kwargs.get(REF_LATENT_COND_KEY)

        newargs = list(args)
        if ref_latents is not None:
            for ref in ref_latents:
                if ref.ndim == 4:
                    ref = ref.unsqueeze(2)
                x = torch.cat([x, ref.to(dtype=x.dtype, device=x.device)], dim=2)
        newargs[0] = x

        # Don't let ref_latents propagate further down the wrapper chain -
        # this wrapper is the only consumer of it.
        newkwargs = dict(kwargs)
        newkwargs.pop(REF_LATENT_COND_KEY, None)

        result = executor(*newargs, **newkwargs)[:, :, :x_temporal_dim]
        return result

    # ------------------------------------------------------------------
    # Step 3 (KSampler): identical to comfy_extras' common_ksampler
    # ------------------------------------------------------------------
    @staticmethod
    def _ksample(model: ModelPatcher, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str,
                 positive, negative, latent: dict, denoise: float) -> dict:
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None)
        )

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed,
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return out
