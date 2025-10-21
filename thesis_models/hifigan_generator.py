"""
Source code inspired from:
- fairseq:
    - Core HiFi-GAN: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/hifigan.py
    - Unit HiFi-GAN: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/codehifigan.py
- jik876: https://github.com/jik876/hifi-gan/blob/master/models.py

For detailed model architecture details, refer to the original paper: https://arxiv.org/abs/2010.05646
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn.functional import leaky_relu

from .resblock import ResBlock
from .utils import init_weights


# Empirically, 0.1 worked best in GAN-based vocoders for audio stability
# The small negative slope (0.1) allows a small gradient to flow even for negative activations
# Helpful in preventing "dead" neurons
LEAKY_RELU_SLOPE = 0.1


class UnitHiFiGANGenerator(nn.Module):
    """
    ## Overview
    Unit-based HiFi-GAN Generator that converts discrete speech units into audio waveforms.
    - **Input**:  Discrete unit IDs (LongTensor [B, T])
    - **Output**: Audio waveform (FloatTensor [B, 1, L])

    ## Requirements
    Config dictionary (`config.json`) expected to contain:

    ### Embedding/Input
    - `num_embeddings`: `int`            (e.g., `1000`)
    - `embedding_dim`: `int`             (e.g., `128`)
    - `model_in_dim`: `int`              (e.g., `128` - must match channel count fed to `conv_pre`)

    ### Generator
    - `upsample_initial_channel`: `int`                      (e.g., `512`)
    - `upsample_rates`: `List[int]`                          (e.g., `[5,4,4,2,2]`)
    - `upsample_kernel_sizes`: `List[int]`                   (e.g., `[11,8,8,4,4]`)
    - `resblock_kernel_sizes`: `List[int]`                   (e.g., `[3,7,11]`)
    - `resblock_dilation_sizes`: `List[Tuple[int,int,int]]`  (e.g., `[(1,3,5), ...]`)

    [Link to downloadable config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
    """

    def __init__(self, config):
        super().__init__()

        # Converts unit IDs to become continuous, learnable feature vectors
        # Should have shape (1000, 128)
        self.unit_embed = nn.Embedding(config["num_embeddings"], config["embedding_dim"])

        # Sanity Check: embedding_dim must equal model_in_dim for checkpoint compatibility
        in_dim = config.get("model_in_dim", config["embedding_dim"])
        assert in_dim == config["embedding_dim"], (
            f"model_in_dim ({in_dim}) must equal embedding_dim ({config['embedding_dim']})"
        )

        # Converts embedding vectors into the generator’s internal channels
        # input channel (128) -> output channel (512)
        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=config["upsample_initial_channel"],
                kernel_size=7,
                padding=3
            )
        )

        # Upsampling x Multi-Receptive Field Fusion (MRF) module
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        in_ch = config["upsample_initial_channel"]

        for stride, k_up in zip(config["upsample_rates"], config["upsample_kernel_sizes"]):
            out_ch = in_ch // 2

            # Upsampling (ConvTranspose)
            # At each stage, channels halve (e.g., 512 -> 256 -> 128 -> ...)
            self.ups.append(
                weight_norm(
                    # Upsamples time by stride (e.g., 5, 4, 4, 2, 2)
                    nn.ConvTranspose1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=k_up,
                        stride=stride,
                        padding=(k_up - stride) // 2
                    )
                )
            )

            # MRF
            # For each upsample stage, create several ResBlocks:
            #   e.g., kernels [3, 7, 11] with dilations (1, 3, 5) for each
            stage_blocks = nn.ModuleList()
            for ks, ds in zip(config["resblock_kernel_sizes"], config["resblock_dilation_sizes"]):
                # Each ResBlock has the same channels (out_ch) but different receptive fields
                stage_blocks.append(
                    ResBlock(
                        channels=out_ch,
                        kernel_size=ks,
                        dilations=tuple(ds)
                    )
                )
            self.resblocks.append(stage_blocks)

            # Update for next iteration
            in_ch = out_ch

        # Converts the final features into 1-channel waveform output
        self.conv_post = weight_norm(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=1,
                kernel_size=7,
                padding=3
            )
        )

        self.num_kernels = len(config["resblock_kernel_sizes"])

        # Override default weights to stabilize training and reduce artifacts
        self.conv_pre.apply(init_weights)
        for up in self.ups:
            up.apply(init_weights)
        self.conv_post.apply(init_weights)


    def forward(self, units: torch.LongTensor) -> torch.Tensor:
        # [B,T] -> [B,T,C] -> [B,C,T]
        x = self.unit_embed(units).transpose(1, 2)

        # pre-conv
        x = self.conv_pre(x)

        # Upsample x MRF
        for upsample, stage_blocks in zip(self.ups, self.resblocks):
            x = leaky_relu(x, LEAKY_RELU_SLOPE)
            x = upsample(x)

            # sum & average MRF outputs
            res_outputs = [ resblock(x) for resblock in stage_blocks ]
            x = sum(res_outputs) / self.num_kernels

        # post-conv
        x = leaky_relu(x, LEAKY_RELU_SLOPE)
        x = torch.tanh(self.conv_post(x))
        return x


    def remove_weight_norm(self):
        """Strip weight norm for faster inference"""
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        remove_weight_norm(self.conv_post)
        for stage in self.resblocks:
            for rb in stage:
                rb.remove_weight_norm()
