from src.util.components import *
from src.attributegan.discriminator import *
from src.attributegan.generator import *
from src.util.utils import *


class LightweightGAN(nn.Module):
    def __init__(
            self,
            *,
            latent_dim,
            image_size,
            optimizer="adam",
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            disc_output_size=5,
            attn_res_layers=[],
            freq_chan_attn=False,
            ttur_mult=1.,
            lr=2e-4,
            rank=0,
            ddp=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size=image_size,
            latent_dim=latent_dim,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            transparent=transparent,
            greyscale=greyscale,
            attn_res_layers=attn_res_layers,
            freq_chan_attn=freq_chan_attn
        )

        self.G = Generator(**G_kwargs)

        self.D = Discriminator(
            image_size=image_size,
            fmap_max=fmap_max,
            fmap_inverse_coef=fmap_inverse_coef,
            transparent=transparent,
            greyscale=greyscale,
            attn_res_layers=attn_res_layers,
            disc_output_size=disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr=lr * ttur_mult, betas=(0.5, 0.9))


        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented
