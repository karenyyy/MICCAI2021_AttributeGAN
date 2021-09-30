from src.util.components import *
from src.util.utils import *

norm_class = nn.SyncBatchNorm

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(chan * 2, chan, 1))))
])

class Generator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            latent_dim=512,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            attn_res_layers=[],
            freq_chan_attn=False
    ):
        super().__init__()
        resolution = log2(image_size)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.image_size = image_size

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim*2, 4),
            norm_class(latent_dim*2),
            nn.GLU(dim=1)
        )
        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = attn_and_ff(chan_in)

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                sle = GlobalContext(
                    chan_in=chan_out,
                    chan_out=sle_chan_out
                )

            layer = nn.ModuleList([
                nn.Sequential(
                    upsample(),
                    Blur(),
                    nn.Conv2d(chan_in + chan_out//4 if res == 2 else chan_in, chan_out * 2, 3, padding=1),
                    norm_class(chan_out * 2),
                    nn.GLU(dim=1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)
        self.label_upsample = nn.Upsample(scale_factor=2)
        self.init_label_preprocess = nn.Sequential(
            nn.ConvTranspose2d(25, features[0]//4, 4, 1, 0),
            nn.SyncBatchNorm(features[0]//4),
            nn.ReLU())


    def forward(self, x, labels):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)

        encoded_labels = []
        for label_idx in range(5):
            label = labels[label_idx]
            encoded_labels.append(label)
        label = torch.cat(encoded_labels, 1)
        label = label.unsqueeze(2).unsqueeze(3)
        label = self.init_label_preprocess(label.type(torch.cuda.FloatTensor))
        x = torch.cat((x, label), 1)

        x = F.normalize(x, dim=1)

        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):

            if exists(attn):
                x = attn(x)

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)








