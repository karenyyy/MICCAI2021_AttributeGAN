from src.util.components import *
from src.util.utils import *

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(chan * 2, chan, 1))))
])

class Discriminator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            disc_output_size=5,
            attn_res_layers=[]
    ):
        super().__init__()
        resolution = log2(image_size)

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** resolution

            attn = None
            if image_width in attn_res_layers:
                attn = attn_and_ff(chan_in)

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding=1),
            Residual(Rezero(GSA(dim=64, norm_queries=True, batch_norm=False))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    nn.Conv2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(Rezero(GSA(dim=32, norm_queries=True, batch_norm=False))),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in=last_chan, chan_out=init_channel)
        self.decoder2 = SimpleDecoder(chan_in=features[-2][-1], chan_out=init_channel) if resolution >= 9 else None

        self.linear_emd = nn.Linear(in_features=128 ** 2,
                  out_features=25*16)
        self.linear_out = nn.Linear(in_features=25, out_features=25)
        self.linear_out32x32 = nn.Linear(in_features=25, out_features=25)

        self.embedding = nn.Embedding(num_embeddings=25, embedding_dim=16)

        self.to_feature_space = nn.Sequential(
                    nn.Conv2d(32, 32, 1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 1, 1)
                )

    def forward(self, x, one_hot_labels, calc_aux_loss=False, calc_contra_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)

            if calc_contra_loss and x.size(-1) == 128:
                emd = self.to_feature_space(x).flatten(1)
                cls_embed_ = self.linear_emd(emd)
                cls_embed_ = F.normalize(cls_embed_, dim=1).cuda(x.device)

                encoded_labels = []
                for label_idx in range(5):
                    label = one_hot_labels[label_idx]
                    encoded_labels.append(label)
                label = torch.cat(encoded_labels, 1)
                cls_proxy_ = self.embedding(label.type(torch.LongTensor).cuda(x.device))
                cls_proxy_ = rearrange(cls_proxy_, 'a b c -> a (b c)')
                cls_proxy_ = F.normalize(cls_proxy_, dim=1)

            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size=recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            select_random_quadrant = lambda rand_quadrant, img: \
            rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random.random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(img_part, size=recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        out_32x32 = out_32x32.flatten(1)

        out = self.linear_out(out)
        out_32x32 = self.linear_out32x32(out_32x32)

        encoded_labels = []
        for label_idx in range(len(one_hot_labels)):
            label = one_hot_labels[label_idx]
            encoded_labels.append(label)
        label = torch.cat(encoded_labels, 1)
        proj = torch.mul(label, out)
        proj_32x32 = torch.mul(label, out_32x32)

        if calc_aux_loss:
            if calc_contra_loss:
                return proj + out, proj_32x32 + out_32x32, aux_loss, cls_proxy_, cls_embed_
            else:
                return proj + out, proj_32x32 + out_32x32, aux_loss
        else:
            if calc_contra_loss:
                return proj + out, proj_32x32 + out_32x32, cls_proxy_, cls_embed_
            else:
                return proj + out, proj_32x32 + out_32x32
