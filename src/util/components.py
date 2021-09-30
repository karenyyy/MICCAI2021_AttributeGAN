
from src.util.utils import *

class SimpleDecoder(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out=3,
            num_upsamples=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding=1),
                nn.GLU(dim=1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


def random_hflip(tensor, prob):
    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))


@contextmanager
def null_context():
    yield


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)

def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)

def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)


AUGMENT_FNS = {
    'offset': [rand_offset],
    'offset_h': [rand_offset_h],
    'offset_v': [rand_offset_v],
    'translation': [rand_translation],
}

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, label, prob=0., types=[], detach=False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random.random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, label, **kwargs)

class GlobalContext(nn.Module):
    def __init__(
            self,
            *,
            chan_in,
            chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, rank, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.rank = rank
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).cuda(self.rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp(
            (self.remove_diag(similarity_matrix) - margin) / temperature)

        inst2proxy_positive = torch.exp((self.cosine_similarity(inst_embed, proxy) - margin) / temperature)

        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(
                mask_4_remove_negatives)
            inst2inst_positives = instance_zone * mask_4_remove_negatives
            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive

        denomerator = torch.cat([torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature * (numerator / denomerator)).mean()

        return criterion
