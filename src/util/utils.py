from config import *

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']
LEVEL_DIM = {"cell_crowding": 5,
             "cell_polarity": 4,
             "mitosis": 4,
             "nucleoli": 3,
             "pleomorphism": 5}

def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def default(val, d):
    return val if exists(val) else d

def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)


class NanException(Exception):
    pass

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        return self.g * self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def get_attribute_vectors(num_rows, attribute='cell_crowding'):
    if attribute == 'cell_crowding':
        labels = torch.stack([torch.stack(
                                        [torch.Tensor([l] + [0,1,1,2]) for _ in range(num_rows)], dim=0)
                                     for l in np.arange(5)], dim=0)

    elif attribute == 'cell_polarity':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([1] + [l] + [1,1,2]) for _ in range(num_rows)], dim=0)
                for l in np.arange(4)], dim=0)

    elif attribute == 'mitosis':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([2]*2 + [l] + [1,3]) for _ in range(num_rows)], dim=0)
                for l in np.arange(4)], dim=0)

    elif attribute == 'nucleoli':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([1,3,1] + [l] +[2]) for _ in range(num_rows)], dim=0)
                for l in np.arange(3)], dim=0)

    elif attribute == 'pleomorphism':
        labels = torch.stack([torch.stack(
            [torch.Tensor([3,0,1,1] + [l]) for _ in range(num_rows)], dim=0)
            for l in np.arange(5)], dim=0)
    else:
        print('Please check the attributes.')
    labels = rearrange(labels, 'n b d -> (n b) d')
    labels = labels.cuda()
    return labels


def generate_in_chunks(max_batch_size, model, num_classes, *args):
    # split into 5 parts
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))

    for idx in range(len(split_args)):
        arg = split_args[idx]
        arg = list(arg)
        label_batch = arg[1]
        label_batch2 = rearrange(label_batch, 'b c -> c b')

        one_hot_labels = []
        for i in range(label_batch2.size(0)):
            mask = torch.HalfTensor(label_batch2.size(1), num_classes) \
                .zero_() \
                .scatter_(1, label_batch2[i].type(torch.LongTensor).unsqueeze(1), 1)

            one_hot_labels.append(mask.cuda(arg[0].device))

        arg[1] = one_hot_labels
        split_args[idx] = tuple(arg)

    # print('** split_args: ', split_args)
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] =+1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


def set_temperature(tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if tempering_type == 'continuous':
        t = start_temperature + step_count * (end_temperature - start_temperature) / total_step
    elif tempering_type == 'discrete':
        tempering_interval = total_step // (tempering_step + 1)
        t = start_temperature + \
            (step_count // tempering_interval) * (end_temperature - start_temperature) / tempering_step
    else:
        t = start_temperature
    return t

