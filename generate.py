from src.attributegan.model import *
from src.util.utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

PRETRAIN_MODEL = 'pretrain_model/pretrained_model.pt'

GAN_ours = LightweightGAN(
    optimizer='adam',
    lr=4e-4,
    latent_dim=512,
    attn_res_layers=[32, 64],
    freq_chan_attn=1,
    image_size=512,
    ttur_mult=1,
    fmap_max=512,
    disc_output_size=5,
    transparent=False,
    greyscale=False,
    rank=0,
    fmap_inverse_coef=12
)


GAN_noattn = LightweightGAN(
    optimizer='adam',
    lr=4e-4,
    latent_dim=512,
    attn_res_layers=[32, 64],
    freq_chan_attn=1,
    image_size=512,
    ttur_mult=1,
    fmap_max=512,
    disc_output_size=5,
    transparent=False,
    greyscale=False,
    rank=0,
    fmap_inverse_coef=12
)


GAN_nocl = LightweightGAN(
    optimizer='adam',
    lr=4e-4,
    latent_dim=512,
    attn_res_layers=[32, 64],
    freq_chan_attn=1,
    image_size=512,
    ttur_mult=1,
    fmap_max=512,
    disc_output_size=5,
    transparent=False,
    greyscale=False,
    rank=0,
    fmap_inverse_coef=12
)


@torch.no_grad()
def generate_truncated(G, latent, label):
    generated_images = generate_in_chunks(8, G, 5, latent, label)
    return generated_images.clamp_(0., 1.)

def get_attribute_vectors_vis(num_rows, attribute='cell_crowding'):
    if attribute == 'cell_crowding':
        labels = torch.stack([torch.stack(
            [torch.Tensor([l, 0, 2, 1, 2]) for _ in range(num_rows)], dim=0)
            for l in np.arange(1, 5)], dim=0)

    elif attribute == 'cell_polarity':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([2, 0, 2, 1, 2]) for _ in range(num_rows)], dim=0),
                torch.stack(
                    [torch.Tensor([1, 2, 2, 1, 2]) for _ in range(num_rows)], dim=0),
                torch.stack(
                    [torch.Tensor([2, 2, 2, 1, 2]) for _ in range(num_rows)], dim=0)
            ], dim=0)

    elif attribute == 'mitosis':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([2, 2, l, 1, 2]) for _ in range(num_rows)], dim=0)
                for l in np.arange(1, 4)], dim=0)

    elif attribute == 'nucleoli':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([2, 2, 1, 0, 2]) for _ in range(num_rows)], dim=0),
                torch.stack(
                    [torch.Tensor([2, 2, 1, 1, 2]) for _ in range(num_rows)], dim=0),
            ], dim=0)

    elif attribute == 'pleomorphism':
        labels = torch.stack(
            [torch.stack(
                [torch.Tensor([2, 2, 1, 1, l]) for _ in range(num_rows)], dim=0)
                for l in np.arange(1, 5)], dim=0)
    else:
        print('Please check the attributes.')

    labels = rearrange(labels, 'n b d -> (n b) d')
    labels = labels.cuda()
    return labels


with torch.no_grad():

    load_data_ours = torch.load(PRETRAIN_MODEL)

    GAN_ours.load_state_dict(load_data_ours['GAN'])
    GAN_ours.eval()

    ori_latents = torch.randn((8, 512)).cuda()
    ori_latents = ori_latents.repeat(5, 1)

    for attr in ['cell_crowding', 'cell_polarity', 'mitosis', 'nucleoli', 'pleomorphism']:
        labels = get_attribute_vectors_vis(8, attribute=attr)
        latents = ori_latents[:LEVEL_DIM[attr] * 8, :]

        print(f'\n---------------- {attr} --------------\n')

        generated_images = generate_truncated(GAN_ours.G, latents, labels)
        torchvision.utils.save_image(generated_images, f'results/examples/example_ours_{attr}.png')
        img = mpimg.imread(f'results/examples/example_ours_{attr}.png', 0)
        plt.imshow(img)
        plt.title('our results')
        plt.show()
