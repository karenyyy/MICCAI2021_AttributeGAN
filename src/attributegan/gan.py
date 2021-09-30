from src.attributegan.model import *
from src.util.utils import *
from src.util.data_loader import *

df = pd.read_csv('data/attribute_with_levels_freqids_and_cl_new_labels.csv')
dictt = json.load(open('data/dataset.json', ))


class Trainer():
    def __init__(
            self,
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            optimizer='adam',
            num_workers=None,
            latent_dim=512,
            image_size=512,
            num_image_tiles=8,
            fmap_max=512,
            transparent=False,
            greyscale=False,
            batch_size=12,
            gp_weight=10,
            gradient_accumulate_every=1,
            attn_res_layers=[],
            freq_chan_attn=False,
            disc_output_size=5,
            antialias=False,
            lr=2e-4,
            lr_mlp=1.,
            ttur_mult=1.,
            save_every=1000,
            evaluate_every=1000,
            aug_prob=None,
            aug_types=['translation', 'cutout'],
            dataset_aug_prob=0.,
            calculate_fid_every=None,
            calculate_fid_num_images=12800,
            clear_fid_cache=False,
            is_ddp=False,
            rank=0,
            world_size=1,
            log=False,
            amp=False,
            *args,
            **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.writer = SummaryWriter(os.path.join(self.models_dir, self.name))

        self.config_path = self.models_dir / name / '.config.json'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.generator_top_k_gamma = 0.99
        self.generator_top_k_frac = 0.5

        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.syncbatchnorm = is_ddp

        self.amp = amp
        self.G_scaler = GradScaler(enabled=self.amp)
        self.D_scaler = GradScaler(enabled=self.amp)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    def init_GAN(self):
        args, kwargs = self.GAN_params

        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr=self.lr,
            latent_dim=self.latent_dim,
            attn_res_layers=self.attn_res_layers,
            freq_chan_attn=self.freq_chan_attn,
            image_size=self.image_size,
            ttur_mult=self.ttur_mult,
            fmap_max=self.fmap_max,
            disc_output_size=self.disc_output_size,
            transparent=self.transparent,
            greyscale=self.greyscale,
            rank=self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.greyscale = config.pop('greyscale', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    def set_data_src(self):
        global df
        self.batch_df = df[df['freq_id'] == 0]
        self.dataset = HistologyDataset(self.batch_df)

        imbalance_sampler = ImbalancedDatasetSampler(self.dataset, attr=self.steps % 5)

        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 num_workers=self.num_workers,
                                                 batch_size=self.batch_size // self.world_size,
                                                 sampler=imbalance_sampler,
                                                 drop_last=True,
                                                 pin_memory=True)
        self.loader = cycle(dataloader)

        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob = default(self.aug_prob, 0)
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G.cuda() if not self.is_ddp else self.G_ddp
        D = self.GAN.D.cuda() if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug.cuda() if not self.is_ddp else self.D_aug_ddp

        amp_context = autocast if self.amp else null_context

        self.contrastive_criterion = Conditional_Contrastive_loss(self.rank, self.batch_size, True)

        path_batch, image_batch, label_batch, cl_labels = next(self.loader)

        image_batch.cuda(self.rank)
        image_batch.requires_grad_()
        label_batch = torch.transpose(torch.stack(label_batch, dim=0), 0, 1)
        label_batch2 = rearrange(label_batch, 'b c -> c b')
        real_one_hot_labels = []
        for i in range(label_batch2.size(0)):
            mask = torch.HalfTensor(label_batch.size(0), 5) \
                .zero_() \
                .scatter_(1, label_batch2[i].unsqueeze(1), 1)
            real_one_hot_labels.append(mask)

        # train discriminator
        for p in G.parameters():
            p.requires_grad = False
        for p in D.parameters():
            p.requires_grad = True
        for p in D_aug.parameters():
            p.requires_grad = True

        self.GAN.D_opt.zero_grad()
        disc_loss = 0
        t = set_temperature("discrete", 1.0, 1.0, self.steps, 1, 50000)

        for i in range(self.gradient_accumulate_every):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            with amp_context():
                generated_images = G(latents, real_one_hot_labels)

                fake_output, fake_output_32x32 = D(generated_images.detach(), real_one_hot_labels,
                                                       calc_aux_loss=False,
                                                       calc_contra_loss=False)
                real_output, real_output_32x32, real_aux_loss, cls_proxy, cls_embed = D(image_batch,
                                                                                            real_one_hot_labels,
                                                                                            calc_aux_loss=True,
                                                                                            calc_contra_loss=True)

                real_output_loss = real_output
                fake_output_loss = fake_output

                divergence = hinge_loss(real_output_loss, fake_output_loss)
                divergence_32x32 = hinge_loss(real_output_32x32, fake_output_32x32)
                disc_loss += divergence + divergence_32x32
                disc_loss += real_aux_loss

                real_cls_mask = make_mask(cl_labels, 94, self.rank)
                contra_loss = self.contrastive_criterion(cls_embed, cls_proxy, real_cls_mask,
                                                         cl_labels, t, 0)
                if not contra_loss.isnan():
                    disc_loss += contra_loss

            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            self.D_scaler.scale(disc_loss).backward()
            total_disc_loss += disc_loss

        self.last_recon_loss = real_aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        self.D_scaler.step(self.GAN.D_opt)
        self.D_scaler.update()

        # train generator
        for p in G.parameters():
            p.requires_grad = True
        for p in D.parameters():
            p.requires_grad = False
        for p in D_aug.parameters():
            p.requires_grad = False

        self.GAN.G_opt.zero_grad()
        gen_loss = 0

        for i in range(self.gradient_accumulate_every):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            with amp_context():
                generated_images = G(latents, real_one_hot_labels)

                fake_output, fake_output_32x32, cls_proxy, cls_embed = D(generated_images, real_one_hot_labels,
                                                                             calc_aux_loss=False,
                                                                             calc_contra_loss=True)


                fake_output_loss = fake_output.mean(dim=1) + fake_output_32x32.mean(dim=1)

                loss = fake_output_loss.mean()
                gen_loss += loss

                real_cls_mask = make_mask(cl_labels, 94, self.rank)

                contra_loss = self.contrastive_criterion(cls_embed, cls_proxy, real_cls_mask,
                                                             cl_labels, t, 0)
                if not contra_loss.isnan():
                    gen_loss += contra_loss

                gen_loss = gen_loss / self.gradient_accumulate_every

            self.G_scaler.scale(gen_loss).backward()
            total_gen_loss += gen_loss

        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        self.G_scaler.step(self.GAN.G_opt)
        self.G_scaler.update()

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every), num_image_tiles=self.num_image_tiles)

        self.steps += 1

    @torch.no_grad()
    def generate_truncated(self, G, latent, label):
        generated_images = generate_in_chunks(self.batch_size, G, 5, latent, label)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=8, trunc=1.0):
        self.GAN.eval()
        for attr in ['cell_crowding', 'cell_polarity', 'mitosis', 'nucleoli', 'pleomorphism']:

            print(f'\n---------------- {attr} --------------\n')
            labels = get_attribute_vectors(1, attribute=attr)
            latents_lst = [torch.randn((1, self.latent_dim)).cuda() for _ in
                           range(num_image_tiles)]

            for n in range(LEVEL_DIM[attr]):
                label_for_current_level = labels[n]
                label_for_current_level = label_for_current_level.unsqueeze(0)
                label_for_current_level = rearrange(label_for_current_level, 'b c -> c b')
                one_hot_labels = []
                for k in range(label_for_current_level.size(0)):
                    mask = torch.HalfTensor(label_for_current_level.size(1), 5) \
                        .zero_() \
                        .scatter_(1, label_for_current_level[k].type(torch.LongTensor).unsqueeze(1), 1)

                    one_hot_labels.append(mask.cuda(self.rank))

                for i in range(num_image_tiles):
                    num_rows = num_image_tiles

                    generated_images = self.GAN.G(latents_lst[i], one_hot_labels).clamp_(0., 1.)
                    generated_ema_images = self.GAN.GE(latents_lst[i], one_hot_labels).clamp_(0., 1.)

                    if i == 0:
                        row_generated_images = generated_images
                        row_generated_ema_images = generated_ema_images
                    else:
                        row_generated_images = torch.cat((row_generated_images, generated_images), 0)
                        row_generated_ema_images = torch.cat((row_generated_ema_images, generated_ema_images), 0)
                if n == 0:
                    total_generated_images = row_generated_images
                    total_generated_ema_images = row_generated_ema_images
                else:
                    total_generated_images = torch.cat((total_generated_images, row_generated_images), 0)
                    total_generated_ema_images = torch.cat((total_generated_ema_images, row_generated_ema_images), 0)

            fake_img = torchvision.utils.make_grid(total_generated_images.detach(), normalize=True,
                                                   scale_each=True)
            self.writer.add_image(f'{attr}', fake_img, self.steps)

            torchvision.utils.save_image(total_generated_images, str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}.png'),
                                         nrow=num_rows,
                                         normalize=True)
            print(str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}.png'), 'saved!!')

            fake_img = torchvision.utils.make_grid(total_generated_ema_images.detach(), normalize=True,
                                                   scale_each=True)
            self.writer.add_image(f'{attr}_EMA', fake_img, self.steps)
            torchvision.utils.save_image(total_generated_ema_images, str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}-ema.png'),
                                         nrow=num_rows,
                                         normalize=True)
            print(str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}-ema.png'), 'saved!!')

        ori_latents = torch.randn((8, self.latent_dim)).cuda()
        ori_latents = ori_latents.repeat(5, 1)

        for attr in ['cell_crowding', 'cell_polarity', 'mitosis', 'nucleoli', 'pleomorphism']:
            labels = get_attribute_vectors(num_rows, attribute=attr)
            latents = ori_latents[:LEVEL_DIM[attr] * 8, :]

            print(f'\n---------------- {attr} --------------\n')

            generated_images = self.generate_truncated(self.GAN.G, latents, labels)
            fake_img = torchvision.utils.make_grid(generated_images.detach(), normalize=True,
                                                   scale_each=True)
            self.writer.add_image(f'{attr}_double_check', fake_img, self.steps)

            torchvision.utils.save_image(generated_images, str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}.png'),
                                         nrow=num_rows)
            print(str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}.png'), 'saved!!')

            generated_images = self.generate_truncated(self.GAN.GE, latents, labels)
            fake_img = torchvision.utils.make_grid(generated_images.detach(), normalize=True, scale_each=True)
            self.writer.add_image(f'{attr} (EMA)_double_check', fake_img, self.steps)
            torchvision.utils.save_image(generated_images, str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}-ema.png'),
                                         nrow=num_rows)
            print(str(
                Path(self.results_dir) / self.name / f'{attr}_{str(num)}-ema.png'), 'saved!!')

    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles=8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1, print_version=True):
        self.load_config()

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data['D_scaler'])

    def get_checkpoints(self):
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums
