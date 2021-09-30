from src.attributegan.gan import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run_training(
        rank, world_size, model_arg, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12351'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_arg.update(
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size
    )
    print('model_arg: ', model_arg)
    model = Trainer(**model_arg)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src()

    for _ in tqdm(range(num_train_steps - model.steps), initial=model.steps, total=num_train_steps, mininterval=10.,
                  desc=f'{name}<{data}>'):
        model.train()
        if is_main and _ % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()


def train_ddp(
        data='/data/AttributeGAN/data/Images',  # change to the path where you save the images
        results_dir='saved_results',
        models_dir='saved_models',
        name='histology',
        new=False,
        load_from=-1,
        image_size=512,
        optimizer='adam',
        fmap_max=512,
        batch_size=64,
        gradient_accumulate_every=1,
        num_train_steps=50000,
        learning_rate=2e-4,
        save_every=1000,
        evaluate_every=100,
        aug_prob=None,
        aug_types=['translation', 'offset', 'offset_h', 'offset_v'],
        dataset_aug_prob=0.,
        attn_res_layers=[32, 64],
        disc_output_size=5,
        antialias=False,
        num_image_tiles=8,
        num_workers=8,
        multi_gpus=True,
        calculate_fid_every=None,
        calculate_fid_num_images=12800,
        clear_fid_cache=False,
        seed=42,
        amp=False,
):
    model_args = dict(
        name=name,
        results_dir=results_dir,
        models_dir=models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        attn_res_layers=attn_res_layers,
        disc_output_size=disc_output_size,
        antialias=antialias,
        image_size=image_size,
        num_image_tiles=num_image_tiles,
        optimizer=optimizer,
        num_workers=num_workers,
        fmap_max=fmap_max,
        lr=learning_rate,
        save_every=save_every,
        evaluate_every=evaluate_every,
        aug_prob=aug_prob,
        aug_types=aug_types,
        dataset_aug_prob=dataset_aug_prob,
        calculate_fid_every=calculate_fid_every,
        calculate_fid_num_images=calculate_fid_num_images,
        clear_fid_cache=clear_fid_cache,
        amp=amp
    )

    world_size = 2
    mp.spawn(run_training,
             args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    fire.Fire(train_ddp)
