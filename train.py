import torch
import hydra
from trainer.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

OmegaConf.register_new_resolver(
    "join",
    lambda sep, xs: sep.join(str(x) for x in xs)
)

@hydra.main(config_path='configs', config_name='config.yaml')
def train_pipeline(cfg: DictConfig):
    print(cfg)

    if not cfg.wandb.disabled:
        wandb.init(
            entity="",
            project='skci',
            group=cfg.wandb.group,
            name=f"{cfg.data.type}_dseed-{cfg.data.data_seed}_tseed-{cfg.train.seed}",
            tags=cfg.wandb.tags + [cfg.model_a.kernel_type, cfg.data.type, 
                                   cfg.wandb.task, cfg.train.Vt_type], # tags: kernel type: [rbf, linear], dataset type ['type1, type2'], purpose: [debug, exp]
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
    else:
        wandb.init(mode="disabled")  # completely disables logging
    # initialize data
    datagen = instantiate(cfg.data)

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize network
    kernel_a = instantiate(cfg.model_a).to(device)
    kernel_b = instantiate(cfg.model_b).to(device)
    kernel_c = instantiate(cfg.model_c).to(device)
    kernel_ca = instantiate(cfg.model_ca).to(device)
    kernel_cb = instantiate(cfg.model_cb).to(device)


    # initialize the trainer object and fit the network to the task
    trainer = Trainer(cfg.train, kernel_a, kernel_b, kernel_c, kernel_ca, kernel_cb, datagen, device)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    train_pipeline()