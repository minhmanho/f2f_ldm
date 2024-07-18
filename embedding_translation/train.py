import torch
import torch.nn as nn
import glob
import argparse
import os
import random
import os.path as osp
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd

class WGAN_GP_Loss(nn.Module):
    def __init__(self):
        super(WGAN_GP_Loss, self).__init__()
        self.lambda_gp = 10

    def forward(self, real_output, fake_output, real_data, fake_data, discriminator):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(real_data.device)
        interpolated = autograd.Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
        interpolated_output = discriminator(interpolated)

        grad_outputs = torch.ones_like(interpolated_output).to(real_data.device)
        gradients = autograd.grad(outputs=interpolated_output, inputs=interpolated,
                                  grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        wasserstein_loss = torch.mean(fake_output) - torch.mean(real_output)
        wgan_gp_loss = wasserstein_loss + self.lambda_gp * gradient_penalty 

        return wgan_gp_loss

class EmbeddingsDataset(Dataset):
    def __init__(self, data_root, phase='train', feat_type=None):
        def load_embeddings(paths):
            embeddings = []
            for path in tqdm(paths):
                if feat_type == "batch":
                    embeddings.extend(list(torch.load(path).squeeze()))
                else:
                    embeddings.append(torch.load(path).squeeze())
            return embeddings
        
        self.embeddings_A = load_embeddings(glob.glob(osp.join(data_root, phase+"A", "*.pt")))
        self.embeddings_B = load_embeddings(glob.glob(osp.join(data_root, phase+"B", "*.pt")))

        print(osp.join(data_root, phase+"A: "), len(self.embeddings_A))
        print(osp.join(data_root, phase+"B: "), len(self.embeddings_B))

    def __len__(self):
        return len(self.embeddings_A)
    
    def __getitem__(self, idx):
        embedding_A = self.embeddings_A[idx % len(self.embeddings_A)]
        embedding_B = self.embeddings_B[random.randint(0, len(self.embeddings_B) - 1)]
        return embedding_A, embedding_B

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        encoder_channels = [in_channels, 256, 128, 64, 32]
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_channels[i], encoder_channels[i+1]),
                nn.LeakyReLU()
            ) for i in range(len(encoder_channels) - 1)
        ])

        decoder_channels = [32, 64, 128, 256, out_channels]
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(decoder_channels[i], decoder_channels[i+1]),
                nn.LeakyReLU()
            ) for i in range(len(decoder_channels) - 1)
        ])

        self.final = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)

        encoder_outputs.pop()

        for decoder_layer in self.decoder:
            x = decoder_layer(x)
            if encoder_outputs:
                x = x + encoder_outputs.pop()

        output = self.final(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        output = self.model(x)
        return output

class CycleGAN(pl.LightningModule):
    def __init__(self, params):
        super(CycleGAN, self).__init__()
        self.params = params

        self.generator_AB = Generator(in_channels=params["input_nc"], out_channels=params["output_nc"])
        self.generator_BA = Generator(in_channels=params["output_nc"], out_channels=params["input_nc"])
        self.discriminator_A = Discriminator(in_channels=params["output_nc"])
        self.discriminator_B = Discriminator(in_channels=params["input_nc"])

        self.wgan_gp_loss_A = WGAN_GP_Loss()
        self.wgan_gp_loss_B = WGAN_GP_Loss()
        self.cycle_loss = nn.L1Loss()

        self.automatic_optimization = False

    def forward(self, real_A):
        fake_B = self.generator_AB(real_A)
        return fake_B

    def forward_cycle(self, real_A, real_B):
        fake_B = self.generator_AB(real_A)
        reconstructed_A = self.generator_BA(fake_B)
        fake_A = self.generator_BA(real_B)
        reconstructed_B = self.generator_AB(fake_A)
        return fake_B, reconstructed_A, fake_A, reconstructed_B

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch
        opt_G, opt_D = self.optimizers()

        loss_D = self.discriminator_loss(real_A, real_B)
        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()

        loss_G = self.generator_loss(real_A, real_B)
        opt_G.zero_grad()
        self.manual_backward(loss_G)
        opt_G.step()

        self.log_dict({"g_loss": loss_G, "d_loss": loss_D}, prog_bar=True)

    def generator_loss(self, real_A, real_B):
        fake_B = self.generator_AB(real_A)
        fake_A = self.generator_BA(real_B)
        rec_A = self.generator_BA(fake_B)
        rec_B = self.generator_AB(fake_A)

        loss_adv_G_A = -torch.mean(self.discriminator_A(fake_A))
        loss_adv_G_B = -torch.mean(self.discriminator_B(fake_B))
        loss_cycle_A = self.cycle_loss(rec_A, real_A)
        loss_cycle_B = self.cycle_loss(rec_B, real_B)

        loss_G = loss_adv_G_A + loss_adv_G_B + loss_cycle_A + loss_cycle_B
        return loss_G

    def discriminator_loss(self, real_A, real_B):
        fake_B = self.generator_AB(real_A).detach()
        fake_A = self.generator_BA(real_B).detach()

        real_output_A = self.discriminator_A(real_A)
        fake_output_A = self.discriminator_A(fake_A)
        loss_discriminator_A = self.wgan_gp_loss_A(real_output_A, fake_output_A, real_A, fake_A, self.discriminator_A)

        real_output_B = self.discriminator_B(real_B)
        fake_output_B = self.discriminator_B(fake_B)
        loss_discriminator_B = self.wgan_gp_loss_B(real_output_B, fake_output_B, real_B, fake_B, self.discriminator_B)

        loss_D = loss_discriminator_A + loss_discriminator_B
        return loss_D

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch
        loss_G = self.generator_loss(real_A, real_B)
        self.log_dict({"val_g_loss": loss_G}, prog_bar=True)
        return loss_G

    def configure_optimizers(self):
        # Optimizers
        optimizer_G = torch.optim.Adam(
            list(self.generator_AB.parameters()) + list(self.generator_BA.parameters()), lr=params['lr']
        )
        optimizer_D = torch.optim.Adam(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()), lr=params['lr']
        )
        return optimizer_G, optimizer_D

class DataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def setup(self, stage=None):
        self.train_dataset = EmbeddingsDataset(self.params['dataroot'], phase='train', feat_type=params['feat_type'])
        self.val_dataset = EmbeddingsDataset(self.params['dataroot'], phase='val', feat_type=params['feat_type'])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['batch_size'], num_workers=self.params['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params['batch_size'], num_workers=self.params['num_workers'], shuffle=False)
    
def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--feat_type', type=str, default=None)
    parser.add_argument('--feat_dim', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--save_name', type=str, default='cycleGAN_wgan_gp')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    params = {
        'dataroot': args.dataroot,
        'batch_size': args.batch_size,
        'num_workers': 16,
        'input_nc': args.feat_dim,
        'output_nc': args.feat_dim,
        'lr': 0.0002,
        'epochs': 10000,
        'feat_type': args.feat_type,
    }

    save_name = args.save_name
    if params['feat_type'] is not None:
        save_name += f'_{params["feat_type"]}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_g_loss',
        dirpath='checkpoints/',
        filename=save_name+'-{epoch:02d}-{val_g_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    tb_logger = pl.loggers.TensorBoardLogger('logs/', name=save_name)
    model = CycleGAN(params)
    data = DataModule(params)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=params['epochs'],
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )
    trainer.fit(model, data)
