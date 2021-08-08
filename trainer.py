import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from generator import Generator
from discriminator import Discriminator
from T2IDataset import Text2ImageDataset
import numpy as np
from PIL import Image
from visualdl import LogWriter


class Trainer(object):
    def __init__(self, batch_size, num_workers, epochs, split):
        self.G = Generator()
        self.D = Discriminator()
        self.noise_dim = 100
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = epochs
        self.dataset = Text2ImageDataset('Data/flowers.hdf5', split=self.split)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=self.num_workers)
        self.scheduler_G = paddle.optimizer.lr.LambdaDecay(learning_rate=0.0001, lr_lambda=lambda x: 0.95**x)
        self.scheduler_D = paddle.optimizer.lr.LambdaDecay(learning_rate=0.0004, lr_lambda=lambda x: 0.95**x)
        self.optD = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999,
                                          parameters=self.D.parameters())
        self.optG = paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999,
                                          parameters=self.G.parameters())

    def train(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        self.D.train()
        self.G.train()
        with LogWriter(logdir='Data') as writer:
            for epoch in range(self.num_epochs):
                iter = 0
                for sample in self.dataloader():
                    iter += 1
                    right_images = sample['right_images'].cuda()
                    right_embed = sample['right_embed'].cuda()
                    wrong_images = sample['wrong_images'].cuda()
                    inter_embed = sample['inter_embed'].cuda()
                    real_labels = paddle.ones([right_images.shape[0]]).cuda()
                    fake_labels = paddle.zeros([right_images.shape[0]]).cuda()
                    smooth_real_labels = real_labels - 0.1
                    smooth_real_labels = smooth_real_labels.cuda()
                    # train net_D
                    self.optD.clear_grad()
                    outputs, activation_real = self.D(right_images, right_embed)
                    real_loss = criterion(outputs, smooth_real_labels)
                    real_score = outputs
                    outputs, _ = self.D(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                    noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
                    noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
                    fake_images = self.G(right_embed, noise)
                    outputs, _ = self.D(fake_images.detach(), right_embed)
                    fake_loss = criterion(outputs, fake_labels)
                    fake_score = outputs

                    d_loss = fake_loss + real_loss + wrong_loss
                    d_loss.backward()
                    self.optD.step()

                    # train netG
                    self.optG.clear_grad()
                    outputs, activation_fake = self.D(fake_images, right_embed)
                    _, activation_real = self.D(right_images, right_embed)
                    g_loss = criterion(outputs, real_labels)
                    noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
                    noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
                    inter_images = self.G(inter_embed, noise)
                    outputs, _ = self.D(inter_images, inter_embed)
                    g_loss_inter = criterion(outputs, real_labels)
                    g_loss = g_loss + g_loss_inter
                    g_loss.backward()
                    self.optG.step()
                    # self.optG.clear_grad()
                    # noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
                    # noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
                    # fake_images = self.G(right_embed, noise)
                    # outputs, activation_fake = self.D(fake_images, right_embed)
                    # _, activation_real = self.D(right_images, right_embed)
                    # activation_fake = paddle.mean(activation_fake, 0)
                    # activation_real = paddle.mean(activation_real, 0)
                    # g_loss = criterion(outputs, real_labels) + 100 * l2_loss(activation_fake, activation_real.detach()) + 50 * l1_loss(fake_images, right_images)
                    # g_loss.backward()
                    # self.optG.step()
                    print('[%d/%d][%d/%d] Loss_D: %.3f  Loss_G: %.3f  D(X): %.3f  D(G(x)):  %.3f'
                          % (epoch, self.num_epochs, iter, len(self.dataloader), d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
                # self.scheduler_G.step()
                # self.scheduler_D.step()
                writer.add_scalar(tag='loss_D_train', value=d_loss.item(), step=epoch)
                writer.add_scalar(tag='loss_G_train', value=g_loss.item(), step=epoch)
                writer.add_scalar(tag='D(x)_train', value=real_score.mean().item(), step=epoch)
                writer.add_scalar(tag='D(G(x)_train', value=fake_score.mean().item(), step=epoch)
                fake_images = (fake_images + 1) / 2.0
                out_img = fake_images.detach().numpy()[0].transpose((1, 2, 0)) * 255
                out_img = Image.fromarray(out_img.astype(np.uint8))
                out_img.save(rf"image/{epoch}.png")
                if (epoch+1) % 10 == 0:
                    paddle.save(self.G.state_dict(), 'model/netG_%03d.pth' % (epoch+1))
                    paddle.save(self.D.state_dict(), 'model/netD_%03d.pth' % (epoch+1))

    def sample(self):
        self.G.load_dict(paddle.load('netG_200_train.pth'))
        self.G.train()
        save_dir = 'models/'
        for s in self.dataloader():
            right_images = s['right_images']
            right_embed = s['right_embed']
            txt = s['txt']
            noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
            noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
            fake_images = self.G(right_embed, noise)
            fake_images = (fake_images + 1) / 2.0
            for image, t in zip(fake_images, txt):
                im = image.detach().numpy().transpose((1, 2, 0)) * 255
                #print(im)
                im = Image.fromarray(im.astype(np.uint8))
                im.save(save_dir + '{0}.png'.format(t.replace("/", "")[:100]))
                print(t)