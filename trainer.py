#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from generator import Generator
from discriminator import Discriminator
from T2IDataset import Text2ImageDataset
import numpy as np
from PIL import Image
from visualdl import LogWriter


# define the trainer
class Trainer(object):
    def __init__(self, batch_size, num_workers, epochs, split, noise_dim, projected_embed_dim, ngf, ndf):
        # initialize
        self.G = Generator(noise_dim, projected_embed_dim, ngf)
        self.D = Discriminator(projected_embed_dim, ndf)
        self.noise_dim = noise_dim
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
        self.D.train()
        self.G.train()
        # write the training process into the log file
        with LogWriter(logdir='Data') as writer:
            for epoch in range(self.num_epochs):
                iter = 0
                for sample in self.dataloader():
                    iter += 1
                    # get the training data
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
                    # get the judgement for real image and right embed
                    outputs, activation_real = self.D(right_images, right_embed)
                    real_loss = criterion(outputs, smooth_real_labels)
                    real_score = outputs
                    # get the judgement for real image and wrong embed, this is the CLS trick in the original paper
                    outputs, _ = self.D(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs
                    # generate the fake samples
                    noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
                    noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
                    fake_images = self.G(right_embed, noise)
                    # get the judgement for fake image and right embed
                    outputs, _ = self.D(fake_images.detach(), right_embed)
                    fake_loss = criterion(outputs, fake_labels)
                    fake_score = outputs
                    # get the loss of discriminator
                    d_loss = fake_loss + real_loss + wrong_loss
                    d_loss.backward()
                    self.optD.step()

                    # train netG
                    self.optG.clear_grad()
                    # get the judgement for fake image and right embed
                    outputs, activation_fake = self.D(fake_images, right_embed)
                    _, activation_real = self.D(right_images, right_embed)
                    g_loss = criterion(outputs, real_labels)
                    # generate the interpolated images, this is the INT trick in the original paper
                    noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
                    noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
                    inter_images = self.G(inter_embed, noise)
                    outputs, _ = self.D(inter_images, inter_embed)
                    # get the loss of generator
                    g_loss_inter = criterion(outputs, real_labels)
                    g_loss = g_loss + g_loss_inter
                    g_loss.backward()
                    self.optG.step()
                    # print the training logs
                    print('[%d/%d][%d/%d] Loss_D: %.3f  Loss_G: %.3f  D(X): %.3f  D(G(x)):  %.3f'
                          % (epoch, self.num_epochs, iter, len(self.dataloader), d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))
                writer.add_scalar(tag='loss_D_train', value=d_loss.item(), step=epoch)
                writer.add_scalar(tag='loss_G_train', value=g_loss.item(), step=epoch)
                writer.add_scalar(tag='D(x)_train', value=real_score.mean().item(), step=epoch)
                writer.add_scalar(tag='D(G(x)_train', value=fake_score.mean().item(), step=epoch)
                # save the fake images generated by generators
                fake_images = (fake_images + 1) / 2.0
                out_img = fake_images.detach().numpy()[0].transpose((1, 2, 0)) * 255
                out_img = Image.fromarray(out_img.astype(np.uint8))
                out_img.save(rf"image/{epoch}.png")
                # save the parameters of models
                if (epoch+1) % 10 == 0:
                    paddle.save(self.G.state_dict(), 'model/netG_%03d.pdparams' % (epoch+1))
                    paddle.save(self.D.state_dict(), 'model/netD_%03d.pdparams' % (epoch+1))

    def sample(self, model_path):
        # load the parameters into the models
        self.G.load_dict(paddle.load(model_path))
        self.G.train()
        save_dir = 'sample/'
        for s in self.dataloader():
            # get the data in test set
            right_images = s['right_images']
            right_embed = s['right_embed']
            txt = s['txt']
            # generate fake samples
            noise = paddle.randn(shape=[right_images.shape[0], self.noise_dim]).cuda()
            noise = paddle.reshape(noise, shape=[noise.shape[0], 100, 1, 1])
            fake_images = self.G(right_embed, noise)
            fake_images = (fake_images + 1) / 2.0
            # save the fake images
            for image, t in zip(fake_images, txt):
                im = image.detach().numpy().transpose((1, 2, 0)) * 255
                im = Image.fromarray(im.astype(np.uint8))
                im.save(save_dir + '{0}.png'.format(t.replace("/", "")[:100]))
                print(t)
