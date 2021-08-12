import paddle
import paddle.nn as nn


# define the generator
class Generator(nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.image_size = 64
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64
        self.conv_w_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.02))
        self.batch_w_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(mean=1.0, std=0.02))
        # reduce the dimension of sentence embeddings
        self.pro_module = nn.Sequential(
            nn.Linear(self.embed_dim, self.projected_embed_dim),
            nn.BatchNorm1D(num_features=self.projected_embed_dim, weight_attr=self.batch_w_attr),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # the generator networks
        self.netG = nn.Sequential(
            nn.Conv2DTranspose(in_channels=self.latent_dim, out_channels=self.ngf * 8, kernel_size=4, stride=1,
                               padding=0
                               , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ngf * 8, weight_attr=self.batch_w_attr),
            nn.ReLU(),
            # 512 x 4 x 4
            nn.Conv2DTranspose(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1
                               , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ngf * 4, weight_attr=self.batch_w_attr),
            nn.ReLU(),
            # 256 x 8 x 8
            nn.Conv2DTranspose(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1
                               , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ngf * 2, weight_attr=self.batch_w_attr),
            nn.ReLU(),
            # 128 x 16 x 16
            nn.Conv2DTranspose(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1
                               , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ngf, weight_attr=self.batch_w_attr),
            nn.ReLU(),
            # 64 x 32 x 32
            nn.Conv2DTranspose(in_channels=self.ngf, out_channels=self.num_channels, kernel_size=4, stride=2, padding=1
                               , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.Tanh()
            # 3 x 64 x 64
        )

    def forward(self, text_emb, z):
        # inputs: sentence embeddings and latent vector
        # output: fake samples synthesized by the generator
        pro_emb = self.pro_module(text_emb).unsqueeze(2).unsqueeze(3)
        latent_code = paddle.concat([pro_emb, z], 1)
        out = self.netG(latent_code)
        return out
