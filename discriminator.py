import paddle
import paddle.nn as nn


class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        self.conv_w_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.02))
        self.batch_w_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(mean=1.0, std=0.02))
        self.batch_b_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(mean=1.0, std=0.02))

        self.netD = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2D(self.num_channels, self.ndf, 4, 2, 1
                      , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.LeakyReLU(0.2),
            # 64 x 32 x 32
            nn.Conv2D(self.ndf, self.ndf * 2, 4, 2, 1
                      , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ndf * 2, weight_attr=self.batch_w_attr),
            nn.LeakyReLU(0.2),
            # 128 x 16 x 16
            nn.Conv2D(self.ndf * 2, self.ndf * 4, 4, 2, 1
                      , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ndf * 4, weight_attr=self.batch_w_attr),
            nn.LeakyReLU(0.2),
            # 256 x 8 x 8
            nn.Conv2D(self.ndf * 4, self.ndf * 8, 4, 2, 1
                      , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.BatchNorm2D(self.ndf * 8, weight_attr=self.batch_w_attr),
            nn.LeakyReLU(0.2)
            # 512 x 4 x 4
        )

        self.pro_module = nn.Sequential(
            nn.Linear(self.embed_dim, self.projected_embed_dim),
            nn.BatchNorm1D(self.projected_embed_dim, weight_attr=self.batch_w_attr),
            nn.LeakyReLU(0.2)
        )

        self.Get_Logits = nn.Sequential(
            # 512 x 4 x 4
            nn.Conv2D(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0
                      , weight_attr=self.conv_w_attr, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, img, text_emb):
        pro_emb = self.pro_module(text_emb)
        cat_emb = paddle.expand(pro_emb, shape=(4, 4, pro_emb.shape[0], pro_emb.shape[1]))
        cat_emb = paddle.transpose(cat_emb, perm=[2, 3, 0, 1])
        hidden = self.netD(img)
        hidden_cat = paddle.concat([hidden, cat_emb], 1)
        out = self.Get_Logits(hidden_cat)
        out = paddle.reshape(out, shape=[-1, 1])
        return out.squeeze(1), hidden
