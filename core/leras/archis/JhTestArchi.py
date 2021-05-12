from core.leras import nn
tf = nn.tf

class JhTestArchi(nn.ArchiBase):
    """
    resolution

    mod     None - default
            'quick'
    """
    def __init__(self, resolution, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''

        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size) )
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def on_build(self, in_ch, out_ch, kernel_size=3):
                    self.n = in_ch // out_ch
                    self.conv1 = nn.Conv2D(in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME')

                def forward(self, x):
                    x1 = nn.bilinear_additive_upsampling(x, n=self.n)
                    x2 = self.conv1(x)
                    x2 = tf.nn.leaky_relu(x2, 0.1)
                    x2 = nn.depth_to_space(x2, 2)
                    return x1 + x2

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = tf.nn.leaky_relu(inp + x, 0.2)
                    return x

            class ResBlock(nn.ModelBase):
                def on_build(self, filters, kernel_size=3, stride=1, conv_shortcut=True):
                    # self.bn_axis = 3 if nn.data_format == "NHWC" else 1

                    if conv_shortcut:
                        self.conv_shortcut = nn.Conv2D(filters, 4 * filters, kernel_size=1, strides=stride, padding='SAME')
                    else:
                        self.conv_shortcut = None

                    self.conv1 = nn.Conv2D(filters if conv_shortcut else 4 * filters, filters, kernel_size=1, strides=stride, padding='SAME')
                    self.conv2 = nn.Conv2D(filters, filters, kernel_size=kernel_size, padding='SAME')
                    self.conv3 = nn.Conv2D(filters, 4 * filters, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    if self.conv_shortcut is not None:
                        shortcut = self.conv_shortcut(inp)
                    else:
                        shortcut = inp

                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv3(x)
                    x = tf.nn.leaky_relu(shortcut + x, 0.2)
                    return x

            class ResStack(nn.ModelBase):
                def on_build(self, filters, blocks, stride=2):
                    self.blocks = [ResBlock(filters, stride=stride)]
                    for i in range(2, blocks + 1):
                        self.blocks.append(ResBlock(filters, conv_shortcut=False))

                def forward(self, inp):
                    x = inp
                    for block in self.blocks:
                        x = block(x)
                    return x


            class Encoder(nn.ModelBase):
                def __init__(self, in_ch, e_ch, **kwargs ):
                    self.in_ch = in_ch
                    self.e_ch = e_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    self.conv1 = nn.Conv2D(self.in_ch, self.e_ch, kernel_size=7, strides=2, padding='SAME')
                    self.stack1 = ResStack(self.e_ch, 3, stride=1)
                    self.stack2 = ResStack(2 * self.e_ch, 4)
                    self.stack3 = ResStack(4 * self.e_ch, 6)

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = nn.max_pool(x, kernel_size=3, strides=2)
                    x = self.stack1(x)
                    x = self.stack2(x)
                    x = self.stack3(x)

                    return x

                def get_out_res(self, res):
                    return res // (2**4)

                def get_out_ch(self):
                    return self.e_ch * 16

            lowest_dense_res = resolution // (32 if 'd' in opts else 16)

            class Inter(nn.ModelBase):
                def __init__(self, e_ch, ae_ch, ae_out_ch, **kwargs):
                    self.e_ch, self.ae_ch, self.ae_out_ch = e_ch, ae_ch, ae_out_ch
                    super().__init__(**kwargs)

                def on_build(self):
                    e_ch, ae_ch, ae_out_ch = self.e_ch, self.ae_ch, self.ae_out_ch
                    if 'u' in opts:
                        self.dense_norm = nn.DenseNorm()

                    self.stack4 = ResStack(8 * self.e_ch, 3)

                    self.dense1 = nn.Dense( 32 * self.e_ch, ae_ch )
                    self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                    self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = inp
                    x = self.stack4(x)
                    x = nn.global_avg_pool(x)
                    if 'u' in opts:
                        x = self.dense_norm(x)
                    x = self.dense1(x)
                    x = self.dense2(x)
                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                    x = self.upscale1(x)
                    return x

                def get_out_res(self):
                    return lowest_dense_res * 2

                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):
                def on_build(self, in_ch, d_ch, d_mask_ch ):
                    self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                    self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                    self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                    self.upscale3 = Upscale(d_ch*4, d_ch*1, kernel_size=3)

                    self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                    self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                    self.res2 = ResidualBlock(d_ch*2, kernel_size=3)
                    self.res3 = ResidualBlock(d_ch*1, kernel_size=3)

                    self.out_conv  = nn.Conv2D( d_ch*1, 3, kernel_size=1, padding='SAME')

                    self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                    self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                    self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                    self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)

                    self.out_convm = nn.Conv2D(d_mask_ch*1, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    x = self.upscale0(inp)
                    x = self.res0(x)
                    x = self.upscale1(x)
                    x = self.res1(x)
                    x = self.upscale2(x)
                    x = self.res2(x)
                    x = self.upscale3(x)
                    x = self.res3(x)

                    x = tf.nn.sigmoid(self.out_conv(x))

                    m = self.upscalem0(inp)
                    m = self.upscalem1(m)
                    m = self.upscalem2(m)
                    m = self.upscalem3(m)
                    m = tf.nn.sigmoid(self.out_convm(m))

                    return x, m

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

nn.JhTestArchi = JhTestArchi
