import numpy as np

from models.Model_SAE2 import df, liae
from models.Model_SAE2.df import DF
from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samplelib import *
from interact import interact as io

from samplelib.SampleProcessor import ColorTransferMode


# SAE - Styled AutoEncoder


class SAEModel2(ModelBase):
    encoderH5 = 'encoder.h5'
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'
    decoderH5 = 'decoder.h5'
    decodermH5 = 'decoderm.h5'

    decoder_srcH5 = 'decoder_src.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'

    # override
    def onInitializeOptions(self, is_first_run, ask_override):
        yn_str = {True: 'y', False: 'n'}

        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'

        if is_first_run:
            resolution = io.input_int("Resolution ( 64-256 ?:help skip:128) : ", default_resolution,
                                      help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
            resolution = np.clip(resolution, 64, 256)
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
            self.options['resolution'] = resolution

            self.options['face_type'] = io.input_str("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type,
                                                     ['h', 'f'],
                                                     help_message="Half face has better resolution, but covers less area of cheeks.").lower()
            self.options['learn_mask'] = io.input_bool("Learn mask? (y/n, ?:help skip:y) : ", True,
                                                       help_message="Learning mask can help model to recognize face directions. Learn without mask can reduce model size, in this case converter forced to use 'not predicted mask' that is not smooth as predicted. Model with style values can be learned without mask and produce same quality result.")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            self.options['learn_mask'] = self.options.get('learn_mask', True)

        if (is_first_run or ask_override) and 'tensorflow' in self.device_config.backend:
            def_optimizer_mode = self.options.get('optimizer_mode', 1)
            self.options['optimizer_mode'] = io.input_int(
                "Optimizer mode? ( 1,2,3 ?:help skip:%d) : " % (def_optimizer_mode), def_optimizer_mode,
                help_message="1 - no changes. 2 - allows you to train x2 bigger network consuming RAM. 3 - allows you to train x3 bigger network consuming huge amount of RAM and slower, depends on CPU power.")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', 1)

        if is_first_run:
            self.options['archi'] = io.input_str("AE architecture (df, liae ?:help skip:%s) : " % (default_archi),
                                                 default_archi, ['df', 'liae'],
                                                 help_message="'df' keeps faces more natural. 'liae' can fix overly different face shapes.").lower()  # -s version is slower, but has decreased change to collapse.
        else:
            self.options['archi'] = self.options.get('archi', default_archi)

        default_ae_dims = 256 if 'liae' in self.options['archi'] else 512
        default_e_ch_dims = 42
        default_d_ch_dims = default_e_ch_dims // 2
        def_ca_weights = False

        if is_first_run:
            self.options['ae_dims'] = np.clip(
                io.input_int("AutoEncoder dims (32-1024 ?:help skip:%d) : " % (default_ae_dims), default_ae_dims,
                             help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU."),
                32, 1024)
            self.options['e_ch_dims'] = np.clip(
                io.input_int("Encoder dims per channel (21-85 ?:help skip:%d) : " % (default_e_ch_dims),
                             default_e_ch_dims,
                             help_message="More encoder dims help to recognize more facial features, but require more VRAM. You can fine-tune model size to fit your GPU."),
                21, 85)
            default_d_ch_dims = self.options['e_ch_dims'] // 2
            self.options['d_ch_dims'] = np.clip(
                io.input_int("Decoder dims per channel (10-85 ?:help skip:%d) : " % (default_d_ch_dims),
                             default_d_ch_dims,
                             help_message="More decoder dims help to get better details, but require more VRAM. You can fine-tune model size to fit your GPU."),
                10, 85)
            self.options['multiscale_decoder'] = io.input_bool("Use multiscale decoder? (y/n, ?:help skip:n) : ", False,
                                                               help_message="Multiscale decoder helps to get better details.")
            self.options['ca_weights'] = io.input_bool(
                "Use CA weights? (y/n, ?:help skip: %s ) : " % (yn_str[def_ca_weights]), def_ca_weights,
                help_message="Initialize network with 'Convolution Aware' weights. This may help to achieve a higher accuracy model, but consumes a time at first run.")
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['e_ch_dims'] = self.options.get('e_ch_dims', default_e_ch_dims)
            self.options['d_ch_dims'] = self.options.get('d_ch_dims', default_d_ch_dims)
            self.options['multiscale_decoder'] = self.options.get('multiscale_decoder', False)
            self.options['ca_weights'] = self.options.get('ca_weights', def_ca_weights)

        default_face_style_power = 0.0
        default_bg_style_power = 0.0
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool(
                "Use pixel loss? (y/n, ?:help skip: %s ) : " % (yn_str[def_pixel_loss]), def_pixel_loss,
                help_message="Pixel loss may help to enhance fine details and stabilize face color. Use it only if quality does not improve over time. Enabling this option too early increases the chance of model collapse.")

            default_face_style_power = default_face_style_power if is_first_run else self.options.get(
                'face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip(
                io.input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_face_style_power),
                                default_face_style_power,
                                help_message="Learn to transfer face style details such as light and color conditions. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.1 value and check history changes. Enabling this option increases the chance of model collapse."),
                0.0, 100.0)

            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power',
                                                                                                  default_bg_style_power)
            self.options['bg_style_power'] = np.clip(
                io.input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_bg_style_power),
                                default_bg_style_power,
                                help_message="Learn to transfer image around face. This can make face more like dst. Enabling this option increases the chance of model collapse."),
                0.0, 100.0)

            default_apply_random_ct = ColorTransferMode.NONE if is_first_run else self.options.get('apply_random_ct',
                                                                                                   ColorTransferMode.NONE)
            self.options['apply_random_ct'] = np.clip(io.input_int(
                "Apply random color transfer to src faceset? (0) None, (1) LCT, (2) RCT, (3) RCT-c, (4) RCT-p, "
                "(5) RCT-pc, (6) mRTC, (7) mRTC-c, (8) mRTC-p, (9) mRTC-pc ?:help skip:%s) : " % default_apply_random_ct,
                default_apply_random_ct,
                help_message="Increase variativity of src samples by apply LCT color transfer from random dst "
                             "samples. It is like 'face_style' learning, but more precise color transfer and without "
                             "risk of model collapse, also it does not require additional GPU resources, "
                             "but the training time may be longer, due to the src faceset is becoming more diverse."),
                ColorTransferMode.NONE, ColorTransferMode.MASKED_RCT_PAPER_CLIP)

            if nnlib.device.backend != 'plaidML':  # todo https://github.com/plaidml/plaidml/issues/301
                default_clipgrad = False if is_first_run else self.options.get('clipgrad', False)
                self.options['clipgrad'] = io.input_bool(
                    "Enable gradient clipping? (y/n, ?:help skip:%s) : " % (yn_str[default_clipgrad]), default_clipgrad,
                    help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")
            else:
                self.options['clipgrad'] = False

        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            self.options['apply_random_ct'] = self.options.get('apply_random_ct', ColorTransferMode.NONE)
            self.options['clipgrad'] = self.options.get('clipgrad', False)

        if is_first_run:
            self.options['pretrain'] = io.input_bool("Pretrain the model? (y/n, ?:help skip:n) : ", False,
                                                     help_message="Pretrain the model with large amount of various faces. This technique may help to train the fake with overly different face shapes and light conditions of src/dst data. Face will be look more like a morphed. To reduce the morph effect, some model files will be initialized but not be updated after pretrain: LIAE: inter_AB.h5 DF: encoder.h5. The longer you pretrain the model the more morphed face will look. After that, save and run the training again.")
        else:
            self.options['pretrain'] = False

    # override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements({1.5: 4})

        global resolution
        resolution = self.options['resolution']
        ae_dims = self.options['ae_dims']
        e_ch_dims = self.options['e_ch_dims']
        d_ch_dims = self.options['d_ch_dims']
        self.pretrain = self.options['pretrain'] = self.options.get('pretrain', False)
        if not self.pretrain:
            self.options.pop('pretrain')

        d_residual_blocks = True
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        global ms_count
        self.ms_count = ms_count = 3 if (self.options['multiscale_decoder']) else 1

        global apply_random_ct
        apply_random_ct = self.options.get('apply_random_ct', ColorTransferMode.NONE)
        masked_training = True

        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)

        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)

        target_src_ar = [Input((bgr_shape[0] // (2 ** i),) * 2 + (bgr_shape[-1],)) for i in range(ms_count - 1, -1, -1)]
        target_srcm_ar = [Input((mask_shape[0] // (2 ** i),) * 2 + (mask_shape[-1],)) for i in
                          range(ms_count - 1, -1, -1)]
        target_dst_ar = [Input((bgr_shape[0] // (2 ** i),) * 2 + (bgr_shape[-1],)) for i in range(ms_count - 1, -1, -1)]
        target_dstm_ar = [Input((mask_shape[0] // (2 ** i),) * 2 + (mask_shape[-1],)) for i in
                          range(ms_count - 1, -1, -1)]

        common_flow_kwargs = {'padding': 'zero',
                              'norm': '',
                              'act': ''}
        models_list = []
        weights_to_load = []
        if 'liae' in self.options['archi']:
            self.encoder = modelify(SAEModel2.LIAEEncFlow(resolution, ch_dims=e_ch_dims, **common_flow_kwargs))(
                Input(bgr_shape))

            enc_output_Inputs = [Input(K.int_shape(x)[1:]) for x in self.encoder.outputs]

            self.inter_B = modelify(SAEModel2.LIAEInterFlow(resolution, ae_dims=ae_dims, **common_flow_kwargs))(
                enc_output_Inputs)
            self.inter_AB = modelify(SAEModel2.LIAEInterFlow(resolution, ae_dims=ae_dims, **common_flow_kwargs))(
                enc_output_Inputs)

            inter_output_Inputs = [Input(np.array(K.int_shape(x)[1:]) * (1, 1, 2)) for x in self.inter_B.outputs]

            self.decoder = modelify(
                SAEModel2.LIAEDecFlow(bgr_shape[2], ch_dims=d_ch_dims, multiscale_count=self.ms_count,
                                      add_residual_blocks=d_residual_blocks, **common_flow_kwargs))(inter_output_Inputs)
            models_list += [self.encoder, self.inter_B, self.inter_AB, self.decoder]

            if self.options['learn_mask']:
                self.decoderm = modelify(SAEModel2.LIAEDecFlow(mask_shape[2], ch_dims=d_ch_dims, **common_flow_kwargs))(
                    inter_output_Inputs)
                models_list += [self.decoderm]

            if not self.is_first_run():
                weights_to_load += [[self.encoder, 'encoder.h5'],
                                    [self.inter_B, 'inter_B.h5'],
                                    [self.inter_AB, 'inter_AB.h5'],
                                    [self.decoder, 'decoder.h5'],
                                    ]
                if self.options['learn_mask']:
                    weights_to_load += [[self.decoderm, 'decoderm.h5']]

            warped_src_code = self.encoder(warped_src)
            warped_src_inter_AB_code = self.inter_AB(warped_src_code)
            warped_src_inter_code = Concatenate()([warped_src_inter_AB_code, warped_src_inter_AB_code])

            warped_dst_code = self.encoder(warped_dst)
            warped_dst_inter_B_code = self.inter_B(warped_dst_code)
            warped_dst_inter_AB_code = self.inter_AB(warped_dst_code)
            warped_dst_inter_code = Concatenate()([warped_dst_inter_B_code, warped_dst_inter_AB_code])

            warped_src_dst_inter_code = Concatenate()([warped_dst_inter_AB_code, warped_dst_inter_AB_code])

            pred_src_src = self.decoder(warped_src_inter_code)
            pred_dst_dst = self.decoder(warped_dst_inter_code)
            pred_src_dst = self.decoder(warped_src_dst_inter_code)

            if self.options['learn_mask']:
                pred_src_srcm = self.decoderm(warped_src_inter_code)
                pred_dst_dstm = self.decoderm(warped_dst_inter_code)
                pred_src_dstm = self.decoderm(warped_src_dst_inter_code)

        elif 'df' in self.options['archi']:
            self.DF = DF(bgr_shape, mask_shape, resolution, ae_dims, e_ch_dims, d_ch_dims, self.ms_count, d_residual_blocks, **common_flow_kwargs)
            self.encoder = self.DF.encoder()
            print(self.encoder.summary())
            self.decoder_src = self.DF.decoder(**common_flow_kwargs)
            self.decoder_dst = self.DF.decoder(**common_flow_kwargs)
            models_list += [self.encoder, self.decoder_src, self.decoder_dst]

            if self.options['learn_mask']:
                self.decoder_srcm = self.DF.decoder_mask(**common_flow_kwargs)
                self.decoder_dstm = self.DF.decoder_mask(**common_flow_kwargs)
                models_list += [self.decoder_srcm, self.decoder_dstm]

            if not self.is_first_run():
                weights_to_load += [[self.encoder, 'encoder.h5'],
                                    [self.decoder_src, 'decoder_src.h5'],
                                    [self.decoder_dst, 'decoder_dst.h5']
                                    ]
                if self.options['learn_mask']:
                    weights_to_load += [[self.decoder_srcm, 'decoder_srcm.h5'],
                                        [self.decoder_dstm, 'decoder_dstm.h5'],
                                        ]

            warped_src_code = self.encoder(warped_src)
            warped_dst_code = self.encoder(warped_dst)
            pred_src_src = self.decoder_src(warped_src_code)
            pred_dst_dst = self.decoder_dst(warped_dst_code)
            pred_src_dst = self.decoder_src(warped_dst_code)

            if self.options['learn_mask']:
                pred_src_srcm = self.decoder_srcm(warped_src_code)
                pred_dst_dstm = self.decoder_dstm(warped_dst_code)
                pred_src_dstm = self.decoder_srcm(warped_dst_code)

        if self.is_first_run():
            if self.options.get('ca_weights', False):
                conv_weights_list = []
                for model in models_list:
                    for layer in model.layers:
                        if type(layer) == keras.layers.Conv2D:
                            conv_weights_list += [layer.weights[0]]  # Conv2D kernel_weights
                CAInitializerMP(conv_weights_list)
        else:
            self.load_weights_safe(weights_to_load)

        pred_src_src, pred_dst_dst, pred_src_dst, = [[x] if type(x) != list else x for x in
                                                     [pred_src_src, pred_dst_dst, pred_src_dst, ]]

        if self.options['learn_mask']:
            pred_src_srcm, pred_dst_dstm, pred_src_dstm = [[x] if type(x) != list else x for x in
                                                           [pred_src_srcm, pred_dst_dstm, pred_src_dstm]]

        target_srcm_blurred_ar = [gaussian_blur(max(1, K.int_shape(x)[1] // 32))(x) for x in target_srcm_ar]
        target_srcm_sigm_ar = target_srcm_blurred_ar  # [ x / 2.0 + 0.5 for x in target_srcm_blurred_ar]
        target_srcm_anti_sigm_ar = [1.0 - x for x in target_srcm_sigm_ar]

        target_dstm_blurred_ar = [gaussian_blur(max(1, K.int_shape(x)[1] // 32))(x) for x in target_dstm_ar]
        target_dstm_sigm_ar = target_dstm_blurred_ar  # [ x / 2.0 + 0.5 for x in target_dstm_blurred_ar]
        target_dstm_anti_sigm_ar = [1.0 - x for x in target_dstm_sigm_ar]

        target_src_sigm_ar = target_src_ar  # [ x + 1 for x in target_src_ar]
        target_dst_sigm_ar = target_dst_ar  # [ x + 1 for x in target_dst_ar]

        pred_src_src_sigm_ar = pred_src_src  # [ x + 1 for x in pred_src_src]
        pred_dst_dst_sigm_ar = pred_dst_dst  # [ x + 1 for x in pred_dst_dst]
        pred_src_dst_sigm_ar = pred_src_dst  # [ x + 1 for x in pred_src_dst]

        target_src_masked_ar = [target_src_sigm_ar[i] * target_srcm_sigm_ar[i] for i in range(len(target_src_sigm_ar))]
        target_dst_masked_ar = [target_dst_sigm_ar[i] * target_dstm_sigm_ar[i] for i in range(len(target_dst_sigm_ar))]
        target_dst_anti_masked_ar = [target_dst_sigm_ar[i] * target_dstm_anti_sigm_ar[i] for i in
                                     range(len(target_dst_sigm_ar))]

        pred_src_src_masked_ar = [pred_src_src_sigm_ar[i] * target_srcm_sigm_ar[i] for i in
                                  range(len(pred_src_src_sigm_ar))]
        pred_dst_dst_masked_ar = [pred_dst_dst_sigm_ar[i] * target_dstm_sigm_ar[i] for i in
                                  range(len(pred_dst_dst_sigm_ar))]

        target_src_masked_ar_opt = target_src_masked_ar if masked_training else target_src_sigm_ar
        target_dst_masked_ar_opt = target_dst_masked_ar if masked_training else target_dst_sigm_ar

        pred_src_src_masked_ar_opt = pred_src_src_masked_ar if masked_training else pred_src_src_sigm_ar
        pred_dst_dst_masked_ar_opt = pred_dst_dst_masked_ar if masked_training else pred_dst_dst_sigm_ar

        psd_target_dst_masked_ar = [pred_src_dst_sigm_ar[i] * target_dstm_sigm_ar[i] for i in
                                    range(len(pred_src_dst_sigm_ar))]
        psd_target_dst_anti_masked_ar = [pred_src_dst_sigm_ar[i] * target_dstm_anti_sigm_ar[i] for i in
                                         range(len(pred_src_dst_sigm_ar))]

        if self.is_training_mode:
            self.src_dst_opt = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999,
                                    clipnorm=1.0 if self.options['clipgrad'] else 0.0,
                                    tf_cpu_mode=self.options['optimizer_mode'] - 1)
            self.src_dst_mask_opt = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999,
                                         clipnorm=1.0 if self.options['clipgrad'] else 0.0,
                                         tf_cpu_mode=self.options['optimizer_mode'] - 1)

            if 'liae' in self.options['archi']:
                src_dst_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
                if self.options['learn_mask']:
                    src_dst_mask_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
            else:
                src_dst_loss_train_weights = self.encoder.trainable_weights + self.decoder_src.trainable_weights + self.decoder_dst.trainable_weights
                if self.options['learn_mask']:
                    src_dst_mask_loss_train_weights = self.encoder.trainable_weights + self.decoder_srcm.trainable_weights + self.decoder_dstm.trainable_weights

            if not self.options['pixel_loss']:
                src_loss_batch = sum([10 * dssim(kernel_size=int(resolution / 11.6), max_value=1.0)(
                    target_src_masked_ar_opt[i], pred_src_src_masked_ar_opt[i]) for i in
                                      range(len(target_src_masked_ar_opt))])
            else:
                src_loss_batch = sum(
                    [K.mean(50 * K.square(target_src_masked_ar_opt[i] - pred_src_src_masked_ar_opt[i]), axis=[1, 2, 3])
                     for i in range(len(target_src_masked_ar_opt))])

            src_loss = K.mean(src_loss_batch)

            face_style_power = self.options['face_style_power'] / 100.0

            if face_style_power != 0:
                src_loss += style_loss(gaussian_blur_radius=resolution // 16, loss_weight=face_style_power, wnd_size=0)(
                    psd_target_dst_masked_ar[-1], target_dst_masked_ar[-1])

            bg_style_power = self.options['bg_style_power'] / 100.0
            if bg_style_power != 0:
                if not self.options['pixel_loss']:
                    bg_loss = K.mean((10 * bg_style_power) * dssim(kernel_size=int(resolution / 11.6), max_value=1.0)(
                        psd_target_dst_anti_masked_ar[-1], target_dst_anti_masked_ar[-1]))
                else:
                    bg_loss = K.mean((50 * bg_style_power) * K.square(
                        psd_target_dst_anti_masked_ar[-1] - target_dst_anti_masked_ar[-1]))
                src_loss += bg_loss

            if not self.options['pixel_loss']:
                dst_loss_batch = sum([10 * dssim(kernel_size=int(resolution / 11.6), max_value=1.0)(
                    target_dst_masked_ar_opt[i], pred_dst_dst_masked_ar_opt[i]) for i in
                                      range(len(target_dst_masked_ar_opt))])
            else:
                dst_loss_batch = sum(
                    [K.mean(50 * K.square(target_dst_masked_ar_opt[i] - pred_dst_dst_masked_ar_opt[i]), axis=[1, 2, 3])
                     for i in range(len(target_dst_masked_ar_opt))])

            dst_loss = K.mean(dst_loss_batch)

            feed = [warped_src, warped_dst]
            feed += target_src_ar[::-1]
            feed += target_srcm_ar[::-1]
            feed += target_dst_ar[::-1]
            feed += target_dstm_ar[::-1]

            self.src_dst_train = K.function(feed, [src_loss, dst_loss],
                                            self.src_dst_opt.get_updates(src_loss + dst_loss,
                                                                         src_dst_loss_train_weights))

            if self.options['learn_mask']:
                src_mask_loss = sum(
                    [K.mean(K.square(target_srcm_ar[-1] - pred_src_srcm[-1])) for i in range(len(target_srcm_ar))])
                dst_mask_loss = sum(
                    [K.mean(K.square(target_dstm_ar[-1] - pred_dst_dstm[-1])) for i in range(len(target_dstm_ar))])

                feed = [warped_src, warped_dst]
                feed += target_srcm_ar[::-1]
                feed += target_dstm_ar[::-1]

                self.src_dst_mask_train = K.function(feed, [src_mask_loss, dst_mask_loss],
                                                     self.src_dst_mask_opt.get_updates(src_mask_loss + dst_mask_loss,
                                                                                       src_dst_mask_loss_train_weights))

            if self.options['learn_mask']:
                self.AE_view = K.function([warped_src, warped_dst],
                                          [pred_src_src[-1], pred_dst_dst[-1], pred_dst_dstm[-1], pred_src_dst[-1],
                                           pred_src_dstm[-1]])
            else:
                self.AE_view = K.function([warped_src, warped_dst],
                                          [pred_src_src[-1], pred_dst_dst[-1], pred_src_dst[-1]])


        else:
            if self.options['learn_mask']:
                self.AE_convert = K.function([warped_dst], [pred_src_dst[-1], pred_dst_dstm[-1], pred_src_dstm[-1]])
            else:
                self.AE_convert = K.function([warped_dst], [pred_src_dst[-1]])

        if self.is_training_mode:
            self.src_sample_losses = []
            self.dst_sample_losses = []

            global t
            t = SampleProcessor.Types
            global face_type
            face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF

            global t_mode_bgr
            t_mode_bgr = t.MODE_BGR if not self.pretrain else t.MODE_BGR_SHUFFLE

            global training_data_src_path
            training_data_src_path = self.training_data_src_path
            global training_data_dst_path
            training_data_dst_path = self.training_data_dst_path

            global sort_by_yaw
            sort_by_yaw = self.sort_by_yaw

            if self.pretrain and self.pretraining_data_path is not None:
                training_data_src_path = self.pretraining_data_path
                training_data_dst_path = self.pretraining_data_path
                sort_by_yaw = False

            self.set_training_data_generators([
                SampleGeneratorFace(training_data_src_path,
                                    sort_by_yaw_target_samples_path=training_data_dst_path if sort_by_yaw else None,
                                    random_ct_samples_path=training_data_dst_path if apply_random_ct != ColorTransferMode.NONE else None,
                                    debug=self.is_debug(), batch_size=self.batch_size,
                                    sample_process_options=SampleProcessor.Options(random_flip=self.random_flip,
                                                                                   scale_range=np.array([-0.05,
                                                                                                         0.05]) + self.src_scale_mod / 100.0),
                                    output_sample_types=[{'types': (
                                        t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr),
                                        'resolution': resolution, 'apply_ct': apply_random_ct}] + \
                                                        [{'types': (t.IMG_TRANSFORMED, face_type, t_mode_bgr),
                                                          'resolution': resolution // (2 ** i),
                                                          'apply_ct': apply_random_ct} for i in range(ms_count)] + \
                                                        [{'types': (t.IMG_TRANSFORMED, face_type, t.MODE_M),
                                                          'resolution': resolution // (2 ** i)} for i in
                                                         range(ms_count)]
                                    ),

                SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                                    sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ),
                                    output_sample_types=[{'types': (
                                        t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr),
                                        'resolution': resolution}] + \
                                                        [{'types': (t.IMG_TRANSFORMED, face_type, t_mode_bgr),
                                                          'resolution': resolution // (2 ** i)} for i in
                                                         range(ms_count)] + \
                                                        [{'types': (t.IMG_TRANSFORMED, face_type, t.MODE_M),
                                                          'resolution': resolution // (2 ** i)} for i in
                                                         range(ms_count)])
            ])

    # override
    def get_model_filename_list(self):
        ar = []
        if 'liae' in self.options['archi']:
            ar += [[self.encoder, 'encoder.h5'],
                   [self.inter_B, 'inter_B.h5'],
                   [self.decoder, 'decoder.h5']
                   ]

            if not self.pretrain or self.iter == 0:
                ar += [[self.inter_AB, 'inter_AB.h5'],
                       ]

            if self.options['learn_mask']:
                ar += [[self.decoderm, 'decoderm.h5']]

        elif 'df' in self.options['archi']:
            if not self.pretrain or self.iter == 0:
                ar += [[self.encoder, 'encoder.h5'],
                       ]

            ar += [[self.decoder_src, 'decoder_src.h5'],
                   [self.decoder_dst, 'decoder_dst.h5']
                   ]

            if self.options['learn_mask']:
                ar += [[self.decoder_srcm, 'decoder_srcm.h5'],
                       [self.decoder_dstm, 'decoder_dstm.h5']]
        return ar

    # override
    def onSave(self):
        self.save_weights_safe(self.get_model_filename_list())

    # override
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.set_training_data_generators(None)
        self.set_training_data_generators([
            SampleGeneratorFace(training_data_src_path,
                                sort_by_yaw_target_samples_path=training_data_dst_path if sort_by_yaw else None,
                                random_ct_samples_path=training_data_dst_path if apply_random_ct != ColorTransferMode.NONE else None,
                                debug=self.is_debug(), batch_size=self.batch_size,
                                sample_process_options=SampleProcessor.Options(random_flip=self.random_flip,
                                                                               scale_range=np.array([-0.05,
                                                                                                     0.05]) + self.src_scale_mod / 100.0),
                                output_sample_types=[{'types': (
                                    t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr),
                                    'resolution': resolution, 'apply_ct': apply_random_ct}] + \
                                                    [{'types': (t.IMG_TRANSFORMED, face_type, t_mode_bgr),
                                                      'resolution': resolution // (2 ** i),
                                                      'apply_ct': apply_random_ct} for i in range(ms_count)] + \
                                                    [{'types': (t.IMG_TRANSFORMED, face_type, t.MODE_M),
                                                      'resolution': resolution // (2 ** i)} for i in
                                                     range(ms_count)]
                                ),

            SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                                sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ),
                                output_sample_types=[{'types': (
                                    t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr),
                                    'resolution': resolution}] + \
                                                    [{'types': (t.IMG_TRANSFORMED, face_type, t_mode_bgr),
                                                      'resolution': resolution // (2 ** i)} for i in
                                                     range(ms_count)] + \
                                                    [{'types': (t.IMG_TRANSFORMED, face_type, t.MODE_M),
                                                      'resolution': resolution // (2 ** i)} for i in
                                                     range(ms_count)])
        ])

    # override
    def onTrainOneIter(self, generators_samples, generators_list):
        src_samples = generators_samples[0]
        dst_samples = generators_samples[1]

        feed = [src_samples[0], dst_samples[0]] + \
               src_samples[1:1 + self.ms_count * 2] + \
               dst_samples[1:1 + self.ms_count * 2]

        src_loss, dst_loss, = self.src_dst_train(feed)

        if self.options['learn_mask']:
            feed = [src_samples[0], dst_samples[0]] + \
                   src_samples[1 + self.ms_count:1 + self.ms_count * 2] + \
                   dst_samples[1 + self.ms_count:1 + self.ms_count * 2]
            src_mask_loss, dst_mask_loss, = self.src_dst_mask_train(feed)

        return (('src_loss', src_loss), ('dst_loss', dst_loss))

    # override
    def onGetPreview(self, sample):
        test_S = sample[0][1][0:4]  # first 4 samples
        test_S_m = sample[0][1 + self.ms_count][0:4]  # first 4 samples
        test_D = sample[1][1][0:4]
        test_D_m = sample[1][1 + self.ms_count][0:4]

        if self.options['learn_mask']:
            S, D, SS, DD, DDM, SD, SDM = [np.clip(x, 0.0, 1.0) for x in
                                          ([test_S, test_D] + self.AE_view([test_S, test_D]))]
            DDM, SDM, = [np.repeat(x, (3,), -1) for x in [DDM, SDM]]
        else:
            S, D, SS, DD, SD, = [np.clip(x, 0.0, 1.0) for x in ([test_S, test_D] + self.AE_view([test_S, test_D]))]

        result = []
        st = []
        for i in range(0, len(test_S)):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append(np.concatenate(ar, axis=1))

        result += [('SAE', np.concatenate(st, axis=0)), ]

        if self.options['learn_mask']:
            st_m = []
            for i in range(0, len(test_S)):
                ar = S[i] * test_S_m[i], SS[i], D[i] * test_D_m[i], DD[i] * DDM[i], SD[i] * (DDM[i] * SDM[i])
                st_m.append(np.concatenate(ar, axis=1))

            result += [('SAE masked', np.concatenate(st_m, axis=0)), ]

        return result

    def predictor_func(self, face):
        if self.options['learn_mask']:
            bgr, mask_dst_dstm, mask_src_dstm = self.AE_convert([face[np.newaxis, ...]])
            mask = mask_dst_dstm[0] * mask_src_dstm[0]
            return bgr[0], mask[..., 0]
        else:
            bgr, = self.AE_convert([face[np.newaxis, ...]])
            return bgr[0]

    # override
    def get_converter(self):
        base_erode_mask_modifier = 30 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 0 if self.options['face_type'] == 'f' else 100

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 100 if (self.options['face_style_power'] or self.options['bg_style_power']) and \
                                            self.options['face_type'] == 'f' else 0

        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        from converters import ConverterMasked

        return ConverterMasked(self.predictor_func,
                               predictor_input_size=self.options['resolution'],
                               predictor_masked=self.options['learn_mask'],
                               face_type=face_type,
                               default_mode=1 if self.options['apply_random_ct'] or self.options['face_style_power'] or
                                                 self.options['bg_style_power'] else 4,
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               default_erode_mask_modifier=default_erode_mask_modifier,
                               default_blur_mask_modifier=default_blur_mask_modifier,
                               clip_hborder_mask_per=0.0625 if (self.options['face_type'] == 'f') else 0)
