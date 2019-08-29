import collections
from enum import IntEnum

import cv2
import numpy as np

import imagelib
from facelib import FaceType, LandmarksProcessor

"""
output_sample_types = [
                        {} opts,
                        ...
                      ]

opts:
    'types' : (S,S,...,S)
        where S:
            'IMG_SOURCE'
            'IMG_WARPED'
            'IMG_WARPED_TRANSFORMED''
            'IMG_TRANSFORMED'
            'IMG_LANDMARKS_ARRAY' #currently unused
            'IMG_PITCH_YAW_ROLL'

            'FACE_TYPE_HALF'
            'FACE_TYPE_FULL'
            'FACE_TYPE_HEAD'    #currently unused
            'FACE_TYPE_AVATAR'  #currently unused

            'MODE_BGR'         #BGR
            'MODE_G'           #Grayscale
            'MODE_GGG'         #3xGrayscale
            'MODE_M'           #mask only
            'MODE_BGR_SHUFFLE' #BGR shuffle

    'resolution' : N
    'motion_blur' : (chance_int, range) - chance 0..100 to apply to face (not mask), and range [1..3] where 3 is highest power of motion blur
    'apply_ct' : bool
    'normalize_tanh' : bool

"""


class ColorTransferMode(IntEnum):
    NONE = 0
    LCT = 1
    RCT = 2
    RCT_CLIP = 3
    RCT_PAPER = 4
    RCT_PAPER_CLIP = 5
    MASKED_RCT = 6
    MASKED_RCT_CLIP = 7
    MASKED_RCT_PAPER = 8
    MASKED_RCT_PAPER_CLIP = 9


class SampleProcessor(object):
    class Types(IntEnum):
        NONE = 0

        IMG_TYPE_BEGIN = 1
        IMG_SOURCE = 1
        IMG_WARPED = 2
        IMG_WARPED_TRANSFORMED = 3
        IMG_TRANSFORMED = 4
        IMG_LANDMARKS_ARRAY = 5  # currently unused
        IMG_PITCH_YAW_ROLL = 6
        IMG_PITCH_YAW_ROLL_SIGMOID = 7
        IMG_TYPE_END = 10

        FACE_TYPE_BEGIN = 10
        FACE_TYPE_HALF             = 10
        FACE_TYPE_FULL             = 11
        FACE_TYPE_HEAD             = 12  #currently unused
        FACE_TYPE_AVATAR           = 13  #currently unused
        FACE_TYPE_FULL_NO_ALIGN    = 14
        FACE_TYPE_END = 20

        MODE_BEGIN = 40
        MODE_BGR = 40  # BGR
        MODE_G = 41  # Grayscale
        MODE_GGG = 42  # 3xGrayscale
        MODE_M = 43  # mask only
        MODE_BGR_SHUFFLE = 44  # BGR shuffle
        MODE_END = 50

    class Options(object):

        def __init__(self, random_flip=True, rotation_range=[-10, 10], scale_range=[-0.05, 0.05],
                     tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], extend_forehead=False):
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range
            self.extend_forehead = extend_forehead


    @staticmethod
    def process(sample, sample_process_options: Options, output_sample_types, debug, ct_sample=None):
        SPTF = SampleProcessor.Types

        sample_bgr = sample.load_bgr()
        ct_sample_bgr = None
        ct_sample_mask = None
        h, w, c = sample_bgr.shape

        is_face_sample = sample.landmarks is not None

        if debug and is_face_sample:
            LandmarksProcessor.draw_landmarks(sample_bgr, sample.landmarks, (0, 1, 0))

        params = imagelib.gen_warp_params(sample_bgr, sample_process_options.random_flip,
                                          rotation_range=sample_process_options.rotation_range,
                                          scale_range=sample_process_options.scale_range,
                                          tx_range=sample_process_options.tx_range,
                                          ty_range=sample_process_options.ty_range)

        cached_images = collections.defaultdict(dict)

        sample_rnd_seed = np.random.randint(0x80000000)

        SPTF_FACETYPE_TO_FACETYPE =  {  SPTF.FACE_TYPE_HALF : FaceType.HALF,
                                        SPTF.FACE_TYPE_FULL : FaceType.FULL,
                                        SPTF.FACE_TYPE_HEAD : FaceType.HEAD,
                                        SPTF.FACE_TYPE_FULL_NO_ALIGN : FaceType.FULL_NO_ALIGN }

        outputs = []
        for opts in output_sample_types:

            resolution = opts.get('resolution', 0)
            types = opts.get('types', [])

            random_sub_res = opts.get('random_sub_res', 0)
            normalize_std_dev = opts.get('normalize_std_dev', False)
            normalize_vgg = opts.get('normalize_vgg', False)
            motion_blur = opts.get('motion_blur', None)
            apply_ct = opts.get('apply_ct', ColorTransferMode.NONE)
            normalize_tanh = opts.get('normalize_tanh', False)

            img_type = SPTF.NONE
            target_face_type = SPTF.NONE
            face_mask_type = SPTF.NONE
            mode_type = SPTF.NONE
            for t in types:
                if t >= SPTF.IMG_TYPE_BEGIN and t < SPTF.IMG_TYPE_END:
                    img_type = t
                elif t >= SPTF.FACE_TYPE_BEGIN and t < SPTF.FACE_TYPE_END:
                    target_face_type = t
                elif t >= SPTF.MODE_BEGIN and t < SPTF.MODE_END:
                    mode_type = t

            if img_type == SPTF.NONE:
                raise ValueError('expected IMG_ type')

            if img_type == SPTF.IMG_LANDMARKS_ARRAY:
                l = sample.landmarks
                l = np.concatenate([np.expand_dims(l[:, 0] / w, -1), np.expand_dims(l[:, 1] / h, -1)], -1)
                l = np.clip(l, 0.0, 1.0)
                img = l
            elif img_type == SPTF.IMG_PITCH_YAW_ROLL or img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
                pitch_yaw_roll = sample.pitch_yaw_roll
                if pitch_yaw_roll is not None:
                    pitch, yaw, roll = pitch_yaw_roll
                else:
                    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(sample.landmarks)
                if params['flip']:
                    yaw = -yaw

                if img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
                    pitch = (pitch + 1.0) / 2.0
                    yaw = (yaw + 1.0) / 2.0
                    roll = (roll + 1.0) / 2.0

                img = (pitch, yaw, roll)
            else:
                if mode_type == SPTF.NONE:
                    raise ValueError('expected MODE_ type')

                def do_transform(img, mask):
                    warp = (img_type==SPTF.IMG_WARPED or img_type==SPTF.IMG_WARPED_TRANSFORMED)
                    transform = (img_type==SPTF.IMG_WARPED_TRANSFORMED or img_type==SPTF.IMG_TRANSFORMED)
                    flip = img_type != SPTF.IMG_WARPED

                    img = imagelib.warp_by_params (params, img, warp, transform, flip, True)
                    if mask is not None:
                        mask = imagelib.warp_by_params (params, mask, warp, transform, flip, False)
                        if len(mask.shape) == 2:
                            mask = mask[...,np.newaxis]

                        img = np.concatenate( (img, mask ), -1 )
                    return img

                img = cached_images.get(img_type, None)
                if img is None:

                    img = sample_bgr
                    mask = None
                    cur_sample = sample

                    if is_face_sample:
                        if motion_blur is not None:
                            chance, mb_range = motion_blur
                            chance = np.clip(chance, 0, 100)

                            if np.random.randint(100) < chance:
                                mb_range = [3, 5, 7, 9][: np.clip(mb_range, 0, 3) + 1]
                                dim = mb_range[np.random.randint(len(mb_range))]
                                img = imagelib.LinearMotionBlur(img, dim, np.random.randint(180))

                        mask = cur_sample.load_mask(extend_forehead=sample_process_options.extend_forehead)

                        if cur_sample.ie_polys is not None:
                            cur_sample.ie_polys.overlay_mask(mask)

                    if sample.face_type == FaceType.MARK_ONLY:
                        if mask is not None:
                            img = np.concatenate( (img, mask), -1 )
                    else:
                        img = do_transform (img, mask)

                    cached_images[img_type] = img

                if is_face_sample and target_face_type != SPTF.NONE:
                    ft = SPTF_FACETYPE_TO_FACETYPE[target_face_type]
                    if ft > sample.face_type:
                        raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, ft) )

                    if sample.face_type == FaceType.MARK_ONLY:
                        img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, sample.shape[0], ft), (sample.shape[0],sample.shape[0]), flags=cv2.INTER_CUBIC )

                        mask = img[...,3:4] if img.shape[2] > 3 else None
                        img  = img[...,0:3]
                        img = do_transform (img, mask)
                        img = cv2.resize( img, (resolution,resolution), cv2.INTER_CUBIC )
                    else:
                        img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, ft), (resolution,resolution), flags=cv2.INTER_CUBIC )

                else:
                    img = cv2.resize(img, (resolution, resolution), cv2.INTER_CUBIC)

                if random_sub_res != 0:
                    sub_size = resolution - random_sub_res
                    rnd_state = np.random.RandomState(sample_rnd_seed + random_sub_res)
                    start_x = rnd_state.randint(sub_size + 1)
                    start_y = rnd_state.randint(sub_size + 1)
                    img = img[start_y:start_y + sub_size, start_x:start_x + sub_size, :]

                img = np.clip(img, 0, 1)
                img_bgr = img[..., 0:3]
                img_mask = img[..., 3:4]

                if apply_ct and ct_sample is not None:
                    if ct_sample_bgr is None:
                        ct_sample_bgr = ct_sample.load_bgr()

                    if apply_ct == ColorTransferMode.LCT:
                        img_bgr = imagelib.linear_color_transfer(img_bgr, ct_sample_bgr)

                    elif ColorTransferMode.RCT <= apply_ct <= ColorTransferMode.MASKED_RCT_PAPER_CLIP:
                        ct_options = {
                                ColorTransferMode.RCT:                      (False, False, False),
                                ColorTransferMode.RCT_CLIP:                 (False, False, True),
                                ColorTransferMode.RCT_PAPER:                (False, True, False),
                                ColorTransferMode.RCT_PAPER_CLIP:           (False, True, True),
                                ColorTransferMode.MASKED_RCT:               (True, False, False),
                                ColorTransferMode.MASKED_RCT_CLIP:          (True, False, True),
                                ColorTransferMode.MASKED_RCT_PAPER:         (True, True, False),
                                ColorTransferMode.MASKED_RCT_PAPER_CLIP:    (True, True, True),
                            }

                        use_masks, use_paper, use_clip = ct_options[apply_ct]
                        if not use_masks:
                            img_bgr = imagelib.reinhard_color_transfer(img_bgr, ct_sample_bgr, clip=use_clip,
                                                                       preserve_paper=use_paper)
                        else:
                            if ct_sample_mask is None:
                                ct_sample_mask = ct_sample.load_mask(extend_forehead=sample_process_options.extend_forehead)

                            img_bgr = imagelib.reinhard_color_transfer(img_bgr, ct_sample_bgr, clip=use_clip,
                                                                       preserve_paper=use_paper, source_mask=img_mask,
                                                                       target_mask=ct_sample_mask)

                if normalize_std_dev:
                    img_bgr = (img_bgr - img_bgr.mean((0, 1))) / img_bgr.std((0, 1))
                elif normalize_vgg:
                    img_bgr = np.clip(img_bgr * 255, 0, 255)
                    img_bgr[:, :, 0] -= 103.939
                    img_bgr[:, :, 1] -= 116.779
                    img_bgr[:, :, 2] -= 123.68

                if mode_type == SPTF.MODE_BGR:
                    img = img_bgr
                elif mode_type == SPTF.MODE_BGR_SHUFFLE:
                    rnd_state = np.random.RandomState(sample_rnd_seed)
                    img = np.take(img_bgr, rnd_state.permutation(img_bgr.shape[-1]), axis=-1)
                elif mode_type == SPTF.MODE_G:
                    img = np.concatenate((np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), -1), img_mask), -1)
                elif mode_type == SPTF.MODE_GGG:
                    img = np.concatenate(
                        (np.repeat(np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), -1), (3,), -1), img_mask),
                        -1)
                elif mode_type == SPTF.MODE_M and is_face_sample:
                    img = img_mask

                if not debug:
                    if normalize_tanh:
                        img = np.clip(img * 2.0 - 1.0, -1.0, 1.0)
                    else:
                        img = np.clip(img, 0.0, 1.0)

            outputs.append(img)

        if debug:
            result = []

            for output in outputs:
                if output.shape[2] < 4:
                    result += [output, ]
                elif output.shape[2] == 4:
                    result += [output[..., 0:3] * output[..., 3:4], ]

            return result
        else:
            return outputs


"""
        close_sample = sample.close_target_list[ np.random.randint(0, len(sample.close_target_list)) ] if sample.close_target_list is not None else None
        close_sample_bgr = close_sample.load_bgr() if close_sample is not None else None

        if debug and close_sample_bgr is not None:
            LandmarksProcessor.draw_landmarks (close_sample_bgr, close_sample.landmarks, (0, 1, 0))
        RANDOM_CLOSE               = 0x00000040, #currently unused
        MORPH_TO_RANDOM_CLOSE      = 0x00000080, #currently unused

if f & SPTF.RANDOM_CLOSE != 0:
                img_type += 10
            elif f & SPTF.MORPH_TO_RANDOM_CLOSE != 0:
                img_type += 20
if img_type >= 10 and img_type <= 19: #RANDOM_CLOSE
    img_type -= 10
    img = close_sample_bgr
    cur_sample = close_sample

elif img_type >= 20 and img_type <= 29: #MORPH_TO_RANDOM_CLOSE
    img_type -= 20
    res = sample.shape[0]

    s_landmarks = sample.landmarks.copy()
    d_landmarks = close_sample.landmarks.copy()
    idxs = list(range(len(s_landmarks)))
    #remove landmarks near boundaries
    for i in idxs[:]:
        s_l = s_landmarks[i]
        d_l = d_landmarks[i]
        if s_l[0] < 5 or s_l[1] < 5 or s_l[0] >= res-5 or s_l[1] >= res-5 or \
            d_l[0] < 5 or d_l[1] < 5 or d_l[0] >= res-5 or d_l[1] >= res-5:
            idxs.remove(i)
    #remove landmarks that close to each other in 5 dist
    for landmarks in [s_landmarks, d_landmarks]:
        for i in idxs[:]:
            s_l = landmarks[i]
            for j in idxs[:]:
                if i == j:
                    continue
                s_l_2 = landmarks[j]
                diff_l = np.abs(s_l - s_l_2)
                if np.sqrt(diff_l.dot(diff_l)) < 5:
                    idxs.remove(i)
                    break
    s_landmarks = s_landmarks[idxs]
    d_landmarks = d_landmarks[idxs]
    s_landmarks = np.concatenate ( [s_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )
    d_landmarks = np.concatenate ( [d_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )
    img = imagelib.morph_by_points (sample_bgr, s_landmarks, d_landmarks)
    cur_sample = close_sample
else:
    """
