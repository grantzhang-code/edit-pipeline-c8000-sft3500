"""Training-identical Source image preparation for the vLLM engine.

Training (ms-swift 4.4.2, Qwen3VLTemplate) read RGB MDS images that had already
been alpha-composited onto white, shrank images above max_pixels with the
template's rescale_image (bilinear), resized the result with qwen_vl_utils
0.0.14 fetch_image (PIL default bicubic) and called the image processor with
do_resize=False. vLLM must therefore receive the resized image and must not
resize again (mm_processor_kwargs do_resize=False).

swift maps MIN_PIXELS=3136 / IMAGE_MAX_TOKEN_NUM=1024 to qwen_vl_utils token
limits with factor = patch 16 x merge 2 = 32: min 3136 // 32**2 = 3 tokens,
max 1024 tokens.
"""
import math

from PIL import Image

IMAGE_COLOR_POLICY = 'alpha_composited_on_white_like_training_mds'
IMAGE_RESIZE_POLICY = 'swift_rescale_image_bilinear_1mp_then_qwen_vl_utils_0.0.14_bicubic_then_do_resize_false'
FACTOR = 32
MIN_PIXELS = (3136 // FACTOR ** 2) * FACTOR ** 2
MAX_PIXELS = 1024 * FACTOR ** 2
MAX_RATIO = 200


def training_rgb(im):
    """Match the training MDS sources (mode original_resolution_rgb_white_no_resize).

    Transparent sources were alpha-composited onto opaque white before training;
    a plain convert('RGB') would instead expose the hidden RGB under alpha.
    Opaque images keep the original convert('RGB') path unchanged.
    """
    if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
        rgba = im.convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba).convert('RGB')
    return im.convert('RGB')


# smart_resize is qwen_vl_utils 0.0.14 vision_process.smart_resize, verbatim
# except that the training limits above are bound explicitly.
def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    assert max_pixels >= min_pixels, 'The max_pixels of image must be greater than or equal to min_pixels.'
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f'absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}')
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def swift_rescale(image, max_pixels=MAX_PIXELS):
    """ms-swift 4.4.2 template._preprocess_inputs -> vision_utils.rescale_image.

    Training passed max_pixels=1048576 to the template, so Sources above 1 MP were
    first shrunk with torchvision T.Resize((int(h), int(w))). For PIL input that
    is PIL bilinear resampling; qwen_vl_utils then resizes this result again.
    """
    width, height = image.width, image.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return image
    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return image.resize((int(width_scaled), int(height_scaled)), Image.Resampling.BILINEAR)


def training_image(im):
    """RGB image exactly as the training template handed it to the processor."""
    image = swift_rescale(training_rgb(im))
    height, width = smart_resize(image.height, image.width)
    return image.resize((width, height))
