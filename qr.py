from PIL import Image
from pyzbar.pyzbar import decode
import helpers as Helpers

def reader(bgr_image):
    rgb_image = Helpers.convert_color(bgr_image, "BGR", "RGB")
    pil_im = Image.fromarray(rgb_image)

    img = pil_im.convert('L')

    result = decode(img)

    decoded_qr_code = result[0].data
    decoded_qr_code = decoded_qr_code.decode("utf-8") # convert from byte to string

    return decoded_qr_code
