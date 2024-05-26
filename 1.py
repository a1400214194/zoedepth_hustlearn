import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


# # ZoeD_K
# conf = get_config("zoedepth", "infer", config_version="kitti")
# model_zoe_k = build_model(conf)
#
# # ZoeD_NK
# conf = get_config("zoedepth_nk", "infer")
# model_zoe_nk = build_model(conf)
##### sample prediction
# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("D:/projects/MonoDepth-main/datasets/sync/basement_0001a/rgb_00000.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor



# Tensor
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)



# From URL
from zoedepth.utils.misc import get_image_from_url

# Example URL
URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


image = get_image_from_url(URL)  # fetch
depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "D:/projects/MonoDepth-main/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)



# Convert to PIL Image and save colored output
colored_image = Image.fromarray(colored)
fpath_colored = "D:/projects/MonoDepth-main/output_colored.png"
colored_image.save(fpath_colored)
