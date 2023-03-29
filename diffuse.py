import cv2
import numpy as np
import diffusers 
from diffusers import StableDiffusionControlNetPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from PIL import Image


image = load_image("https://huggingface.co/krea/aesthetic-controlnet/resolve/main/krea.jpg")

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

pipe = StableDiffusionControlNetPipeline.from_pretrained("krea/aesthetic-controlnet").to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

output = pipe(
    "fantasy flowers",
    canny_image,
    num_inference_steps=20,
    guidance_scale=4,
    width=768,
    height=768,
)

result = output.images[0]
result.save("result.png")
