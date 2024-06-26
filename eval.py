import requests
from io import BytesIO
from PIL import Image
from aura_sr import AuraSR
import typer
import torch
import time

aura_sr = AuraSR.from_pretrained()


def upscale_image(
    url: str = "https://images.unsplash.com/photo-1563729784474-d77dbb933a9e?q=80&w=2187&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
):
    image = load_image_from_url(url)
    image.thumbnail((512, 512))

    start = time.time()
    upscaled_image = aura_sr.upscale_4x(image)
    torch.cuda.synchronize()
    print(f"Time taken to upscale the image: {time.time()-start:.2f} seconds")

    image.save("original_image.jpg")
    upscaled_image.save("upscaled_image.jpg")


def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)


if __name__ == "__main__":
    typer.run(upscale_image)
