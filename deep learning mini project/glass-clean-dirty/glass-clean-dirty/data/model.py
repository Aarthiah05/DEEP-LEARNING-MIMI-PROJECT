# eda.ipynb cell
import os, random
from PIL import Image
import matplotlib.pyplot as plt

base = "data"

classes = os.listdir(base)
for c in classes:
    print(c, len(os.listdir(os.path.join(base, c))))

# show sample images
fig, axes = plt.subplots(2,4, figsize=(12,6))
i = 0
for c in classes[:2]:
    imgs = os.listdir(os.path.join(base,c))[:4]
    for im in imgs:
        ax = axes.flatten()[i]
        img = Image.open(os.path.join(base,c,im)).convert("RGB")
        ax.imshow(img)
        ax.set_title(c)
        ax.axis("off")
        i += 1
plt.show()
