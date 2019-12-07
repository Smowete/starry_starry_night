from uwimg import *

im = load_image("data/dog.jpg")
brush = load_image("brushes/b02.png")
res = mix_image(im, brush)
save_image(res, "mixmixmix")
