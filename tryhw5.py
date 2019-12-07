from uwimg import *

im = load_image("data/flower.jpg")
res = apply_brushes(im)
save_image(res, "flower")


im = load_image("data/sunset_small.jpg")
res = apply_brushes(im)
save_image(res, "sunset")

