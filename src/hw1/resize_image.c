#include <math.h>
#include "image.h"


float nn_interpolate(image im, float x, float y, int c) {
    return get_pixel(im, round(x), round(y), c);
}

image nn_resize(image im, int w, int h) {
    image ret = make_image(w, h, im.c);
    float w_gap = 1.0 * im.w / ret.w;
    float h_gap = 1.0 * im.h / ret.h;
    for (int c = 0; c < im.c; c++) {
        for (int y = 0; y < ret.h; y++) {
            for (int x = 0; x < ret.w; x++) {
                float a = -0.5 + w_gap * (x + 0.5);
                float b = -0.5 + h_gap * (y + 0.5);
                set_pixel(ret, x, y, c, nn_interpolate(im, a, b, c));
            }
        }
    }
    return ret;
}

float bilinear_interpolate(image im, float x, float y, int c) {
    int x_l = floor(x);
    int x_r = ceil(x);
    int y_l = floor(y);
    int y_r = ceil(y);
    float ul = get_pixel(im, x_l, y_l, c);
    float ur = get_pixel(im, x_r, y_l, c);
    float dl = get_pixel(im, x_l, y_r, c);
    float dr = get_pixel(im, x_r, y_r, c);

    float q1 = dl * (y - y_l) + ul * (y_r - y);
    float q2 = dr * (y - y_l) + ur * (y_r - y);
    return q2 * (x - x_l) + q1 * (x_r - x);
}

image bilinear_resize(image im, int w, int h) {
    image ret = make_image(w, h, im.c);
    float w_gap = 1.0 * im.w / ret.w;
    float h_gap = 1.0 * im.h / ret.h;
    for (int c = 0; c < im.c; c++) {
        for (int y = 0; y < ret.h; y++) {
            for (int x = 0; x < ret.w; x++) {
                float a = -0.5 + w_gap * (x + 0.5);
                float b = -0.5 + h_gap * (y + 0.5);
                set_pixel(ret, x, y, c, bilinear_interpolate(im, a, b, c));
            }
        }
    }
    return ret;
}

