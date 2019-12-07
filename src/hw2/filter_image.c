#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"

#define TWOPI 6.2831853


void l1_normalize(image im) {
    for (int c = 0; c < im.c; c++) {
        float sum = 0.0;
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                sum += get_pixel(im, x, y, c);
            }
        }
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                if (sum != 0) {
                    set_pixel(im, x, y, c, get_pixel(im, x, y, c) / sum);
                } else {
                    set_pixel(im, x, y, c, 1.0 / im.w / im.h);
                }
            }
        }
    }
}

image make_box_filter(int w) {
    image ret = make_image(w, w, 1);
    for (int y = 0; y < w; y++) {
        for (int x = 0; x < w; x++) {
            set_pixel(ret, x, y, 0, 1.0 / w / w);
        }
    }
    return ret;
}

image convolve_image(image im, image filter, int preserve) {
    assert(filter.c == im.c || filter.c == 1);

    image ret;
    if (preserve) {
        ret = make_image(im.w, im.h, im.c);
    } else {
        ret = make_image(im.w, im.h, 1);
    }

    for (int h = 0; h < im.h; h++) {
        for (int w = 0; w < im.w; w++) {
            float total_sum = 0.0;
            for (int c = 0; c < im.c; c++) {
                float sum = 0.0;
                for (int b = 0; b < filter.h; b++) {  // b: kernel index h direction
                    for (int a = 0; a < filter.w; a++) {  // a: kernel index w direction
                        int x = w - filter.w/2 + a;  // pixel w direction
                        int y = h - filter.h/2 + b;  // pixel h direction
                        if (filter.c == 1) { 
                            sum += get_pixel(im, x, y, c) * get_pixel(filter, a, b, 0);
                        } else {
                            sum += get_pixel(im, x, y, c) * get_pixel(filter, a, b, c);
                        }
                    }
                }
                if (preserve) {
                    set_pixel(ret, w, h, c, sum);
                }
                total_sum += sum;
            }
            if (!preserve) {
                set_pixel(ret, w, h, 0, total_sum);
            }
        }
    }

    return ret;
}

image make_highpass_filter() {
    image ret = make_image(3, 3, 1);
    set_pixel(ret, 1, 0, 0, -1.0);
    set_pixel(ret, 0, 1, 0, -1.0);
    set_pixel(ret, 2, 1, 0, -1.0);
    set_pixel(ret, 1, 2, 0, -1.0);
    set_pixel(ret, 1, 1, 0, 4.0);
    return ret;
}

image make_sharpen_filter() {
    image ret = make_image(3, 3, 1);
    set_pixel(ret, 1, 0, 0, -1.0);
    set_pixel(ret, 0, 1, 0, -1.0);
    set_pixel(ret, 2, 1, 0, -1.0);
    set_pixel(ret, 1, 2, 0, -1.0);
    set_pixel(ret, 1, 1, 0, 5.0);
    return ret;
}

image make_emboss_filter() {
    image ret = make_image(3, 3, 1);
    set_pixel(ret, 0, 0, 0, -2.0);
    set_pixel(ret, 1, 0, 0, -1.0);
    set_pixel(ret, 0, 1, 0, -1.0);
    set_pixel(ret, 1, 1, 0, 1.0);
    set_pixel(ret, 2, 1, 0, 1.0);
    set_pixel(ret, 1, 2, 0, 1.0);
    set_pixel(ret, 2, 2, 0, 2.0);
    return ret;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: Emboss and Sharpen should use preserve, since in these cases we still want our image to be colored, and we  
//      those filters are also just designed to make artifacts to the image but not finding some features.
//      The high pass filter should not use preserve since we are using it to find edges and no need to keep the image colored.

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: We need to clamp the image after applying all of the 3 filters. Since during the convolution calculation, 
//      it is possible for the value to overflow.

image make_gaussian_filter(float sigma) {
    int w = ceil(sigma * 6);
    if (w % 2 == 0) {
        w += 1;
    }
    int r = w / 2;
    image ret = make_image(w, w, 1);

    for (int j = -r; j <= r; j++) {
        for (int i = -r; i <= r; i++) {
            float value = (1.0 / (TWOPI * sigma * sigma)) * exp(-1.0 * (i * i + j * j) / (2.0 * sigma * sigma));
            set_pixel(ret, i+r, j+r, 0, value);
        }
    }
    l1_normalize(ret);
    return ret;
}

image add_image(image a, image b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c) {
        return make_image(0,0,0);
    }
    image ret = make_image(a.w, a.h, a.c);
    for (int c = 0; c < a.c; c++) {
        for (int y = 0; y < a.h; y++) {
            for (int x = 0; x < a.w; x++) {
                set_pixel(ret, x, y, c, get_pixel(a, x, y, c) + get_pixel(b, x, y, c));
            }
        }
    }
    return ret;
}

image sub_image(image a, image b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c) {
        return make_image(0,0,0);
    }
    image ret = make_image(a.w, a.h, a.c);
    for (int c = 0; c < a.c; c++) {
        for (int y = 0; y < a.h; y++) {
            for (int x = 0; x < a.w; x++) {
                set_pixel(ret, x, y, c, get_pixel(a, x, y, c) - get_pixel(b, x, y, c));
            }
        }
    }
    return ret;
}

image mix_image(image base, image brush) {
    int cx = brush.w / 2;
    int cy = brush.h / 2;

    image ret = make_image(base.w, base.h, base.c);

    for (int c = 0; c < base.c; c++) {
        for (int y = 0; y < base.h; y++) {
            for (int x = 0; x < base.w; x++) {
                if (x > brush.w || y > brush.h) {
                    set_pixel(ret, x, y, c, get_pixel(base, x, y, c));
                    continue;
                }
                float opacity = get_pixel(brush, x, y, 0);
                set_pixel(ret, x, y, c, get_pixel(base, x, y, c) * (1 - opacity) + get_pixel(base, cx, cy, c) * opacity);
            }
        }
    }
    return ret;

}


image make_gx_filter() {
    image ret = make_image(3, 3, 1);
    set_pixel(ret, 0, 0, 0, -1.0);
    set_pixel(ret, 0, 1, 0, -2.0);
    set_pixel(ret, 0, 2, 0, -1.0);
    set_pixel(ret, 2, 0, 0, 1.0);
    set_pixel(ret, 2, 1, 0, 2.0);
    set_pixel(ret, 2, 2, 0, 1.0);
    return ret;
}

image make_gy_filter() {
    image ret = make_image(3, 3, 1);
    set_pixel(ret, 0, 0, 0, -1.0);
    set_pixel(ret, 1, 0, 0, -2.0);
    set_pixel(ret, 2, 0, 0, -1.0);
    set_pixel(ret, 0, 2, 0, 1.0);
    set_pixel(ret, 1, 2, 0, 2.0);
    set_pixel(ret, 2, 2, 0, 1.0);
    return ret;
}

void feature_normalize(image im) {
    float min = get_pixel(im, 0, 0, 0);
    float max = get_pixel(im, 0, 0, 0);
    for (int c = 0; c < im.c; c++) {
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                min = MIN(min, get_pixel(im, x, y, c));
                max = MAX(max, get_pixel(im, x, y, c));
            }
        }
    }
    for (int c = 0; c < im.c; c++) {
        float range = max - min;
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                if (range != 0) {
                    set_pixel(im, x, y, c, (get_pixel(im, x, y, c) - min) / range);
                } else {
                    set_pixel(im, x, y, c, 0);
                }
            }
        }
    }
}

image *sobel_image(image im) {
    image* ret = calloc(2, sizeof(image));
    ret[0] = make_image(im.w, im.h, 1);
    ret[1] = make_image(im.w, im.h, 1);

    image fgx = make_gx_filter();
    image fgy = make_gy_filter();
    image gx = convolve_image(im, fgx, 0);
    image gy = convolve_image(im, fgy, 0);

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float gx_val = get_pixel(gx, x, y, 0);
            float gy_val = get_pixel(gy, x, y, 0);
            set_pixel(ret[0], x, y, 0, sqrtf(gx_val * gx_val + gy_val * gy_val));
            set_pixel(ret[1], x, y, 0, atan2f(gy_val, gx_val));
        }
    }
    free_image(gx);
    free_image(gy);
    free_image(fgx);
    free_image(fgy);
    return ret;
}

image colorize_sobel(image im) {
    image *res = sobel_image(im);
    image mag = res[0];
    image theta = res[1];
    feature_normalize(mag);
    feature_normalize(theta);

    image ret = make_image(im.w, im.h, 3);
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            set_pixel(ret, x, y, 0, get_pixel(theta, x, y, 0));
            set_pixel(ret, x, y, 1, get_pixel(mag, x, y, 0));
            set_pixel(ret, x, y, 2, get_pixel(mag, x, y, 0));
        }
    }
    hsv_to_rgb(ret);
    return ret;
}