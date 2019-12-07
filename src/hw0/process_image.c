#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"


float get_pixel(image im, int x, int y, int c) {
    if (x < 0) {
        x = 0;
    } else if (x >= im.w) {
        x = im.w - 1;
    }
    if (y < 0) {
        y = 0;
    } else if (y >= im.h) {
        y = im.h - 1;
    }
    return im.data[x + y * im.w + c * im.w * im.h];
}

void set_pixel(image im, int x, int y, int c, float v) {
    if (x < 0) {
        x = 0;
    } else if (x >= im.w) {
        x = im.w - 1;
    }
    if (y < 0) {
        y = 0;
    } else if (y >= im.h) {
        y = im.h - 1;
    }
    im.data[x + y * im.w + c * im.w * im.h] = v;
}

image copy_image(image im) {
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im) {
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float r = im.data[x + y * im.w + 0 * im.w * im.h]; 
            float g = im.data[x + y * im.w + 1 * im.w * im.h];
            float b = im.data[x + y * im.w + 2 * im.w * im.h];
            gray.data[x + y * im.w] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
    return gray;
}

void shift_image(image im, int c, float v) {
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            im.data[x + y * im.w + c * im.w * im.h] += v;
        }
    }
}

void clamp_image(image im) {
    for (int c = 0; c < im.c; c++) {
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                if (im.data[x + y * im.w + c * im.w * im.h] < 0.0) {
                    im.data[x + y * im.w + c * im.w * im.h] = 0.0;
                } else if (im.data[x + y * im.w + c * im.w * im.h] > 1.0) {
                    im.data[x + y * im.w + c * im.w * im.h] = 1.0;
                }
                
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c) {
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c) {
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im) {
    if (im.c != 3) {
        printf("rgb_to_hsv failed: image channels must be 3\n");
        return;
    }
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float r = im.data[x + y * im.w + 0 * im.w * im.h]; 
            float g = im.data[x + y * im.w + 1 * im.w * im.h];
            float b = im.data[x + y * im.w + 2 * im.w * im.h];

            float v = three_way_max(r, g, b);
            float m = three_way_min(r, g, b);
            float c = v - m;
            float s = 0; 
            if (r > 0 || g > 0 || b > 0) {
                s = c / v;
            }

            float hh = 0; 
            if (c != 0) {
                if (v == r) {
                    hh = (g - b) / c;
                } else if (v == g) {
                    hh = (b - r) / c + 2;
                } else if (v == b) {
                    hh = (r - g) / c + 4;
                }
            }
            float h = 0; 
            if (hh < 0) {
                h = hh / 6 + 1;
            } else {
                h = hh / 6;
            }
            if (h < 0) {
                h += 1;
            } else if (h >= 1) {
                h -= 1;
            }

            im.data[x + y * im.w + 0 * im.w * im.h] = h;
            im.data[x + y * im.w + 1 * im.w * im.h] = s;
            im.data[x + y * im.w + 2 * im.w * im.h] = v;
        }
    }
}

void hsv_to_rgb(image im) {
    if (im.c != 3) {
        printf("hsv_to_rgb failed: image channels must be 3\n");
        return;
    }
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float h = im.data[x + y * im.w + 0 * im.w * im.h]; 
            float s = im.data[x + y * im.w + 1 * im.w * im.h];
            float v = im.data[x + y * im.w + 2 * im.w * im.h];

            float c = v * s;
            float m = v - c;
            float hh = h * 6;

            float temp = hh;
            while (temp >= 2.0) { 
                temp -= 2.0;
            }
            temp -= 1.0;
            if (temp < 0) {
                temp = -temp;
            }
            temp = 1.0 - temp;
            float x2 = c * temp;

            float rr = 0; 
            float gg = 0; 
            float bb = 0;
            
            if (hh >= 0 && hh < 1) {
                rr = c;
                gg = x2;
                bb = 0;
            } else if (hh >= 1 && hh < 2) {
                rr = x2;
                gg = c;
                bb = 0;
            } else if (hh >= 2 && hh < 3) {
                rr = 0; 
                gg = c;
                bb = x2;
            } else if (hh >= 3 && hh < 4) {
                rr = 0;
                gg = x2;
                bb = c;
            } else if (hh >= 4 && hh < 5) {
                rr = x2;
                gg = 0; 
                bb = c;
            } else if (hh >= 5 && hh < 6) {
                rr = c;
                gg = 0;
                bb = x2;
            } 
            
            float r = 0; 
            float g = 0; 
            float b = 0;

            if (c == 0) {
                r = v;
                g = v;
                b = v;
            } else {
                r = rr + m;
                g = gg + m;
                b = bb + m;
            }

            im.data[x + y * im.w + 0 * im.w * im.h] = r;
            im.data[x + y * im.w + 1 * im.w * im.h] = g;
            im.data[x + y * im.w + 2 * im.w * im.h] = b;
        }
    }

}

void scale_image(image im, int c, float v) {
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            im.data[x + y * im.w + c * im.w * im.h] *= v;
        }
    }
}





/*
void rgb_to_hcl(image im) {
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float r = im.data[x + y * im.w + 0 * im.w * im.h]; 
            float g = im.data[x + y * im.w + 1 * im.w * im.h];
            float b = im.data[x + y * im.w + 2 * im.w * im.h];

            float v = three_way_max(r, g, b);
            float m = three_way_min(r, g, b);
            float c = v - m;
            float s = 0; 
            if (r > 0 || g > 0 || b > 0) {
                s = c / v;
            }

            float hh = 0; 
            if (c != 0) {
                if (v == r) {
                    hh = (g - b) / c;
                } else if (v == g) {
                    hh = (b - r) / c + 2;
                } else if (v == b) {
                    hh = (r - g) / c + 4;
                }
            }
            float h = 0; 
            if (hh < 0) {
                h = hh / 6 + 1;
            } else {
                h = hh / 6;
            }
            if (h < 0) {
                h += 1;
            } else if (h >= 1) {
                h -= 1;
            }

            im.data[x + y * im.w + 0 * im.w * im.h] = h;
            im.data[x + y * im.w + 1 * im.w * im.h] = s;
            im.data[x + y * im.w + 2 * im.w * im.h] = v;
        }
    }
}

*/