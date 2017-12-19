from skimage.transform import resize
from util import apply_forward_convs, apply_reverse_convs
from util import mobilenet_layers
from util import plot_conv_maps
import numpy as np
import skimage.io as skio
import skvideo.io
import sys
import tensorflow as tf

if __name__ == '__main__':
    mode = sys.argv[1]
    end_layer = sys.argv[2]
    img_height = int(sys.argv[3])
    img_width = int(sys.argv[4])
    crop_x = int(sys.argv[5])
    crop_y = int(sys.argv[6])
    crop_w = int(sys.argv[7])
    crop_h = int(sys.argv[8])
    show_image = sys.argv[9] == 'show'
    if 'mp4' in sys.argv[10]:
        offset = int(sys.argv[11])
        vreader = skvideo.io.vreader(sys.argv[10])
        for _ in range(offset): next(vreader)
        full_img = next(vreader)
    else:
        full_img = skio.imread(sys.argv[10])

    layer_idx = map(lambda x: x[0], mobilenet_layers).index(end_layer)
    layers = mobilenet_layers[:layer_idx + 1]
    input_placeholder = \
        tf.placeholder('float32', [1, img_height, img_width, 1])
    input_img = np.zeros([img_height, img_width])
    input_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = 1
    input_img = input_img.reshape(1, img_height, img_width, 1)
    input_layer = ('input', 1, 1, 1)

    sess = tf.Session()
    if mode == 'forward':
        conv_layers = apply_forward_convs(input_placeholder, layers)
        conv_outputs = sess.run(conv_layers,
                                feed_dict={input_placeholder: input_img})
        full_img = resize(full_img, (img_height, img_width))
        plot_conv_maps(full_img, [input_img] + conv_outputs,
                       [input_layer] + layers, show_image=show_image)
    elif mode == 'backward':
        conv_layers = apply_reverse_convs(input_placeholder, layers)
        conv_outputs = sess.run(conv_layers,
                                feed_dict={input_placeholder: input_img})
        full_img = resize(full_img, (conv_outputs[0].shape[1],
                                     conv_outputs[0].shape[2]))
        plot_conv_maps(full_img, conv_outputs + [input_img],
                       layers + [input_layer], show_image=show_image)
    else:
        raise Exception('No matching mode for mode %s' % mode)
