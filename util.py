from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mobilenet_layers = [
    ('conv1', 3, 3, 2),
    ('conv2_1/dw-sep', 3, 3, 1),
    ('conv2_2/dw-sep', 3, 3, 2),
    ('conv3_1/dw-sep', 3, 3, 1),
    ('conv3_3/dw-sep', 3, 3, 2),
    ('conv4_1/dw-sep', 3, 3, 1),
    ('conv4_2/dw-sep', 3, 3, 2),
    ('conv5_1/dw-sep', 3, 3, 1),
    ('conv5_2/dw-sep', 3, 3, 1),
    ('conv5_3/dw-sep', 3, 3, 1),
    ('conv5_4/dw-sep', 3, 3, 1),
    ('conv5_5/dw-sep', 3, 3, 1),
    ('conv5_6/dw-sep', 3, 3, 2),
    ('conv6/dw-sep', 3, 3, 1),
]

def apply_forward_convs(img, layers):
    tf_layer_ops = [img]
    for name, h, w, stride in layers:
        batch, height, width, channels = tf_layer_ops[-1].get_shape()
        mean_filter = np.ones([h, w, channels, channels], dtype=np.float32)
        mean_filter /= np.sum(mean_filter)
        filter_stride = [batch, stride, stride, channels]
        layer_op = tf.nn.conv2d(tf_layer_ops[-1], mean_filter,
                                filter_stride, 'SAME', name=name)
        tf_layer_ops.append(layer_op)
    tf_layer_ops = tf_layer_ops[1:]
    return tf_layer_ops

# Pass in layers in the forward order, omit any layers after where the
# feature map lies in the network.
def apply_reverse_convs(feature_map, layers):
    tf_layer_ops = [feature_map]
    for name, h, w, stride in reversed(layers):
        batch, height, width, channels = \
            [int(i) for i in tf_layer_ops[-1].get_shape()]
        mean_filter = np.ones([h, w, channels, channels], dtype=np.float32)
        mean_filter /= np.sum(mean_filter)
        target_size = [batch, height * stride, width * stride, channels]
        filter_stride = [batch, stride, stride, channels]
        layer_op = tf.nn.conv2d_transpose(
            tf_layer_ops[-1], mean_filter, target_size,
            filter_stride, 'SAME', name=name)
        tf_layer_ops.append(layer_op)
    tf_layer_ops = tf_layer_ops[1:][::-1]
    return tf_layer_ops

def plot_conv_maps(input_img, conv_outputs, layers, show_image=True):
    img_height, img_width, img_channels = input_img.shape
    grid_height = int(np.ceil(np.sqrt(len(conv_outputs))))
    grid_width = (len(conv_outputs) // grid_height) + 1
    fig, axes = plt.subplots(grid_height, grid_width,
                             figsize=(6*grid_height, 6*grid_width))
    axes = axes.flatten()
    for i in range(len(layers)):
        name, h, w, stride = layers[i]
        conv_output = np.squeeze(conv_outputs[i])
        ax = axes[i]
        if show_image:
            img_weight = resize(conv_output, (img_height, img_width))
            img_weight = np.dstack([img_weight] * img_channels)
            img_weight *= (1 / np.max(img_weight))
            weighted_img = input_img * img_weight 
            ax.imshow(weighted_img)
        else:
            ax.imshow(conv_output)
        ax.set_title(name)
    plt.savefig('out.png', bbox_inches='tight')
