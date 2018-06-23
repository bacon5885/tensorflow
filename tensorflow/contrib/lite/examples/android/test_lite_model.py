"""Load the tflite model in python and use it to do inference on an image.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
assert tf.__version__.startswith('1.9') or tf.__version__.startswith('1.10')
import math


# interpreter = tf.contrib.lite.Interpreter(model_path='QUANTIZED_UINT8.lite')
interpreter = tf.contrib.lite.Interpreter(model_path='FLOAT.lite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']
print('input---------------------------')
print(input_details)
print('')
print('output--------------------------')
print(output_details)

pil_img = Image.open('prada2.jpg').resize((300, 300))
# pil_img = Image.open('starbucks.jpg').resize((300, 300))
# pil_img = Image.open('cocacola.jpg').resize((300, 300))
input_data = np.array(pil_img, dtype=np.uint8)
input_data = input_data.astype(np.float32) / 128. - 1.
input_data = np.expand_dims(input_data, 0)
print('input shape -------------------------')
print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
for i, output_detail in enumerate(output_details):
    output_tensor = interpreter.get_tensor(output_detail['index'])
    # print('output shape: {}'.format(output_tensor.shape))
    # print(output_tensor)
    if i == 0:
        output_locations = output_tensor
    elif i == 1:
        output_probs = output_tensor

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

output_probs = softmax(np.squeeze(output_probs), axis=-1)
print('non-trivial output probs:')
print(output_probs[output_probs[:, 1] > 0.3, :])
print('----------------------------------------------')
print('highest score')
print(max(output_probs[:, 1]))
print('---------------------------------------')

idx = np.argmax(output_probs[:, 1])

# get first bbox with high enough score

if output_probs[idx, 1] > 0.3:
    selected_bb = output_locations[0, idx, 0, :]
else:
    selected_bb = None

def load_box_priors():
    with open('assets/box_priors.txt') as f:
        box_priors = []
        for line in f:
            if len(line.strip()) < 5:
                continue
            fields = line.strip().split(' ')
            vec = [float(x) for x in fields]
            box_priors.append(vec)
        return box_priors

boxPriors = load_box_priors()
Y_SCALE = X_SCALE = 10.
H_SCALE = W_SCALE = 5.
def decodeCenterSizeBoxes(predictions):
    # 'predictions' has the predicted bounding boxes.
    for i in range(1917):
        ycenter = predictions[0][i][0][0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
        xcenter = predictions[0][i][0][1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
        h = math.exp(predictions[0][i][0][2] / H_SCALE) * boxPriors[2][i];
        w = math.exp(predictions[0][i][0][3] / W_SCALE) * boxPriors[3][i];

        ymin = ycenter - h / 2.;
        xmin = xcenter - w / 2.;
        ymax = ycenter + h / 2.;
        xmax = xcenter + w / 2.;

        predictions[0][i][0][0] = ymin;
        predictions[0][i][0][1] = xmin;
        predictions[0][i][0][2] = ymax;
        predictions[0][i][0][3] = xmax;
    return predictions

decodeCenterSizeBoxes(output_locations)

# TODO: Non maximum suppression on output boxes.

def draw_bb(pil_img, bb):
    dr = ImageDraw.Draw(pil_img)
    ymin, xmin, ymax, xmax = tuple(bb)
    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin
    x = int(x * pil_img.width)
    y = int(y * pil_img.height)
    width = int(width * pil_img.width)
    height = int(height * pil_img.height)
    dr.line(((x, y), (x + width, y)), width=5, fill='blue')
    dr.line(((x + width, y), (x + width, y + height)), width=5, fill='blue')
    dr.line(((x + width, y + height), (x, y + height)), width=5, fill='blue')
    dr.line(((x, y + height), (x, y)), width=5, fill='blue')
    return pil_img

if selected_bb is not None:
    pil_img = draw_bb(pil_img, selected_bb)
    pil_img.save('detection_result.jpg')
    print('saved detection result as image')
