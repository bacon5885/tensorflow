"""Load the tflite model in python and use it to do inference on an image.
"""

import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('1.9') or tf.__version__.startswith('1.10')


interpreter = tf.contrib.lite.Interpreter(model_path='QUANTIZED_UINT8.lite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']
print('input---------------------------')
print(input_details)
print('')
print('output--------------------------')
print(output_details)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
for output_detail in output_details:
    output_tensor = interpreter.get_tensor(output_detail['index'])
    print('output shape: {}'.format(output_tensor.shape))
    print(output_tensor)


