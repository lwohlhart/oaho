import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl
import grasp as grasp_util
import cv2

from data_loader.oaho_loader import TFGraspConfigurationRecordDataLoader

config = {
  'train_files': ['data/oaho_synth_test.tfrecord'],
  'train_batch_size': 4
}
train_data = TFGraspConfigurationRecordDataLoader(config, mode='train')

train_dataset = train_data.input_fn()
train_iter = train_dataset.make_one_shot_iterator()
train_input, train_target = train_iter.get_next()



#grasp_util.Grasp([0.1,2], 1).as_bb

np.random.seed(42)
#tf.enable_eager_execution()

def get_example(depth, width, height, grasps):
  return tf.train.Example(features=tf.train.Features(feature={
    'depth': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(depth, -1))),
    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    'grasps': tf.train.Feature(float_list=tf.train.FloatList(value=grasps)),
    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
  }))

w,h = 640,480
d1 = np.random.normal(size=(h,w))
y = 50
x = 30
d1[y,x] = 10
g1 = np.array([34.1,35.1,0.5,40, 55.1,56.1, 0.2, 80])


g2 = np.array([34.1,35.1,0.5,40])

ex1_serialized = get_example(d1,w,h,g1).SerializeToString()
ex2_serialized = get_example(d1,w,h,g2).SerializeToString()


def fake_generator():
  while True:
    yield ex1_serialized if np.random.choice(2) == 0 else ex2_serialized

def parse_example(example):

  features = {
    'depth': tf.io.FixedLenFeature((640*480), tf.float32),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'height': tf.io.FixedLenFeature((), tf.int64),
    'grasps': tf.io.VarLenFeature(dtype=tf.float32)
  }

  parsed_features = tf.io.parse_single_example(example, features)
  w, h = parsed_features['width'], parsed_features['height']
  dim = (h,w,1)

  depth = tf.reshape(parsed_features['depth'], dim)
  grasps = parsed_features['grasps']

  return {'input': depth}, {'quality': depth, 'grasps': grasps, 'angle': depth, 'width': depth}


ds = tf.data.Dataset.from_generator(fake_generator, tf.string)

ds = ds.map(map_func = parse_example)
ds = ds.repeat(100)
ds = ds.batch(10,drop_remainder=True)
it = ds.make_one_shot_iterator()
input_iter, target_iter = it.get_next()

g_dense = tf.reshape(tf.sparse_tensor_to_dense (target_iter['grasps'],-1), (10,-1,4)) 

pos_output = target_iter['quality']
angle = target_iter['angle']
width_output = target_iter['width']

#fake network output
out_shape = tf.shape(pos_output)
b, h, w, d = out_shape[0], out_shape[1], out_shape[2], out_shape[3]


flat_tensor = tf.reshape(pos_output, (b, -1, d))

# argmax of the flat tensor
argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

# convert indexes into 2D coordinates
argmax_x = argmax % w
argmax_y = argmax // w

angle_avg = kl.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(angle)
width_avg = kl.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(width_output)

# stack and return 2D coordinates
grasp_center = tf.stack((argmax_x, argmax_y), axis=1)
grasp_angle = tf.batch_gather(tf.reshape(angle_avg, (b, -1,d)), argmax)
grasp_width = tf.batch_gather(tf.reshape(width_avg, (b, -1,d)), argmax)


detection_grasps = tf.concat([tf.cast(grasp_center,tf.float32), grasp_angle, grasp_width], axis=1)
detection_grasps = tf.reshape(detection_grasps, (b, -1, 4))

groundtruth_grasps = tf.reshape(tf.sparse_tensor_to_dense (target_iter['grasps'], -1), (b, -1, 4))



def update_op_fn(groundtruth_grasps_batched, detection_grasps_batched):
  for (groundtruth_grasps, detection_grasps) in zip(
               groundtruth_grasps_batched, detection_grasps_batched):
    print(detection_grasps.shape)
    if len(detection_grasps.shape) != 2:
        raise ValueError('All entries in detection_grasps expected to be of '
                         'rank 2.')
    if detection_grasps.shape[1] != 4:
        raise ValueError('All entries in detection_grasps should have '
                         'shape[1] == 4.') 

    
	

update_op = tf.py_func(update_op_fn, [groundtruth_grasps, detection_grasps], [])


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

gauss_kernel = gaussian_kernel(5,0.0,1.0)
# Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

blurred_image = input_iter['input']
blurred_image = tf.nn.conv2d(blurred_image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")


sess = tf.Session()
x, y = sess.run(it.get_next())
  
#print(sess.run(detection_grasps))
print(sess.run(update_op))


blurred_image_result = sess.run(blurred_image)

cv2.imwrite('blurred_image_temp.png', np.uint8(255*blurred_image_result[0]))



#print(parsed_features)
