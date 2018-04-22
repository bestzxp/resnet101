import tensorflow as tf
from model import VGGNet
import numpy as np
from data_loader import DataLoader
import time

ckpt_path = '../ckpt/model_0.ckpt'
net = VGGNet([224, 224], 128, training=False)
net.build()

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)



# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7

if ckpt_path:
    saver.restore(sess, ckpt_path)
loader = DataLoader()
batch = 64
valid_batch_num = loader.valid_urls.shape[0] // batch

cou = 1
# for idx in range(valid_batch_num):
#     res = loader.get_valid_batch_data(batch)
#     feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
#     # sess.run(optimizer, feed_dict=feed_dicts)
#     fc_16 = sess.run([net.fc_16], feed_dict=feed_dicts)
#     fc_16 = np.array(fc_16[0])
#     for i in range(batch):
#         if np.argmax(fc_16[i, :]) == np.argmax([res[1][i, :]]):
#             cou += 1
#
# print(cou, cou/loader.valid_urls.shape[0])

file = open('../res_{}.csv'.format(int(time.time())), 'w')
file.write('id,predicted\n')

for i in range(12800):
    if i%1000==0:
        print(i)
    imgs = loader.get_test_image('/home/sun/zxp/vgg_net/data/test/{}.jpg'.format(i + 1))
    if imgs is not None:
        img_num = imgs.shape[0]
        feed_dicts = {net.inputs: imgs, net.ground_truth: np.zeros((1, 128))}
        fc_16 = sess.run(net.fc_16, feed_dict=feed_dicts)
        fc_16 = np.sum(fc_16, axis=0)
        # print(i, np.argmax(fc_16)+1)
        file.write('{},{}\n'.format(i+1, np.argmax(fc_16)+1))
    else:
        file.write('{},{}\n'.format(i + 1, 1))
file.close()


