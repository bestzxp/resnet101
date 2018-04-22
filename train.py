import tensorflow as tf
import sys
from resnet import ResNet
from data_loader import DataLoader
import numpy as np
# 0.01 in ckpt_1


learning_rate=0.001

net = ResNet([224, 224], 128)
net.build()
loss = net.loss()
# print(tf.global_variables())
ckpt_path = None # '../ckpt/model_0.ckpt'

loader = DataLoader()

sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

ls = tf.summary.scalar('loss', loss)

train_writer = tf.summary.FileWriter('./log_train', sess.graph)
valid_writer = tf.summary.FileWriter('./log_valid', sess.graph)

batch = 16
batch_num = loader.images_urls.shape[0] // batch
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
valid_batch_num = loader.valid_urls.shape[0] // batch

if ckpt_path is not None:
    saver.restore(sess, ckpt_path)
else:
    sess.run(tf.global_variables_initializer())

global_step = 0
valid_step = 0
for i in range(100000):

    # train
    cou = 0
    total_loss = 0
    for idx in range(batch_num):
        true_num = 0
        global_step += 1
        res = loader.get_batch_data(batch)
        feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
        # sess.run(optimizer, feed_dict=feed_dicts)
        _, ls_, l, fc_16 = sess.run([optimizer, ls, loss, net.result], feed_dict=feed_dicts)
        total_loss+=l
        for j in range(batch):
            if np.argmax(fc_16[j, :]) == np.argmax([res[1][j, :]]):
                cou += 1
                true_num += 1
        train_writer.add_summary(ls_, global_step)
        sys.stdout.write("\r-train epoch:%3d, idx:%4d, loss: %10.6f true_num: %2d / %2d" % (i, idx, l, true_num, batch))
    print("\nepoch:{}, train avg_loss:{}".format(i, total_loss/batch_num))
    saver.save(sess, './ckpt/model_{}.ckpt'.format(i))

    loader.shuffle()

    # test
    cou = 0
    total_loss = 0
    for idx in range(valid_batch_num):
        valid_step += 1
        true_num = 0
        res = loader.get_valid_batch_data(batch)
        feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
        # sess.run(optimizer, feed_dict=feed_dicts)
        ls_, l, fc_16 = sess.run([ls, loss, net.result], feed_dict=feed_dicts)
        # fc_16 = sess.run([net.fc_16], feed_dict=feed_dicts)
        # fc_16 = np.array(fc_16[0])
        for j in range(batch):
            # print(i, fc_16.shape, res[1].shape)
            if np.argmax(fc_16[j, :]) == np.argmax([res[1][j, :]]):
                cou+=1
                true_num+=1
        total_loss += l
        valid_writer.add_summary(ls_, valid_step)
        sys.stdout.write("\r-valid epoch:%3d, idx:%4d, loss: %0.6f true_num: %2d / %2d" % (i, idx, l, true_num, batch))
    loader.valid_cursor = 0
    print("\nepoch:{}, valid avg_loss:{}, total_true_num:{}, accuarcy:{}".format(i, total_loss / valid_batch_num, cou, cou /loader.valid_urls.shape[0]))

