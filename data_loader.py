import os
import numpy as np
import cv2

classes = range(1, 129)

class DataLoader(object):
    def __init__(self):
        self.classes = classes
        self.images_urls, self.labels = self.load_all_data()
        self.valid_urls, self.valid_labels = self.load_all_data(type='valid')
        self.cursor = 0
        self.avg = np.array([[[158.14044652223768, 166.1790830525745, 176.01642383807538]]])

        self.valid_cursor = 0


    def load_all_data(self, type='train'):
        img_path_urls = []
        label = []
        for i in range(len(classes)):
            images = os.listdir('/home/sun/zxp/vgg_net/data/{}/{}/'.format(type, classes[i]))
            images = ['/home/sun/zxp/vgg_net/data/{}/{}/'.format(type, classes[i])
                                  + image for image in images]
            img_path_urls.extend(images)
            label.extend([i] * len(images))
            print('{}: {} has {} images'.format(type, classes[i], len(images)), '===', i, classes[i])
            # print('class {} has {} images'.format(classes[i], len(images)))
        print('\ntotal images num: {}'.format(len(img_path_urls)))
        img_path_urls = np.array(img_path_urls)
        label = np.array(label)
        perm = np.arange(len(img_path_urls))
        np.random.shuffle(perm)
        return img_path_urls[perm], label[perm]

    def get_batch_data(self, batch_size):
        images = np.zeros([batch_size, 224, 224, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.cursor < self.images_urls.shape[0]:
                images[i, :] = (self.get_image(self.images_urls[self.cursor]) - self.avg)
                labels[i, :] = self.one_hot(self.labels[self.cursor], len(classes))
                self.cursor += 1
        return images, labels

    def get_valid_batch_data(self, batch_size):
        images = np.zeros([batch_size, 224, 224, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.valid_cursor < self.valid_urls.shape[0]:
                images[i, :] = (self.get_image(self.valid_urls[self.valid_cursor]) - self.avg)
                labels[i, :] = self.one_hot(self.valid_labels[self.valid_cursor], len(classes))
                self.valid_cursor += 1
        return images, labels


    def shuffle(self):
        perm = np.arange(len(self.images_urls))
        np.random.shuffle(perm)
        self.images_urls = self.images_urls[perm]
        self.labels = self.labels[perm]
        self.cursor = 0

    def get_image(self, image_url):
        img = cv2.imread(image_url)
        img = cv2.resize(img, (224, 224))
        return img

    def one_hot(self, i, num_classes):
        label = np.zeros((1, num_classes))
        label[0, i] = 1
        return label

    def get_test_image(self, url):
        if os.path.exists(url):
            img = cv2.imread(url)
            return self.do_change(img)
        else:
            return None

    def do_change(self, img):
        res = []
        width, height, _ = img.shape
        res.append(cv2.resize(img, (224, 224)) - self.avg)
        if width<50 or height<50:
            return None
        if width<100 or height<100:
            return np.array(res)
        im1 = img[width//8:, height//8:, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[width // 4:, height // 4:, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[width // 4:, :, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[:, height // 4:, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[:width - width // 8:, :height - height // 8, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[width // 8: width - width // 8, height // 8: height - height // 8, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        im1 = img[width // 4:width - width // 4:, :, :]
        res.append(cv2.resize(im1, (224, 224)) - self.avg)

        return np.array(res)



    # def preprocess(self, image):

if __name__ == '__main__':
    loader = DataLoader()
    loader.get_test_image('/home/sun/Disk/funi/data/test/1.jpg')
    # loader.shuffle()
    # R, G, B = [0, 0, 0]
    # for i in range(len(loader.images_urls)):
    #     if i % 100 == 0:
    #         print(i)
    #     image = loader.get_image(loader.images_urls[i])
    #     B += (np.sum(image[:, :, 0]) / (224 * 224))
    #     G += (np.sum(image[:, :, 1]) / (224 * 224))
    #     R += (np.sum(image[:, :, 2]) / (224 * 224))
    # B /= len(loader.images_urls)
    # G /= len(loader.images_urls)
    # R /= len(loader.images_urls)
    # print(B, G, R)
