from math import sqrt

import cv2
import numpy as np

from ..data_loader import data_generator, load_single_img, load_single_img2, DataGenerator


def diff(im1, im2):
    return cv2.absdiff(im1, im2)


def preview(generator):
    def collect_points(name, lst, img):
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                lst.append([x, y])
                cv2.circle(img, (x, y), 3, (255, 0, 0))
                cv2.imshow(name, img)
                print([x, y])

        return click_callback

    def square_four_points(pts):
        [x1, y1] = pts[0]
        [x2, y2] = pts[1]

        delta_x = abs(x1 - x2)
        delta_y = abs(y1 - y2)

        if delta_x > delta_y:
            return [[x1, y1], [x1 + delta_x, y1], [x1 + delta_x, y1 + delta_x]]
        else:
            return [[x1, y1 + delta_y], [x1, y1], [x1 + delta_y, y1]]

    def crop(pts, img):

        img_h, img_w = img.shape[0:2]

        x_min = max(0, pts[0][0])
        x_max = min(img_w, pts[1][0])
        y_min = max(0, pts[0][1])
        y_max = min(img_h, pts[1][1])

        return img[y_min + 2:y_max - 2, x_min + 2:x_max - 2]

    img, img_sim, path = load_single_img('/real', '/sim', 641, resize=False)

    def findH(img, img_sim):
        img_pts = []
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', collect_points('img', img_pts, img))
        cv2.waitKey(-1)

        img_pts = square_four_points(img_pts)
        print('img pts', img_pts)

        img_sim_pts = []
        cv2.imshow('img_sim', img_sim)
        cv2.setMouseCallback('img_sim', collect_points('img_sim', img_sim_pts, img_sim))
        cv2.waitKey(-1)

        img_sim_pts = square_four_points(img_sim_pts)
        print('img sim pts', img_sim_pts)

        img_h, img_w, _ = img.shape

        h = cv2.getAffineTransform(np.float32(img_pts), np.float32(img_sim_pts))
        im_dst = cv2.warpAffine(img, h, (img_w, img_h))

        cv2.imshow('distorted', im_dst)
        cv2.waitKey(-1)

        crop_pts = [
            [int(x[0]) for x in h.dot(np.array([[0], [0], [1]])).tolist()],
            [int(x[0]) for x in h.dot(np.array([[img_w], [img_h], [1]])).tolist()]
        ]

        cv2.waitKey(-1)
        background_img = load_single_img2('background.png', resize=False)

        background_img_t = cv2.warpAffine(background_img, h, (img_w, img_h))
        cv2.imshow('bg uncriop img', np.concatenate([background_img_t], axis=1))
        cropped_background_img_t = crop(crop_pts, background_img_t)
        cv2.imshow('bg img', np.concatenate([cropped_background_img_t], axis=1))
        cv2.waitKey(0)

        cv2.imwrite('aligned/background.png', cropped_background_img_t)

        return h, crop_pts

    k = 0

    h, crop_pts = findH(img, img_sim)
    cv2.destroyAllWindows()

    for (real, fake, depth, cls, path) in generator:

        for i in range(len(real)):
            img_h, img_w, _ = real[i].shape

            real_t = cv2.warpAffine(real[i], h, (img_w, img_h))

            cropped_real_t = crop(crop_pts, real_t)
            cropped_fake_t = crop(crop_pts, fake[i])
            cropped_depth_t = crop(crop_pts, depth[i])

            assert cropped_real_t.shape[0:2] == cropped_depth_t.shape[0:2] == cropped_fake_t.shape[0:2]

            cv2.imshow('depth', np.concatenate([cropped_depth_t], axis=1))
            cv2.imshow('frame', np.concatenate([real[i], fake[i], diff(real[i], fake[i]),
                                                real_t, diff(real_t, fake[i])],
                                               axis=1))

            cv2.imshow('cropped', np.concatenate([cropped_real_t, cropped_fake_t,
                                                  diff(cropped_real_t, cropped_fake_t)],
                                                 axis=1))

            np.save('aligned/depth/' + path[i][:-4] + '.npy', cropped_depth_t)
            cv2.imwrite('aligned/real/' + path[i], cropped_real_t)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                exit()
            k += 1
        cv2.destroyAllWindows()

    return h, crop_pts


if __name__ == '__main__':
    generator = DataGenerator(
        'real',
        sim_path='sim',
        depth_path='depth',
        shuffle=False,
        batch_size=99,
        resize=False
    )

    h = preview(generator)
