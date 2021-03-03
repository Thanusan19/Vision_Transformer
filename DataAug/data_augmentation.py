from PIL.Image import blend
import cv2
import platform
import matplotlib.pyplot as plt
import random
import numpy as np
import math

diatome_im_size = np.array((256, 256))
vit_im_size = np.array((384, 384))

#im = cv2.imread('/Users/guillaume/Downloads/Image_created_with_a_mobile_phone.png', cv2.COLOR_BGR2RGB)
im = cv2.imread('./AAMB/IDF_AAMB_080019.png', cv2.COLOR_BGR2RGB)

# cv2.namedWindow("im_win", cv2.WINDOW_NORMAL)

# # IF mac OS :
# if platform.system() == 'Darwin':
#     cv2.os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

# cv2.imshow("im_win", im)
# cv2.waitKey(0)

def print_9(imlist):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)

        try:
            im = imlist[i]
        except IndexError:
            pass

        im = cv2.resize(im, (384, 384))

        ax.imshow(im)
        
    plt.show()

# l = []
# for i in range(9):
#     l.append(im)

# print_9(l)

# from https://note.nkmk.me/en/python-numpy-generate-gradation-image/
# but modified, inspired by https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/
def get_sigmoid_gradient_2d(start, stop, width, height, is_horizontal):
    x = np.linspace(start, stop, width)
    # print(x)
    s = 1/(1 + np.exp(-x))

    if is_horizontal:
        res = np.tile(s, (height, 1))
        #print(res)
        return res
    else:
        res = np.tile(s, (width, 1)).T
        #print(res)
        return res

def get_sigmoid_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float32)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_sigmoid_gradient_2d(start, stop, width, height, is_horizontal)

    return result

# test = get_sigmoid_gradient_3d(diatome_im_size[0], diatome_im_size[1],
#                                     start_list=np.ones(3)*-4.0,
#                                     stop_list=np.ones(3)*4.0,
#                                     is_horizontal_list=[True, True, True])

# print(np.round(test[-1, -1, :]*255.0).astype(np.uint8))

# plt.imshow(np.round(test*255.0).astype(np.uint8))
# plt.show()

# inspired from https://stackoverflow.com/a/23316542
def make_augmentation(im, num):
    l = []
    im = cv2.resize(im, tuple(diatome_im_size))

    diatome_center = tuple(diatome_im_size/2)
    vit_center = tuple(vit_im_size/2)
    
    w = diatome_im_size[0]
    h = diatome_im_size[1]

    blend_size = (120, 120)
    total_size = blend_size[0] + blend_size[1]

    #print(im.shape)
    ls = blend_size[0]
    fond_couleur = np.median(im[ls:ls+10, 1:10, :], axis=[0,1])
    # fond_couleur = (0, 255, 0)

    fond = np.zeros(tuple(np.append(diatome_im_size, 3)), np.uint8)
    fond[:,:,:] = fond_couleur
    #print(fond_couleur)

    blank = np.ones(tuple(np.append(diatome_im_size, 3)), np.uint8)

    blend_left = get_sigmoid_gradient_3d(blend_size[0], h,
                                        start_list=np.ones(3)*-10.0,
                                        stop_list=np.ones(3)*10.0,
                                        is_horizontal_list=[True, True, True])

    blend_right = get_sigmoid_gradient_3d(blend_size[1], h,
                                        start_list=np.ones(3)*10.0,
                                        stop_list=np.ones(3)*-10.0,
                                        is_horizontal_list=[True, True, True])

    blend_all = np.concatenate([blend_left, blank[:,:w-total_size,:], blend_right], axis=1)
    # print(blend_all.shape)
    # plt.imshow(np.round(blend_all*255.0).astype(np.uint8))
    # plt.show()

    translate_range = (-100, 100)

    for i in range(num):
        new_image = im

        # # from https://stackoverflow.com/a/12890573
        # fond = np.zeros(tuple(np.append(vit_im_size, 3)), np.uint8)
        # fond[:,:,:] = (255, 0, 0)
        # print(fond[0,0,:])
        #fond_couleur = (243, 243, 243)

        # new_image = cv2.addWeighted(new_image, blend, fond_couleur, (1-blend), 0.0)
        new_image = np.round((new_image/255.0 * blend_all + fond/255.0 * (1 - blend_all))*255.0).astype(np.uint8)

        tx_ty = np.floor(((vit_im_size - diatome_im_size)/2))
        trans_mat = np.column_stack([[1, 0], [0, 1], tx_ty])
        new_image = cv2.warpAffine(new_image, trans_mat, tuple(vit_im_size),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        angle = np.random.rand()*360

        rot_mat = cv2.getRotationMatrix2D(vit_center, angle, 1.0)
        new_image = cv2.warpAffine(new_image, rot_mat, tuple(vit_im_size),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        a, b = translate_range
        rand_tx_ty = (b - a) * np.random.random_sample(2) + a
        rand_trans_mat = np.column_stack([[1, 0], [0, 1], rand_tx_ty])
        new_image = cv2.warpAffine(new_image, rand_trans_mat, tuple(vit_im_size),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        l.append(new_image)

    return l

l = make_augmentation(im, 9)
print_9(l)