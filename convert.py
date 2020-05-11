import numpy as np
import os, cv2, gzip

def ubyte2png(root, mode):
    if mode == 'train':
        label_root = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        img_root = os.path.join(root, 'train-images-idx3-ubyte.gz')
    else:
        label_root = os.path.join(root, 't10k-labels-idx1-ubyte.gz')
        img_root = os.path.join(root, 't10k-images-idx3-ubyte.gz')
    
    def my_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    my_mkdir(os.path.join(root, mode))
    
    with gzip.open(label_root, 'rb') as lbpath:
        label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(img_root, 'rb') as imgpath:
        img = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(label), 28, 28)
        
    for i in range(label.min(), label.max()+1):
        my_mkdir(os.path.join(root, mode, str(i)))
        index = np.where(label==i)[0]
        for idx, k in enumerate(index):
            cv2.imwrite(os.path.join(root, mode, str(i), str(idx)+'.png'), img[k])
            
root_list = ['../data/MNIST/raw/', 
'../data/FashionMNIST/raw/', 
'../data/NotMNIST/raw/', 
'../data/EMNIST-letter/raw/']
for root in root_list:
    ubyte2png(root, 'train')
    ubyte2png(root, 'test')