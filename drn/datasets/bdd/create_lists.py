import os
train_images_dir = 'images/train'
train_labels_dir = 'labels/train'
val_images_dir = 'images/val'
val_labels_dir = 'labels/val'
test_images_dir = 'images/test'

f = open('train_images.txt', 'w')
for fn in sorted(os.listdir(train_images_dir)):
    f.write(train_images_dir + '/' + fn + '\n')
f.close()

f = open('train_labels.txt', 'w')
for fn in sorted(os.listdir(train_labels_dir)):
    f.write(train_labels_dir + '/' + fn + '\n')
f.close()

f = open('val_images.txt', 'w')
for fn in sorted(os.listdir(val_images_dir)):
    f.write(val_images_dir + '/' + fn + '\n')
f.close()

f = open('val_labels.txt', 'w')
for fn in sorted(os.listdir(val_labels_dir)):
    f.write(val_labels_dir + '/' + fn + '\n')
f.close()

f = open('test_images.txt', 'w')
for fn in sorted(os.listdir(test_images_dir)):
    f.write(test_images_dir + '/' + fn + '\n')
f.close()
