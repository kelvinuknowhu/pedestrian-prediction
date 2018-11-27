import os
train_images_dir = 'training/images'
train_labels_dir = 'training/train_ids'
val_images_dir = 'validation/images'
val_labels_dir = 'validation/train_ids'
test_images_dir = 'testing/images'

f = open('train_images.txt', 'w')
for fn in os.listdir(train_images_dir):
    f.write(train_images_dir + '/' + fn + '\n')
f.close()

f = open('train_labels.txt', 'w')
for fn in os.listdir(train_labels_dir):
    f.write(train_labels_dir + '/' + fn + '\n')
f.close()

f = open('val_images.txt', 'w')
for fn in os.listdir(val_images_dir):
    f.write(val_images_dir + '/' + fn + '\n')
f.close()

f = open('val_labels.txt', 'w')
for fn in os.listdir(val_labels_dir):
    f.write(val_labels_dir + '/' + fn + '\n')
f.close()

f = open('test_images.txt', 'w')
for fn in os.listdir(test_images_dir):
    f.write(test_images_dir + '/' + fn + '\n')
f.close()
