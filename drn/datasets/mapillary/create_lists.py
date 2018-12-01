import os
train_images_dir = 'training/images'
train_labels_dir = 'training/train_ids'
val_images_dir = 'validation/images'
val_labels_dir = 'validation/train_ids'
test_images_dir = 'testing/images'

f_labels = open('train_labels.txt', 'w')
f_images = open('train_images.txt', 'w')
for fn in sorted(os.listdir(train_labels_dir)):
    f_labels.write(train_labels_dir + '/' + fn + '\n')
    f_images.write(train_images_dir + '/' + fn.split('.')[0] + '.jpg' + '\n')
f_labels.close()
f_images.close()

f_labels = open('val_labels.txt', 'w')
f_images = open('val_images.txt', 'w')
for fn in sorted(os.listdir(val_labels_dir)):
    f_labels.write(val_labels_dir + '/' + fn + '\n')
    f_images.write(val_images_dir + '/' + fn.split('.')[0] + '.jpg' + '\n')
f_labels.close()
f_images.close()

f = open('test_images.txt', 'w')
for fn in sorted(os.listdir(test_images_dir)):
    f.write(test_images_dir + '/' + fn + '\n')
f.close()