import glob
all_images = glob.glob("./image/*")
all_annotations = glob.glob("./labels/*")

y = map(lambda x : x[8:-4], all_images) + map(lambda x : x[9:-4], all_annotations)

fh = open('list.txt', 'w')
fh.write('\n'.join(set(y)))
