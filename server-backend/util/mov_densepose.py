import os

dense_dir = '/home/yaosy/Diskb/bdmeet/yaosy/mowei_denseposePre'
dest_dir = '/home/yaosy/Diskb/bdmeet/yaosy/mowei_densepose'
dense_filenames = sorted(os.listdir(dense_dir))
for i, filename in enumerate(dense_filenames):
    # if i > 10:
    #     break
    if 'IUV' not in filename:
        continue
    print(filename)
    os.rename(os.path.join(dense_dir, filename), os.path.join(dest_dir, filename))
