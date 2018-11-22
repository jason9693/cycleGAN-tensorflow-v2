import util.processing as processing
import os
import numpy as np
import re
import random
import matplotlib.pyplot as plt

def get_datasets(dir_name, resizing=None):
    files = [dir_name+'/'+png for png in os.listdir(dir_name) if not os.path.isdir(dir_name+'/'+png)]
    print(files)
    datas = []
    shapes = []
    for file in files:
        data, shape = processing.load_image(file, meta=True)
        if resizing is not None:
            data = processing.imresize(data, (resizing, resizing), preserve_range=True)
        datas.append(data)
        shapes.append(shape)

    return (datas, shapes)
    #np.array([{'data':i, 'shape':n} for i,n in [processing.load_image(file, meta=True) for file in files]])
    #return [dir+file_name for file_name in os.listdir(dir)]

def get_file_names(dir_name):
    files = [dir_name + '/' + png for png in os.listdir(dir_name) if not os.path.isdir(dir_name + '/' + png)]
    return files

#def get_datasets_with_file_name

class DataIO:
    def __init__(self, dir_name):
        self.files = [dir_name + '/' + png for png in os.listdir(dir_name) if not os.path.isdir(dir_name + '/' + png)]
        self.idx = 0


    def count_files(self):
        return len(self.files)

    def get_datasets(self, resizing=None):
        datas = []
        shapes = []
        for file in self.files:
            data, shape = processing.load_image(file, meta=True)
            if resizing is not None:
                data = processing.imresize(data, (resizing, resizing))
            datas.append(data)
            shapes.append(shape)

        return (datas, shapes)

    def get_shuffle_cropped_data_batch(self, batch_size, resizing=None):
        datas = []
        shapes = []

        # files_batch = random.sample(self.files, batch_size)
        if self.idx + batch_size > len(self.files):
            start_idx = -batch_size + len(self.files)
            last_idx = len(self.files)
            self.idx = 0

        else:
            start_idx = self.idx
            last_idx = self.idx + batch_size
            self.idx += batch_size

        for file in self.files[start_idx: last_idx]:
            data, shape = processing.load_image(file, meta=True)
            if resizing is not None:
                if resizing > min(shape[0], shape[1]):
                    return self.get_data_batch(batch_size, resizing)
                rand_w = random.randint(0, shape[0]-resizing)
                rand_h = random.randint(0, shape[1]-resizing)
                data = data[rand_w:rand_w+resizing, rand_h:rand_h+resizing, :]
            # data = processing.preprocess_image(data)
            datas.append(data)
            shapes.append((resizing, resizing))

        return (datas, shapes)

    def get_data_batch(self, batch_size, resizing=None):
        datas = []
        shapes = []

        #files_batch = random.sample(self.files, batch_size)
        if self.idx + batch_size > len(self.files):
            start_idx = -batch_size + len(self.files)
            last_idx = len(self.files)
            self.idx = 0

        else:
            start_idx = self.idx
            last_idx = self.idx + batch_size
            self.idx += batch_size

        for file in self.files[start_idx: last_idx]:
            data, shape = processing.load_image(file, meta=True)
            if resizing is not None:
                data = processing.imresize(data, (resizing, resizing))
            # data = processing.preprocess_image(data)
            datas.append(data)
            shapes.append(shape)

        return (datas, shapes)

    def get_rand_data_batch(self, batch_size, resizing=None):
        datas = []
        shapes = []
        files_batch = random.sample(self.files, batch_size)
        for file in files_batch:
            data, shape = processing.load_image(file, meta=True)
            if resizing is not None:
                data = processing.imresize(data, (resizing, resizing))
            #data = processing.preprocess_image(data)
            datas.append(data)
            shapes.append(shape)

        return (datas, shapes, files_batch)