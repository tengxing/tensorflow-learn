from queue import Queue
from threading import Thread
import json
import os
import requests
import shutil


## Directory where all photos will be downloaded
BASE_IMG_DIRECTORY = '/home/tengxing/data/imgs'


"""
    Download an image to BASE_IMG_DIRECTORY
"""
def download_image(name, url):
    with requests.Session() as request_session:
        request_session.get(url, hooks=dict(response=get_response_hook(name)), stream=True)

log_file = open('download.logs', 'a')
def log(text):
    log_file.write(text + '\n')

def get_response_hook(file_name):
    def print_url(r, *args, **kwargs):
        if r.status_code == 200:
            with open(get_image_path(file_name), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        else:
            print('Error in download of: {}'.format(file_name))
            log('Error in download of: {}'.format(file_name))
    return print_url

"""
    Read photos.txt file.
    This file contains a map of <id>,<url> of all the photos of the dataset.
    return: Dictionary of {name:url}
"""
def read_photos():
    photos = {}
    with open('photos/photos.txt') as f:
        for line in f:
            [name,url] = line.split(',', 1)
            url = url.strip('\n')
            name = int(name)
            photos[name] = url

    return photos

photos_queue = Queue()

"""
    Compose path to a photo given its name.
"""
def get_image_path(name):
    return '{}/{}.jpeg'.format(BASE_IMG_DIRECTORY, name)

"""
    Check whether a photo is already downloaded
"""
def is_photo_dowloaded(name):
    return os.path.isfile(get_image_path(name))

"""
    Start download of all not downloaded photos.
"""
def start_async_download():
    #Set to download all photos if haven't been download yet
    photos = read_photos()
    for name, url in photos.items():
        if not is_photo_dowloaded(name):
            photos_queue.put((name, url))

    #Each worker that consume items from photos_queue
    def worker():
        while not photos_queue.empty():
            name, url = photos_queue.get()
            #proccess item
            download_image(name, url)
            photos_queue.task_done()

    print('Set to download {} photos'.format(photos_queue.qsize()))

    # Start each worker in a diferent thread
    for i in range(NUM_WORKER_THREAD):
        t = Thread(target = worker)
        t.daemon = True
        t.start()

    photos_queue.join()
    log_file.close()
    print("All tasks completed")


if __name__ == '__main__':
    NUM_WORKER_THREAD = 12
    start_async_download()
