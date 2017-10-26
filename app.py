# -*- coding: utf-8 -*-

import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
#import pandas as pd
from flask import jsonify
from PIL import Image
import cStringIO as StringIO
import urllib
#import exifutil
import cv2

import darknet as dn

REPO_DIRNAME = os.path.abspath(os.path.dirname('./'))
UPLOAD_FOLDER = '/tmp/darknet_flask'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        string_buffer.save(filename)
        logging.info('Saving to %s.', filename)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(filename)
    return flask.render_template(
        'index.html', has_result=True, result=result, 
        imagesrc=embed_image_html(filename)
    )


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    result = app.clf.classify_image(filename)
    
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(filename)
    )


@app.route('/classify_rest', methods=['POST'])
def classify_rest():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return jsonify(val = 'Cannot open uploaded image.')

    result = app.clf.classify_image(filename)

    return jsonify(	val = result)

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    #image = image[:,:,(2,1,0)]
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

class ImagenetDetector(object) :
    def __init__(self, cfg_file, weights_file, meta_file) :
        self.net = dn.load_net(cfg_file, weights_file, 0)
        self.meta = dn.load_meta(meta_file)
        
    def classify_image(self, image) :
        res = dn.detect(self.net, self.meta, image)
        return res

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-c', '--cfg',
        help="choose cfg file",
        action='store', default="cfg/tiny-yolo.cfg")
    parser.add_option(
        '-w', '--weights',
        help="choose weights file",
        action='store', default="tiny-yolo.weights")
    parser.add_option(
        '-m', '--meta',
        help="choose meta file",
        action='store', default="cfg/coco.data")

    # Initialize classifier + warm start by forward for allocation
    opts, args = parser.parse_args()
    app.clf = ImageDetector(opts.cfg, opts.weights, opts.meta)
    
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)