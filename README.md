# darknet-python-flask
Configure the darknet rest api server using the python flask.

## installation
* install darknet
<https://pjreddie.com/darknet/install/> - how to install darknet
* install python packages
<pre><code>
pip install flask tornado Pillow
</code></pre>

## implementation
<pre><code>
git clone https://github.com/haracejacob/darknet-python-flask.git
cd darknet-python-flask
python app.py -c[--cfg] cfg_file_path -w[--weights] weights_file_path -m[--meta] meta_file_path -p[--port] port_number
</code></pre>


## reference
* darknet <https://github.com/pjreddie/darknet>
* caffe examples web-demo <https://github.com/BVLC/caffe/tree/master/examples/web_demo>
