FROM tensorflow/tensorflow:1.8.0-devel-py3

RUN cd /tensorflow && git pull origin master && git checkout 6d1603622b1c3b25de0a8d342714fed271308a47
RUN bazel build -c opt --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo

