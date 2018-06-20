FROM tensorflow/tensorflow:1.8.0-devel-py3

# Thanks to https://brightinventions.pl/blog/dockerizing-android-builds/, https://medium.com/@daj/docker-image-for-building-tensorflow-android-demo-app-97c98ce37d9e

RUN cd /tensorflow && git pull origin master && git checkout 6d1603622b1c3b25de0a8d342714fed271308a47
COPY android-ndk-r10e-linux-x86_64.zip /
# android-sdk_r24.2-linux.tgz /
# RUN cd / && unzip ./android-ndk-r10e-linux-x86_64.zip && tar -xzf ./android-sdk_r24.2-linux.tgz
RUN cd / && unzip ./android-ndk-r10e-linux-x86_64.zip
WORKDIR /tensorflow
RUN bazel version
# ENV ANDROID_NDK_HOME=/android-ndk-r10e ANDROID_SDK_HOME=/android-sdk-linux
# RUN cd /android-sdk-linux/tools && ./android update sdk --no-ui
# RUN cd /android-sdk-linux/tools && ./android update sdk --all --no-ui --filter $(seq -s, 27)
COPY sdk-tools-linux-4333796.zip /
RUN unzip /sdk-tools-linux-4333796.zip -d /android
RUN cd /android && yes | ./tools/bin/sdkmanager --licenses
RUN cd /android && ./tools/bin/sdkmanager "platforms;android-24" "platform-tools" "build-tools;26.0.1"
COPY WORKSPACE /tensorflow/WORKSPACE
RUN bazel build -c opt --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo

