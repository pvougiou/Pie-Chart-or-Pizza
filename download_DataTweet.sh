#!/bin/bash

# Downloads and uncompresses our data visualisations dataset.
mkdir -p DataTweet
cd DataTweet
wget -O DataTweet.zip https://www.dropbox.com/s/fhx8wasw9s0o7e8/DataTweet.zip?dl=1
unzip -o DataTweet.zip
echo All required files have been downloaded and un-compressed successfully.
