# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from tensorflow.python.platform import app, flags
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys
import numpy as np
import math
import os
import os.path
import cv2
import collections
import subprocess
from gtts import gTTS
import moviepy.editor as mp
from pydub import AudioSegment
from detectVoiceInWave import findSilentFrames
import spacy
from nltk import word_tokenize,Text,pos_tag
nlp = spacy.load('en')

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


_IMAGE_SIZE = 224
_FRAME_COUNT = 0
_BATCH_SIZE = 1

def set_frameCount(frameCount):
    global _FRAME_COUNT
    _FRAME_COUNT = frameCount

# Computes the TV-L1 optical flow
def compute_TVL1(video_path):

  TVL1 = DualTVL1()
  TVL1.setScalesNumber(1)
  TVL1.setWarpingsNumber(2)
  cap = cv2.VideoCapture(video_path)

  ret, frame1 = cap.read()

  # Error check - for corrupt file
  if ret:
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
  else:
    return -1

  flow = []
  vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  count = 0
  frameCount = 0
  
  for _ in range(vid_len - 2):

    # reduce frames by 8
    count += 1
    if count % 8 == 0:
      frameCount += 1
      ret, frame2 = cap.read()

      # Error check - for corrupt file
      if ret:
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
      else:
        return -1

      curr_flow = TVL1.calc(prev, curr, None)
      assert(curr_flow.dtype == np.float32)

      # truncate pixel values to [-20, 20]
      curr_flow[curr_flow >= 20] = 20
      curr_flow[curr_flow <= -20] = -20

      # rescale between [-1, 1]
      max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
      curr_flow = curr_flow / max_val(curr_flow)
      flow.append(curr_flow)  
      prev = curr

  set_frameCount(frameCount)
  cap.release()
  flow = np.array(flow)
  flow = flow.reshape(_BATCH_SIZE, _FRAME_COUNT, _IMAGE_SIZE, _IMAGE_SIZE, 2)
  return flow

# Process File - for Action Recognition
def _process_video_files(filename, mainDirectory):
  flow = compute_TVL1(filename)

  # Error Check
  if type(flow) == int:
    return -1

  save_name = os.path.join(mainDirectory, 'portion.npy')
  np.save(save_name, flow)
  sys.stdout.flush()
  return 1

# main function
def main(_):

    # File Names and Paths
    silentFramesFile = "/Users/fahadzafar/Documents/University/FYP/im2txt/silenceFrames.txt"
    actionRecognitionFile = "/Users/fahadzafar/Documents/University/FYP/im2txt/action.txt"
    portionFile = "/Users/fahadzafar/Documents/University/FYP/im2txt/portion.mp4"
    portionNPYFile = "/Users/fahadzafar/Documents/University/FYP/im2txt/portion.npy"
    mainDirectory = "/Users/fahadzafar/Documents/University/FYP/im2txt/"
    outputFile = "/Users/fahadzafar/Desktop/Output.mp4"

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    # voice Google or Espeak - Google works with internet, Espeak works without internet
    Google = True

    if not os.path.isfile("silenceFrames.txt"):

        # converting mp4 to wav so that silent frames can be generated
        clip = mp.VideoFileClip(FLAGS.input_files)
        clip.audio.write_audiofile("audio.wav")

        # getting silent frames
        findSilentFrames("audio.wav", silentFramesFile, clip.duration)
        os.remove("audio.wav")

    # naming file
    namesFile = FLAGS.input_files.replace(".mp4", ".txt")
    allNamesList = {}
    if os.path.isfile(namesFile):
        namesUnordered = {}                 # holds name but with unordered time

        # reading names file
        with open(namesFile) as f:
            data = f.readlines()

        # creating a dictionary of names [key: time, value: name]
        for line in data:
            line = line.split()
            time = line[0]
            line.pop(0)
            namesUnordered[time] = ' '.join(line)
        allNamesList = collections.OrderedDict(sorted(namesUnordered.items()))

    # reading file of all silence frames in "silenceFrames"
    if os.path.exists("silenceFrames.txt"):
        with open("silenceFrames.txt") as f:
            silenceFrames = f.readlines()
    else:
        silenceFrames = []

    # check for whether or not first iteration (when not 1st: input video file is Output.mp4)
    firstIteration = True
    tempVideosCount = 1
    inputVideoTitle = FLAGS.input_files
    if len(silenceFrames) > 1:
        outputVideoTitle = "temp%d.mp4" % tempVideosCount
    else:
        outputVideoTitle = outputFile

    for silence in silenceFrames:

        filenames = []
        for file_pattern in FLAGS.input_files.split(","):
            silence = silence.split()

            # check to see if name(s) given or not [False: when not given, True: when given]
            nameCheck = False

            if len(allNamesList) > 0:
                for key, value in allNamesList.iteritems():
                    if float(key) >= float(silence[0]) and float(key) <= float(silence[1]):
                        nameCheck = True
                        characterName = value
                        break

            vidcap = cv2.VideoCapture(file_pattern)
            success, image = vidcap.read()
            count = 1
            initialFrame = int(math.ceil(float(silence[0])) * 1000)         # starting frame
            finalFrame = int(math.floor(float(silence[1])) * 1000)          # ending frame
            currentFrame = initialFrame
            success = True
            while success:
                vidcap.set(0, currentFrame)
                success, image = vidcap.read()
                if success == False:
                    break
                cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
                x = mainDirectory + ("frame%d.jpg" % count)
                filenames.extend(tf.gfile.Glob(x))

                if currentFrame == finalFrame:
                    break

                currentFrame += 1000            # 1 fps (reduce by half to generate 2 fps)
                count += 1

            tf.logging.info("Running caption generation on %d files matching %s",
                            len(filenames), FLAGS.input_files)

        with tf.Session(graph=g) as sess:

            # Load the model from checkpoint.
            restore_fn(sess)

            # Prepare the caption generator. Here we are implicitly using the default
            # beam search parameters. See caption_generator.py for a description of the
            # available beam search parameters.
            generator = caption_generator.CaptionGenerator(model, vocab)

            # dictionary to hold captions and probabilities
            captionsList = dict()

            for filename in filenames:
                with tf.gfile.GFile(filename, "r") as f:
                    image = f.read()
                captions = generator.beam_search(sess, image)
                print("Captions for image %s:" % os.path.basename(filename))
                for i, caption in enumerate(captions):
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    if sentence not in captionsList:
                        captionsList[sentence] = set()
                    captionsList[sentence].add(math.exp(caption.logprob))

            for file in os.listdir(mainDirectory):
                if file.endswith('.jpg'):
                    os.remove(file)

            finalCaptions = {}
            for key, value in captionsList.items():
                finalCaptions[sum(sorted(value)) / float(len(sorted(value)))] = key     # average probability of each caption is selected as key, sentence as value

            # just take out the top 1 caption - based on Probability
            od = collections.OrderedDict(sorted(finalCaptions.items(), reverse=True)[:1])

            text = ""
            for key, value in od.items():
                text = text + value

            # text is a list, converting it to str
            output = ''.join(text)

            # runs only when name is given
            if nameCheck:

                # converting str to unicode format
                parsed_text = nlp(output.decode('utf-8'))       
                for nc in parsed_text.noun_chunks:
                    text = nc.text
                    output = output.replace(text, characterName, 1)   # text (extracted subject here) && 1: only first occurence is modified
                    break


            # extract the chunk - whose action needs to be found
            ffmpeg_extract_subclip(FLAGS.input_files, float(silence[0]), float(silence[1]), targetname="portion.mp4")
            
            # run Action Recognition here
            probability = 0
            if _process_video_files(portionFile, mainDirectory) == 1:

                # after creation of .npy file, run the evaluation file
                subprocess.call(['python3', 'actionRecognition.py', str(_FRAME_COUNT)])

                # read the action
                actionFile = open(actionRecognitionFile, "r")  
                action = actionFile.readlines()
                action = action[0].split()
                probability = float(action[0])
                action = action[1]

            # put action into the sentence only if it's probability is greater than 0.75
            if probability >= 0.75:

                # tokenize words - replace the verb with action
                tokens = word_tokenize(output)
                text = Text(tokens)
                tags = pos_tag(text)
                newSentence = []

                # VBN and VBG - represents different forms of Verb, so that they can be replaced by Action
                for val in tags:
                    if "VBN" in val[1] or "VBG" in val[1]:
                        newSentence.append(action)
                    else:
                        newSentence.append(val[0])

                finalSentence = " ".join(newSentence)
                output = finalSentence

            print("\n" + output + "\n")

            # embedding audio in original
            if Google:
                myobj = gTTS(text=output, lang='en-us', slow=False)
                myobj.save("caption.mp3")
                sound2 = mp.AudioFileClip('caption.mp3')
                sound2.write_audiofile("caption.wav")
            else:
                subprocess.call(["espeak", "-w" + 'caption.wav', output])

            # extracting wav from video
            clip = mp.VideoFileClip(inputVideoTitle)

            # if video has no audio
            if clip.audio is None:

                # adding modified wav to mp4
                video = mp.VideoFileClip(inputVideoTitle)
                video = video.set_audio(mp.AudioFileClip("caption.wav"))
                video.write_videofile(outputVideoTitle)

            else:

                clip.audio.write_audiofile("original.wav")          # extract original mp4 soundtrack
                sound1 = AudioSegment.from_file('original.wav')     # make it wav
                sound2 = AudioSegment.from_file('caption.wav')      # get audio with caption
                sound_with_wave = sound1.overlay(sound2, position=(float(silence[0]))*1000)        # position
                sound_with_wave.export('overlaid.wav', format='wav')

                # adding modified wav to mp4
                video = mp.VideoFileClip(inputVideoTitle)
                video = video.set_audio(mp.AudioFileClip("overlaid.wav"))
                video.write_videofile(outputVideoTitle)

                os.remove("original.wav")
                os.remove("overlaid.wav")

            if Google:
                os.remove("caption.mp3")
            os.remove("caption.wav")

            # updating input and output file names
            inputVideoTitle = "temp%d.mp4" % tempVideosCount
            tempVideosCount += 1

            if tempVideosCount == len(silenceFrames):
                outputVideoTitle = outputFile
            else:
                outputVideoTitle = "temp%d.mp4" % tempVideosCount

    # remove silenceFrames file only if it exists        
    if len(silenceFrames) != 0:
        os.remove(silentFramesFile)

    # remove action.txt file
    if os.path.exists(str(actionRecognitionFile)):
        os.remove(str(actionRecognitionFile))

    # remove portion.mp4 file
    if os.path.exists(str(portionFile)):
        os.remove(str(portionFile))

    # remove portion.npy file
    if os.path.exists(str(portionNPYFile)):
        os.remove(str(portionNPYFile))

    # remove all "temp" mp4 files
    if len(silenceFrames) > 1:
        for file in os.listdir(mainDirectory):
            if file.endswith('.mp4'):
                os.remove(file)


if __name__ == "__main__":
    tf.app.run()
