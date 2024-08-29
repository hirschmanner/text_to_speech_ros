#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String, Bool
import wave
import scipy
import numpy as np

from io import BytesIO

import torch
from TTS.api import TTS
import nltk


# Get device
#device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
#print(TTS().list_models())

# Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# # Run TTS
# # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# # Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# # Text to speech to a file
# tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")






class AudioStreamer:
    def __init__(self, tts_model = "tts_models/de/thorsten/vits", audio_pub_topic='/audio_topic', text_sub_topic = '/tts_request'):
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.framewidth = 1024
        # This message contains the audio meta data

        # Number of channels
        self.audio_channels = 1
        # Sampling rate [Hz]
        self.sample_rate = 22050
        # Audio format (e.g. S16LE)
        self.sample_format = "S16LE"
        # Amount of audio data per second [bits/s]
        self.bitrate = 320
        # Audio coding format (e.g. WAVE, MP3)
        self.coding_format = "wav"

        # NLTK Tokenizer to split sentences
        self.tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')

        # Init TTS
        self.tts = TTS(tts_model).to(device)
        self.audio_pub = rospy.Publisher(f"{audio_pub_topic}/audio", AudioData, queue_size=100)
        self.robot_is_speaking = rospy.Publisher(f"{audio_pub_topic}/active_trigger", Bool, queue_size=1)
        self.tts_sub = rospy.Subscriber(text_sub_topic, String, self.tts_callback)


    def stream_wav_file(self, file_path):
        with wave.open(file_path, 'rb') as wav_file:
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()

            rospy.loginfo("Streaming WAV file: {}".format(self.file_path))

            while not rospy.is_shutdown():
                raw_data = wav_file.readframes(1024)
                if not raw_data:
                    break

                audio_msg = AudioData()
                audio_msg.data = raw_data
                audio_msg.format = "wav"
                audio_msg.sample_rate = frame_rate
                audio_msg.channels = channels
                audio_msg.sample_width = sample_width

                self.audio_pub.publish(audio_msg)
                rospy.sleep(1.0 / frame_rate)

    def tts_callback(self, msg):
        sentence_list = self.tokenizer.tokenize(msg.data)
        for sentence in sentence_list:
            if 'multilingual' in self.tts.model_name:
                wav = np.array(self.tts.tts(text=sentence, 
                                            speaker_wav='/catkin_ws/src/coqui_tts/spongebob_short.wav', 
                                            language='de'))
            else:
                wav = np.array(self.tts.tts(text=sentence))
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

            wav_norm = wav_norm.astype(np.int16)

            #sample_width = wav.getsampwidth()
            #sample_rate = 16000
            #channels = 2
            #wav_buffer = BytesIO()
            #scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
            #wav_buffer.seek(0)
            audio_msg = AudioData()
            rate = rospy.Rate(self.framewidth//2)
            rate = rospy.Rate(self.sample_rate//self.framewidth)
            self.robot_is_speaking.publish(Bool(True))
            for i in range(int(len(wav_norm)//self.framewidth)):
                audio_msg.data = wav_norm[i*self.framewidth:i*self.framewidth+self.framewidth].tobytes()
                #audio_msg.format = "wav"
                #audio_msg.sample_rate = sample_rate
                #audio_msg.channels = channels
                #audio_msg.sample_width = sample_width
                self.audio_pub.publish(audio_msg)
                rate.sleep()
            rospy.sleep(0.5)
            self.robot_is_speaking.publish(Bool(False))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('audio_streamer', anonymous=True)
    audio_streamer = AudioStreamer(tts_model = "tts_models/de/thorsten/vits",#tts_models/multilingual/multi-dataset/xtts_v2", # "tts_models/de/thorsten/vits"
                                   audio_pub_topic='/audio_tts', 
                                   text_sub_topic = '/tts_request')
    try: 
        audio_streamer.run()
    except rospy.ROSInterruptException:
        pass
