from vad import VoiceActivityDetector

class findSilentFrames():

    def __init__(self, wave_input_filename, output_File, duration_of_video):    
        v = VoiceActivityDetector(wave_input_filename)
        raw_detection = v.detect_speech()
        speech_labels = v.convert_windows_to_readible_labels(raw_detection)
        speech_labels = [float(i) for i in speech_labels]
        if len(speech_labels) == 0:
            with open(output_File, 'w') as fp:
                fp.write(str(0.00) + " " + str(duration_of_video) + "\n")
                return

        final = []
        if speech_labels[0] != 0.00:
            final.append(0.00)
            final.append(speech_labels[0] - 0.01)
        count = 1
        length = duration_of_video
        last = len(speech_labels) - 1
        while count <= last:
            final.append(speech_labels[count] + 0.01)
            count += 1
            if count >= last:
                final.append(length)
            else:
                final.append(speech_labels[count] - 0.01)
            count += 1

        count = 1
        while count < len(final):
            if final[count] - final[count - 1] < 4:             # 4 seconds is the minimum limit for caption, as sentence can't fit in less than 4 seconds
                del final[count]
                del final[count - 1]
            else:
                count += 2

        if len(final) != 0:
            with open(output_File, 'w') as fp:
                index = 0
                while index < len(final):
                    fp.write(str(final[index]) + " " + str(final[index + 1]) + "\n")
                    index += 2
        