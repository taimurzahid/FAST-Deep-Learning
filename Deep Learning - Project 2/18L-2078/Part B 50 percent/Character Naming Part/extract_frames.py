
import cv2

#storing all names in this list
allNames = [] 
def timeToSeconds (time):
	time = time.split(":");
	outputTime = (int(time[0]) * 3600) + (int(time[1]) * 60) + int(time[2])
	return outputTime

def getCharacterFaceChunks (filename):
	with open(filename) as chunkFile:
		allChunks = []
		for line in chunkFile:
			line = line.split(' ')
			allChunks.append(line[0])
			allNames.append(line[1])
		return allChunks

if __name__ == '__main__':
	output_names_file = "output_names.txt"
	chunks = getCharacterFaceChunks(output_names_file)
	timeInstances = []
	for chunk in chunks:
		timeInstances.append(timeToSeconds(chunk))

	# save only unique chunks
	#timeInstances = list(set(timeInstances))

	for time in timeInstances:
		count = time
		cap = cv2.VideoCapture("new.mp4")
		while count < time + 10:
			cap.set(cv2.CAP_PROP_POS_MSEC,count*1000)      
			ret,frame = cap.read()
			cv2.imwrite("frame%d.jpg" % count, frame)
			count = count + 1
