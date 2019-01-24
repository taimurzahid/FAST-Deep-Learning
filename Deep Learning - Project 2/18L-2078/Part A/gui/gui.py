import os
from tkinter import *
from tkinter import filedialog as tkFileDialog
from PIL import ImageTk, Image
import webbrowser

class applicationGui:
	def __init__(self,master):
		master.title("Automatic Audio Description of Movies")
		master.resizable(False, False)
		w, h = master.winfo_screenwidth(), master.winfo_screenheight()
		w = w/2
		h = h/2
		master.geometry("%dx%d+%d+%d" % (w, h,w/2,h/2))
		self.path = "final.png"

		self.img = Image.open(self.path)
		self.resized = self.img.resize((int(w), int(h) ))
		self.new_img = ImageTk.PhotoImage(self.resized)

		self.background_label = Label(master, image=self.new_img)
		self.background_label.pack(side = "bottom", fill = "both", expand = "yes")

		self.entryVideo = Entry(self.background_label,width=40, highlightthickness=1, highlightbackground='#27408b',fg='#27408b')

		self.text = Label(self.background_label,text="Get Movie's Audio Description!",bg="white",fg='#27408b', font=('Times New Roman', 26, 'bold'))
		self.text.place(relx=.5, rely=.1, anchor="center")

		self.blind_img = Image.open("smallblind.png")
		self.blind_img2 = ImageTk.PhotoImage(self.blind_img)
	
		self.movie_img = Image.open("6.png")
		self.movie_img2 = ImageTk.PhotoImage(self.movie_img)

		#self.blind_label = Label(self.background_label,image=self.blind_img2,bg='white')
		#self.blind_label.place(relx=.08, rely=.76, anchor="center")
	
		#self.mov_label = Label(self.background_label,image=self.movie_img2,bg='white')
		#self.mov_label.place(relx=.91, rely=.79, anchor="center")


		#self.bottomBar = Label(self.background_label,bg='#27408b',width=100, height=2)
		#self.bottomBar.place(relx= -.1,rely=.99,anchor="w")

		self.labelVideo = Label(self.background_label, text="Enter the video path or browse the video: ",bg="white",fg='#27408b', font=('Times New Roman', 14))
		self.labelVideo.place(relx=.1, rely=.38, anchor="w")

		self.entryVideo.place(relx=.1, rely=.45, anchor="w")

		self.BrowseVideo = Button(self.background_label, text="...", font=('Times New Roman', 12, 'bold'),height=1, command= self.BrowseVideoButtonPressed,fg='#27408b',bg="white",activebackground="white",activeforeground='#27408b', highlightthickness=1, highlightbackground='#27408b')
		self.BrowseVideo.place(relx=.757, rely=.45, anchor="e")

		#play_img =  Image.open("2.png")
		#play_img2 = ImageTk.PhotoImage(play_img)

		self.startButton = Button(self.background_label,height=1,width=10, text="Start", font=('Times New Roman', 12, 'bold'), command= self.startButtonPressed,fg='#27408b',bg="white",activebackground="white",activeforeground='#27408b', highlightthickness=1, highlightbackground='#27408b')
		self.startButton.place(relx=.845, rely=.45, anchor="center")
	
		self.DemoButton = Button(self.background_label, text="Click for Demo", font=('Times New Roman', 12, 'bold'), command= self.demoButtonPressed,fg='#27408b',bg="white",activebackground="white",activeforeground='#27408b', highlightthickness=1, highlightbackground='#27408b')
		self.DemoButton.place(relx=.5, rely=.7, anchor="center")

	def startButtonPressed(self):
    	
		video_path = self.entryVideo.get()
		#print(video_path) 
		os.chdir('/Users/apple/Desktop/Development/Study/deep_learning/Project_2A/AutoNarrate_v0.1/im2txt')
		os.system('bazel-bin/im2txt/run_inference --checkpoint_path="/Users/apple/Desktop/Development/Study/deep_learning/Project_2A/AutoNarrate_v0.1/im2txt/im2txt/model.ckpt-3000000" --vocab_file="/Users/apple/Desktop/Development/Study/deep_learning/Project_2A/AutoNarrate_v0.1/im2txt/im2txt/word_counts.txt" --input_files='+video_path)
	
	def BrowseVideoButtonPressed(self):
		file = tkFileDialog.askopenfilename(title='Choose a video file')
		self.entryVideo.delete(0, END)
		self.entryVideo.insert(0, file)


	def demoButtonPressed(self):
		url = "https://www.youtube.com/watch?v=TBQQziS7tbc"
		webbrowser.open(url)

app = Tk()
applicationGui(app)
app.mainloop()