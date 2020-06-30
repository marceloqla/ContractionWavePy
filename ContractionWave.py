#imports of tkinter
import tkinter as tk                # python 3
from ttkthemes import themed_tk as tk1
from tkinter import font  as tkfont # python 3
from tkinter import filedialog
from tkinter import ttk
#import ttk2 #this one has Spinbox in ttk2
from tkinter import messagebox
#

#imports of external libs
from PIL import ImageTk
from PIL import Image

import multiprocessing
from multiprocessing import Process, Manager, active_children#, Queue
from collections import deque

import os, pickle, cv2, psutil, time, copy, locale
import datetime as dt
from sys import platform as _platform
import warnings
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*",category=UserWarning)

import xlsxwriter
if _platform == "win32" or _platform == "win64":
    import pkg_resources.py2_warn
#import pandas as pd

import matplotlib as mpl

#AXES
mpl.rcParams['axes.titlepad'] = 10
mpl.rcParams["axes.facecolor"]='white'
mpl.rcParams['axes.edgecolor']='black'
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelsize'] = 12

#FONT
mpl.rcParams['font.family'] ='Helvetica'
mpl.rcParams['font.weight'] = 'normal'

#TICK
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'

#FIGURE
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['figure.figsize'] = [8.0, 6.0]


#Legend
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.numpoints'] = 3
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.framealpha'] = None
mpl.rcParams['legend.scatterpoints'] = 3
mpl.rcParams['legend.edgecolor'] = 'inherit'


import matplotlib.pyplot as plt
# plt.style.use('ggplot')

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from matplotlib.backend_bases import key_press_handler
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import seaborn as sns
# sns.set()


from scipy.signal import savgol_filter
import numpy as np
import io

#imports of custom written classes and functions
from customframe import ttkScrollFrame
from draghandlers import PeaksObj, MoveDragHandler#,PeakObj
from customdialogs import AboutDialog, HelpDialog, FolderSelectDialog, CoreAskDialog, SelectMenuItem, AddPresetDialog, DotChangeDialog, SelectPresetDialog, PlotSettingsProgress, QuiverJetSettings, AdjustNoiseDetectDialog, SaveFigureVideoDialog, SaveFigureDialog, SaveTableDialog, SavGolDialog, NpConvDialog, FourierConvDialog, SummarizeTablesDialog, QuiverJetMaximize, WaitDialogProgress
from smoothregress import exponential_fit, noise_detection, peak_detection, smooth_scipy#, smooth_data, _1gaussian, _2gaussian, noise_detection
from tooltip import CreateToolTip

used_separator = "/"

img_opencv = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")

def opticalflowfolder(queueobj, object_to_flow, stamp):
    pyr_scale = object_to_flow.pyr_scale
    levels = object_to_flow.levels
    winsize = object_to_flow.winsize
    iterations = object_to_flow.iterations
    poly_n = object_to_flow.poly_n
    poly_sigma = object_to_flow.poly_sigma
    fps = object_to_flow.FPS
    pixel_val = object_to_flow.pixelsize
    global filesprocessed
    queueobj.put(stamp+" TIME "+ " ".join(["--", "---", "--:--:--"]) )
    if object_to_flow.gtype == "Folder":
        global img_opencv
        files_grabbed = [x for x in os.listdir(object_to_flow.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
        files_grabbed = sorted(files_grabbed)
        files_grabbed = [object_to_flow.gpath + "/" + a for a in files_grabbed]
        starttime = time.time()
        for j in range(len(files_grabbed)-1):    
            #Dense Optical Flow in OpenCV (Gunner Farneback's algorithm)
            frame1 = cv2.imread(r'%s' % files_grabbed[0+j], -1)
            frame1 = frame1.astype('uint8')
            frame2 = cv2.imread(r'%s' %files_grabbed[1+j], -1)
            frame2 = frame2.astype('uint8')
            if len(frame1.shape) >= 3:
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            else:
                prvs = frame1
                prvs2 = frame2
            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            #flows are appended to shared memory object
            meanval = abs(mag.mean() * fps * pixel_val)
            meanval = float("{:.3f}".format(meanval))

            queueobj.put(stamp+" MEANS "+ str(j) +" " +str(meanval))
            queueobj.put(stamp+" PROGRESS "+str((j+1) / (len(files_grabbed)-1)))

            telapsed = time.time() - starttime
            testimated = (telapsed/(j+1))*(len(files_grabbed))

            finishtime = starttime + testimated
            finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

            lefttime = testimated-telapsed  # in seconds
            queueobj.put(stamp+" TIME "+ " ".join([str(int(telapsed)), str(int(lefttime)), str(finishtime)]) )

    elif object_to_flow.gtype == "Video":
        vc = cv2.VideoCapture(r'%s' % object_to_flow.gpath)
        _, frame1 = vc.read()
        frame1 = frame1.astype('uint8')
        total_frames = int(object_to_flow.framenumber)
        starttime = time.time()
        j = 0
        while(vc.isOpened() and j < total_frames -1):
            _, frame2 = vc.read()
            frame2 = frame2.astype('uint8')
            if len(frame1.shape) >= 3:
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            else:
                prvs = frame1
                prvs2 = frame2
            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            meanval = abs(mag.mean() * fps * pixel_val)
            meanval = float("{:.3f}".format(meanval))

            queueobj.put(stamp+" MEANS "+ str(j) +" " +str(meanval))
            queueobj.put(stamp+" PROGRESS "+str((j+1) / (total_frames-1)))

            telapsed = time.time() - starttime
            testimated = (telapsed/(j+1))*(total_frames)
            finishtime = starttime + testimated
            finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time
            lefttime = testimated-telapsed  # in seconds

            queueobj.put(stamp+" TIME "+ " ".join([str(int(telapsed)), str(int(lefttime)), str(finishtime)]) )
            frame1 = frame2.copy()
            j += 1
        pass
    elif object_to_flow.gtype == "Tiff Directory":
        _, images = cv2.imreadmulti(r'%s' % object_to_flow.gpath, None, cv2.IMREAD_COLOR)
        starttime = time.time()
        for j in range(len(images)-1):
            #Dense Optical Flow in OpenCV (Gunner Farneback's algorithm)
            frame1 = images[0+j]
            frame1 = frame1.astype('uint8')
            frame2 = images[1+j]
            frame2 = frame2.astype('uint8')
            if len(frame1.shape) >= 3:
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            else:
                prvs = frame1
                prvs2 = frame2
            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            #flows are appended to shared memory object
            meanval = abs(mag.mean() * fps * pixel_val)
            meanval = float("{:.3f}".format(meanval))

            queueobj.put(stamp+" MEANS "+ str(j) +" " +str(meanval))
            queueobj.put(stamp+" PROGRESS "+str((j+1) / (len(images)-1)))

            telapsed = time.time() - starttime
            testimated = (telapsed/(j+1))*(len(images))

            finishtime = starttime + testimated
            finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

            lefttime = testimated-telapsed  # in seconds
            queueobj.put(stamp+" TIME "+ " ".join([str(int(telapsed)), str(int(lefttime)), str(finishtime)]) )
    queueobj.put(stamp+" TIME "+ " ".join(["--", "---", "--:--:--"]) )


def update_running_tasks():
    global running_tasks, ncores, processingdeque
    while len(running_tasks) < ncores and len(processingdeque) > 0:
        p = processingdeque.popleft()
        p.start()
        running_tasks.append(psutil.Process(pid=p.pid))

def addqueue(group):
    now = dt.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    seconds = '{:02d}'.format(now.second)
    stamp =  '{}{}{}{}{}{}'.format(year, month, day, hour, minute, seconds)
    global stamp_to_group
    stamp_to_group[stamp] = group
    global qmanagerflows, processingdeque, progress_tasks, globalq
    progress_tasks[stamp] = 0.0
    qmanagerflows[stamp+"_means"] = [0.0 for a in range(int(group.framenumber)-1)]
    newtask = Process(target=opticalflowfolder, args=(globalq, group, stamp))
    processingdeque.append(newtask)
    update_running_tasks()

def destroyProcesses():
    active_children()  # Joins all finished processes.
    global running_tasks#, filesprocessed
    actual_tasks = running_tasks.copy()
    for p in actual_tasks:
        if not p.is_running():
            running_tasks.remove(p)
        else:
            p.terminate()
    active_children()  # Joins all finished processes.

def checkQueue():
    active_children()  # Joins all finished processes.
    global running_tasks, progress_tasks, tasks_time, globalq, qmanagerflows, checkqlock
    checkqlock = True
    actual_tasks = running_tasks.copy()

    while not globalq.empty():
        message = globalq.get()
        print(message)
        stamp = message.split()[0]
        msgtype = message.split()[1]
        values = message.split()[2:]
        if msgtype == "PROGRESS":
            progress_tasks[stamp] = float(values[0])
        if msgtype == "TIME":
            tasks_time[stamp] = values
        if msgtype == "MEANS":
            qmanagerflows[stamp+"_means"][int(values[0])] = float(values[1])

    for p in actual_tasks:
        if not p.is_running():  # process has finished
            running_tasks.remove(p)
            update_running_tasks()
        else:
            pass
    checkqlock = False
    return progress_tasks.copy()

default_values = {
    "FPS" : 200,
    "pixelsize" : 0.2375,
    "pyr_scale" : 0.5,
    "levels" : 1,
    "winsize" : 15,
    "iterations" : 1,
    "poly_n" : 7,
    "poly_sigma" : 1.5,
}

default_values_bounds = {
    "FPS" : None,
    "pixelsize" : None,
    "pyr_scale" : (0,1),
    "levels" : (0,100),
    "winsize" : (0,100),
    "iterations" : (0,100),
    "poly_n" : (0,100),
    "poly_sigma" : (0,100),
    "current_windowX" : [1,100],
    "current_windowY" : [1,100],
    "blur_size" : [1,100],
    "kernel_dilation" : [1,100],
    "kernel_erosion" : [1,100],
    "kernel_smoothing_contours" : [1,100],
    "border_thickness" : [1,100],
    "minscale":[0,10000000000],
    "maxscale":[0,10000000000]
}

class PlotSettings(object):
    def __init__(self):
        self.peak_plot_colors = {
            "main": 'black',
            "first":'green',
            "max":'red',
            "min":'blue',
            "last":'purple',
            "fft": 'red',
            "fft_selection": 'purple',
            "noise_true": 'blue',
            "noise_false": 'red',
            "rect_color": '#fd9a53'
        }
        self.plotline_opts = {
            "zero": False,
            "zero_color": 'black',
            "grid": True,
            "grid_color": 'black',
            "time_unit": 's',
            "absolute_time": False
        }
        self.savgol_opts = {
            "window_length": 5,
            "polynomial_order": 2,
            "above_zero": 0,
            "start_x": 0,
            "end_x": 0
        }
        self.np_conv = {
            "window_length": 3,
            "window_type": 'flat',
            "above_zero": 0,
            "start_x": 0,
            "end_x": 0
        }
        self.fourier_opts = {
            "frequency_maintain": 0.33,
            "above_zero": 0,
            "start_x": 0,
            "end_x": 0
        }

    def set_limit(self, tsize):
        self.savgol_opts["end_x"] = tsize
        self.np_conv["end_x"] = tsize
        self.fourier_opts["end_x"] = tsize

def get_Frames_Folder_Number(folderpath):
    global img_opencv
    files_grabbed = [x for x in os.listdir(folderpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
    return len(files_grabbed)

def get_Frames_Video(videopath):
    cap = cv2.VideoCapture(r'%s' %videopath)
    total_frames = cap.get(7)
    return total_frames

def get_Video_FPS(videopath):
    cap = cv2.VideoCapture(r'%s' %videopath)
    fps = cap.get(5)
    return int(fps)

def get_Frames_CTiff(tiffpath):
    _, images = cv2.imreadmulti(r'%s' %tiffpath)
    return len(images)

class AnalysisGroup(object):
    def __init__(self, name, gpath, gtype):
        #Creation settings
        self.name = name
        self.gpath = gpath
        self.framenumber = 0

        self.gtype = gtype

        if self.gtype == "Folder":
            self.framenumber = get_Frames_Folder_Number(self.gpath)
        elif self.gtype == "Video":
            self.framenumber = get_Frames_Video(self.gpath)
        elif self.gtype == "Tiff Directory":
            self.framenumber = get_Frames_CTiff(self.gpath)

        self.lindex = None
        self.id = None

        #Run Flow Settings
        self.saverun = False
        self.FPS = None
        self.pixelsize = None
        self.pyr_scale = None
        self.levels = None
        self.winsize = None
        self.iterations = None
        self.poly_n = None
        self.poly_sigma = None

        #Saved vars during analysis
        self.noisemin = None
        self.delta = None
        self.stopcond = None

        #Run Flow Return Variables
        self.mag_means = []

    def set_valtype(self, valtype, valthis):
        print("class AnalysisGroup setting: " + valtype + " to: " + str(valthis))
        if valtype == "FPS":
            self.FPS = valthis
        elif valtype == "pixelsize":
            self.pixelsize = valthis
        elif valtype == "pyr_scale":
            self.pyr_scale = valthis
        elif valtype == "levels":
            self.levels = valthis
        elif valtype == "winsize":
            self.winsize = valthis
        elif valtype == "iterations":
            self.iterations = valthis
        elif valtype == "poly_n":
            self.poly_n = valthis
        elif valtype == "poly_sigma":
            self.poly_sigma = valthis
        print("class AnalysisGroup setting done")

    def get_valtype(self, valtype):
        print("class AnalysisGroup retrieving: " + valtype)
        if valtype == "FPS":
            return self.FPS
        elif valtype == "pixelsize":
            return self.pixelsize
        elif valtype == "pyr_scale":
            return self.pyr_scale
        elif valtype == "levels":
            return self.levels
        elif valtype == "winsize":
            return self.winsize
        elif valtype == "iterations":
            return self.iterations
        elif valtype == "poly_n":
            return self.poly_n
        elif valtype == "poly_sigma":
            return self.poly_sigma
        print("class AnalysisGroup retrieving done")

class SampleApp(tk1.ThemedTk):

    #TODO LIST:
    #TODO BUG: Install on Spyder
    #TODO: Separar em Classes distintas para cada coisa

    def __init__(self, *args, **kwargs):
        print("class SampleApp def init start")
        self.controller = self
        
        # tk.Tk.__init__(self, *args, **kwargs)
        tk1.ThemedTk.__init__(self, *args, **kwargs)

        if _platform == "linux" or _platform == "linux2":
            # linux
            self.set_theme("scidblue")
            # self.set_theme("clearlooks")
        elif _platform == "darwin":
            # MAC OS X
            try:
                self.set_theme("scidblue")
            except TclError:
                pass
            try:
                self.set_theme("clearlooks")
            except TclError:
                self.set_theme("classic")
        elif _platform == "win32":
            # Windows
            self.set_theme("xpnative")
        elif _platform == "win64":
            # Windows 64-bit
            self.set_theme("xpnative")

        self.title('ContractionWave')
        self.bgcolor = self._get_bg_color()

        if "#" not in self.bgcolor:
            tempwidget = tk.Label(self)
            rgb = tempwidget.winfo_rgb(self.bgcolor)
            r,g,b = [x>>8 for x in rgb]
            facecolordo = '#{:02x}{:02x}{:02x}'.format(r,g,b)
            tempwidget.destroy()
            self.bgcolor = facecolordo

        self.configure(bg=self.bgcolor)

        # icon = ImageTk.PhotoImage(file='icons/Logo_CW.gif')
        icon = ImageTk.PhotoImage(file=os.path.abspath('./icons/Logo_CW.gif'))

        if _platform == "linux" or _platform == "linux2":
            # linux
            self.iconbitmap('@Logo_CW.xbm')
            #http://effbot.org/tkinterbook/wm.htm#Tkinter.Wm.iconbitmap-method%20quote
            #https://stackoverflow.com/questions/42705547/which-file-formats-can-i-use-for-tkinter-icons
            #https://stackoverflow.com/questions/20860325/python-3-tkinter-iconbitmap-error-in-ubuntu
            # self.wm_iconbitmap(bitmap = "@Logo_CW.xbm")
            # img = tk.Image("photo", file="icons/Logo_CW.gif")
            # self.tk.call('wm','iconphoto',self._w,img)
            # self.tk.call('wm', 'iconphoto', self._w, icon)
            # self.wm_iconbitmap(bitmap = "@Logo_CW.ico")
        elif _platform == "darwin":
            # MAC OS X
            self.iconphoto(True, icon)
        elif _platform == "win32":
            # Windows
            self.iconphoto(True, icon)
        elif _platform == "win64":
            # Windows 64-bit
            self.iconphoto(True, icon)

        self.title_font = tkfont.Font(family='Helvetica', size=18)
        self.current_frame = None
        self.queuestarted = False
        self.queue = False
        self.ttkStyles = ttk.Style()

        self.ttkStyles.configure('greyBackground', background="#d3d3d3")
        self.ttkStyles.configure('whiteBackground', background="#ffffff")
        self.ttkStyles.configure('greyBackground.TLabel', background='#d3d3d3')
        self.ttkStyles.configure('whiteBackground.TLabel', background="#ffffff")
        self.ttkStyles.configure('greyBackground.TFrame', background='#d3d3d3')
        self.ttkStyles.configure('whiteBackground.TFrame', background="#ffffff")
        self.ttkStyles.configure('greyBackground.TRadiobutton', background='#d3d3d3')
        self.ttkStyles.configure('greyBackground.TRadioButton', background='#d3d3d3')
        self.ttkStyles.configure('greyBackground.TCheckbutton', background='#d3d3d3')
        self.ttkStyles.configure('greyBackground.TButton', background='#d3d3d3')
        self.ttkStyles.configure('greyBackground.Horizontal.TScale', background='#d3d3d3')

        self.current_analysis = None
        self.peaks = None
        
        self.current_peak = None
        self.current_framelist = None
        self.current_maglist = None
        self.current_anglist = None
        self.current_timescale = "s"
        self.current_speedscale = "µ/s" #\u00B5 or µ
        self.current_areascale = "µ^2"

        self.buffervariables = None
        self.do_reset = False
        self.btn_lock = False
        
        self.done_groups = {}


        self.plotsettings = PlotSettings()
        self.preset_dicts = {
            "Neonate Mice Cardiomyocyte": {
                "pyr_scale" : 0.5,
                "levels" : 1,
                "winsize" : 15,
                "iterations" : 1,
                "poly_n" : 7,
                "poly_sigma" : 1.5,
            },
            "Adult Mice Cardiomyocyte": {
                "pyr_scale" : 0.5,
                "levels" : 3,
                "winsize" : 15,
                "iterations" : 3,
                "poly_n" : 7,
                "poly_sigma" : 1.5,
            }
        }
        if not os.path.exists('userprefs/'):
            os.makedirs('userprefs/')
        if os.path.exists('userprefs/userpresets.pickle'):
            filehandler = open('userprefs/userpresets.pickle', 'rb')
            userpresets = pickle.load(filehandler)
            self.preset_dicts = userpresets
            filehandler.close()

        self.geometry('1074x640')
        self.update_idletasks()
        # self.attributes('-fullscreen', True)
        # self.wm_attributes('-zoomed', 1)

        width = self.winfo_width()
        frm_width = self.winfo_rootx() - self.winfo_x()
        win_width = width + 2 * frm_width
        height = self.winfo_height()
        titlebar_height = self.winfo_rooty() - self.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = self.winfo_screenwidth() // 2 - win_width // 2
        y = self.winfo_screenheight() // 2 - win_height // 2
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.deiconify()

        #define icons
        self.wd = None

        # self.loaddataimg = tk.PhotoImage(file="icons/folder-icon.png")
        # self.progresspic = tk.PhotoImage(file="icons/folder-blue-activities-icon.png")
        # self.startanalysis = tk.PhotoImage(file="icons/folder-green-vbox-icon.png")
        # self.loadpeaks = tk.PhotoImage(file="icons/folder-green-documents-icon.png")
        # self.summtables = tk.PhotoImage(file="icons/folder-green-wine-icon.png")

        #Common
        self.gotostartpage = tk.PhotoImage(file="icons/refresh-sharp.png")
        self.goback = tk.PhotoImage(file="icons/arrow-back-sharp.png")

        #StartPage
        self.loaddataimg = tk.PhotoImage(file="icons/folder-open-sharp.png")
        self.progresspic = tk.PhotoImage(file="icons/ellipsis-horizontal-sharp.png")
        self.startanalysis = tk.PhotoImage(file="icons/analytics-sharp.png")
        self.loadpeaks = tk.PhotoImage(file="icons/pulse-sharp.png")
        self.summtables = tk.PhotoImage(file="icons/apps-sharp.png")

        #PageOne
        # New Data folder-open-sharp
        #     Load Data add-sharp
        #         Folder images-sharp 
        #         Video  film-sharp 
        #         Compressed Tiff duplicate-sharp
        #     Delete All close-sharp
        #     Run All checkmark-done-sharp
        #     Go to the start page refresh-sharp
        #     Check progress ellipsis-horizontal-sharp

        self.openloaddialog = tk.PhotoImage(file="icons/add-sharp.png")
        self.deleteallimg = tk.PhotoImage(file="icons/close-sharp.png")
        self.runallimg = tk.PhotoImage(file="icons/checkmark-done-sharp.png")

        #PageTwo
        # Check Progress ellipsis-horizontal-sharp
        #     Analysis analytics-sharp
        #     Go to the start page refresh-sharp

        #PageThree
        # Start Analysis analytics-sharp
        #     Download Table download-sharp
        #     Start Analysis analytics-sharp
        #     Go to the start page refresh-sharp

        self.downloadtableimg = tk.PhotoImage(file="icons/download-sharp.png")

        #PageFour
        # Go back arrow-back-sharp
        # Analyse Wave Areas pulse-sharp
        # Go to the start page refresh-sharp
        self.analysewvareas = tk.PhotoImage(file="icons/pulse-sharp.png")

        #PageFive
        # Load Saved Waves pulse-sharp
        #     Go back arrow-back-sharp
        #     Quiver/Jet Plots layers-sharp
        #     Go to the start page refresh-sharp
        self.jetquiverpltimg = tk.PhotoImage(file="icons/layers-sharp.png")

        #PageSix
        #     Go back arrow-back-sharp
        #     Go to the start page refresh-sharp

        self.mainapppic = tk.PhotoImage(file="icons/cw_a.png")
        self.playstopicon=tk.PhotoImage(file="icons/startstop.png")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        # container = tk.Frame(self)
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            
            # frame.configure(style='greyBackground.TFrame')

            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_propagate(0)

        self.popuplock = False
        self.currentpopup = None
        self.bind("<Button-1>", self.mouse_function)
        # self.bind("<FocusOut>", self.focus_out_menu)
        self.show_frame("StartPage", firsto=True)

    def showwd(self):
        if self.wd == None:
            self.wd = WaitDialogProgress(self, title='Queueing Groups...')

    def cancelwd(self):
        if self.wd != None:
            self.wd.cancel()
            self.wd.destroy()
        self.wd = None

    
    def on_closing(self):
        destroyProcesses()
        self.quit()
        # self.destroy()
        raise SystemExit(0)

    def focus_out_menu(self, event=None):
        pass

    def mouse_function(self, event=None):
        if self.popuplock == True:
            if (event.widget != self.currentpopup):
                self.currentpopup.grab_release()
                self.currentpopup.unpost()
                event.widget.focus_set()
            self.popuplock = False
            self.currentpopup = None


    def show_frame(self, page_name, firsto=False, bckbtn=None):
        if self.btn_lock == False:
            self.btn_lock = True
            if self.current_frame != None:
                name = self.current_frame.fname
                if name == "PageSix":
                    if self.current_frame.animation_status == True:
                        self.current_frame.startstopanimation()
                    if self.current_frame.maximizeplot != None:
                        self.current_frame.maximizeplot.cancel()
                    if self.current_frame.advsettings != None:
                        self.current_frame.advsettings.cancel()

            '''Show a frame for the given page name'''
            frame = self.frames[page_name]
            if firsto == True:
                self.ttkStyles.configure('defaultTBg', background="#d3d3d3")
                self.ttkStyles.configure('whiteTBg', background="#ffffff")
                # self.ttkStyles.configure("Green.TProgressbar", foreground="#47b526", background="#47b526")
            validate = True
            if page_name == "StartPage":
                self.current_analysis = None
                self.peaks = None
                self.selectedframes = None
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
            if page_name == "PageOne":
                self.current_analysis = None
                self.peaks = None
                self.selectedframes = None
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
            if page_name == "PageTwo":
                self.current_analysis = None
                self.peaks = None
                self.selectedframes = None
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
                frame.add_new_progressframe()
            if page_name == "PageThree":
                self.current_analysis = None
                self.peaks = None
                self.selectedframes = None
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
                frame.generate_default_table()
                frame.listboxpopulate()
            if page_name == "PageFour":
                #self.current_analysis
                self.peaks = None
                self.selectedframes = None
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
                validate = frame.init_vars(bckbtn=bckbtn)
            if page_name == "PageFive":
                self.current_peak = None
                self.current_framelist = None
                self.current_maglist = None
                self.current_anglist = None
                if frame.current_tabletree is not None:

                    frame.current_tabletree.grid_forget()
                    frame.current_subtabletree.grid_forget()
                    frame.current_tablevsb.grid_forget()
                    frame.current_tablevsb2.grid_forget()

                    if len(frame.current_tabletree.selection()) > 0:
                        frame.current_tabletree.selection_remove(frame.current_tabletree.selection()[0])
                    frame.current_tabletree.selection_clear()

                frame.current_tabletree = None
                frame.current_tablevsb = None
                frame.changeval.set(0)
                validate = frame.init_vars()
                frame.settabletype()
            if page_name == "PageSix":
                validate = frame.init_vars()
                frame.init_viz()
                frame.init_ax2()
            # frame.update()

            if self.do_reset == True and validate == False:
                #if is trying to load but can't validate, copy back the previous objects
                if self.buffervariables[0] != None:
                    self.current_analysis = copy.deepcopy(self.buffervariables[0])
                else:
                    self.current_analysis = None
                if self.buffervariables[1] != None:
                    self.peaks = copy.deepcopy(self.buffervariables[1])
                else:
                    self.peaks = None
                if self.buffervariables[2] != None:
                    self.selectedframes = self.buffervariables[2].copy()
                else:
                    self.selectedframes = None
                if self.buffervariables[3] != None:
                    self.current_peak = copy.deepcopy(self.buffervariables[3])
                else:
                    self.current_peak = None

                if self.buffervariables[4] is None:
                    self.current_framelist = None
                else:
                    self.current_framelist = self.buffervariables[4].copy()
                if self.buffervariables[5] is None:
                    self.current_maglist = None
                else:
                    self.current_maglist = self.buffervariables[5].copy()
                if self.buffervariables[6] is None:
                    self.current_anglist = None
                else:
                    self.current_anglist = self.buffervariables[6].copy()

            self.buffervariables = None
            self.do_reset = False

            if validate == True:
                self.current_frame = frame
                frame.tkraise()
                menubar = frame.menubar(self)
                self.configure(menu=menubar)
            self.btn_lock = False
    
    def showabout(self):
        self.btn_lock = True
        AboutDialog(self, title='Contraction Wave')
        self.btn_lock = False
        pass

    def showhelp(self):
        self.btn_lock = True
        HelpDialog(self, title='Help (?)', literals=[("current_help", self.current_frame.fname)])
        self.btn_lock = False

    def reset_and_show(self, page_name):
        self.do_reset = True

        #safety save current vars
        self.btn_lock = True
        self.buffervariables = []
        if self.current_analysis != None:
            self.buffervariables.append(copy.deepcopy(self.current_analysis))
        else:
            self.buffervariables.append(None)
        if self.peaks != None:        
            self.buffervariables.append(copy.deepcopy(self.peaks))
        else:
            self.buffervariables.append(None)
        if self.selectedframes != None:
            self.buffervariables.append(self.selectedframes.copy())
        else:
            self.buffervariables.append(None)
        if self.current_peak != None:        
            self.buffervariables.append(copy.deepcopy(self.current_peak))
        else:
            self.buffervariables.append(None)

        if self.current_framelist is None:     
            self.buffervariables.append(None)  
        else:
            self.buffervariables.append(self.current_framelist.copy())

        if self.current_maglist is None:   
            self.buffervariables.append(None)     
        else:
            self.buffervariables.append(self.current_maglist.copy())

        if self.current_anglist is None:       
            self.buffervariables.append(None) 
        else:
            self.buffervariables.append(self.current_anglist.copy())

        #reset current vars
        self.current_analysis = None
        self.peaks = None
        self.current_peak = None
        self.current_framelist = None
        self.current_maglist = None
        self.current_anglist = None

        #show frame
        self.btn_lock = False
        self.show_frame(page_name)

    def configplotsettings(self):
        self.btn_lock = True
        d = PlotSettingsProgress(self, title='Edit Plot Settings:', literals=[
            ("plotsettingscolors", self.plotsettings.peak_plot_colors.copy()),
            ("plotsettingsline", self.plotsettings.plotline_opts.copy()), 
            ("current_time", self.current_timescale)
            ])
        if d.result:
            self.plotsettings.peak_plot_colors = d.result["peak_plot_colors"]
            self.plotsettings.plotline_opts = d.result["plotline_opts"]
            self.current_timescale = self.plotsettings.plotline_opts["time_unit"]
        if self.current_frame.fname == "PageThree":
            self.current_frame.update_headings()
            self.current_frame.onselect_event()
        if self.current_frame.fname == "PageFour":
            self.current_frame.plotsettings = self.plotsettings
            print(" def configplotsettings self.current_frame.update_with_delta_freq()")
            self.current_frame.update_with_delta_freq()
        if self.current_frame.fname == "PageFive":
            self.current_frame.plotsettings = self.plotsettings
            for peak in self.peaks:
                peak.switch_timescale(self.current_timescale)
            self.current_frame.settabletype()
            self.current_frame.tabletreeselection()
        if self.current_frame.fname == "PageSix":
            for peak in self.peaks:
                peak.switch_timescale(self.current_timescale)
            self.current_peak.switch_timescale(self.current_timescale)
            self.current_frame.plotsettings = self.plotsettings
            self.current_frame.init_ax2()
        self.btn_lock = False

    def saveplotsettings(self):  
        self.btn_lock = True      
        #Check if saving folder exists
        if not os.path.exists('savedplotprefs'):
            os.makedirs('savedplotprefs/')
        f = filedialog.asksaveasfile(title = "Save Plot Preferences", mode='w', initialdir= "./savedplotprefs/")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        f.close()
        try:
            fname = str(f.name)
            fname = r'%s' %fname
            fname2 = fname.split(".")[0]+".pickle" 
            filehandler = open(r'%s' % fname2, 'wb') 
            # pickle.dump(self.plotsettings.peak_plot_colors.copy(), filehandler, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.plotsettings.peak_plot_colors.copy(), filehandler, protocol=3)
            filehandler.close()
        except Exception as e:
            messagebox.showerror("Error", "Could not save Plot Preferences file\n" + str(e))
        self.btn_lock = False

    def loadplotsettings(self):
        self.btn_lock = True
        if not os.path.exists('savedplotprefs'):
            os.makedirs('savedplotprefs/')        
        filename = filedialog.askopenfilename(title = "Load Plot Preferences file", initialdir="./savedplotprefs/",filetypes = (("pickle analysis files","*.pickle"),("all files","*.*")))
        filename = r'%s' %filename
        validate = True
        if filename != None:
            try:                
                filehandler = open(r'%s' %filename, 'rb')
                try:
                    diskclass = pickle.load(filehandler)
                except Exception as e:
                    messagebox.showerror("Error", "Could not load file\n" + str(e))
                    validate = False
                filehandler.close()
                if validate == True and isinstance(diskclass, dict):
                    self.plotsettings.peak_plot_colors = diskclass
                elif validate == True:
                    messagebox.showerror("Error", "Loaded File is not a Plot Preferences File")
                    validate = False
            except Exception as e:
                messagebox.showerror("Error", "File could not be loaded\n" + str(e))
                validate = False
        else:
            messagebox.showerror("Error", "File could not be loaded")
            validate = False
        if self.current_frame.fname == "PageFour":
            print(" def loadplotsettings self.current_frame.update_with_delta_freq()")
            self.current_frame.update_with_delta_freq()
        if self.current_frame.fname == "PageFive":
            self.current_frame.tabletreeselection()
        if self.current_frame.fname == "PageSix":
            self.current_frame.init_ax2()
        self.btn_lock = False

    def checkTheQueue(self):
        current_progress_tasks = checkQueue()
        self.queue = False
        global stamp_to_group, qmanagerflows, progress_tasks
        for etask in progress_tasks.keys():
            if current_progress_tasks[etask] < 1.0:
                self.queue = True
            elif current_progress_tasks[etask] == 1.0 and etask not in self.done_groups.keys():
                doneg = stamp_to_group[etask]
                doneg.mag_means = list(qmanagerflows[etask+"_means"]).copy()
                if doneg.saverun == True:
                    #Check if saving folder exists
                    if not os.path.exists('savedgroups'):
                        os.makedirs('savedgroups/')
                    try:
                        filehandler = open("savedgroups/" + etask + ".pickle" , 'wb') 
                        # pickle.dump(doneg, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(doneg, filehandler, protocol=3)
                        filehandler.close()
                    except Exception as e:
                        messagebox.showerror("Error", "Could not save Analysis file\n" + str(e))
                self.done_groups[etask] = doneg

        self.frames["PageTwo"].add_new_progressframe()

        if self.queue == True:
            self.after(100, self.checkTheQueue)

class StartPage(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(style='greyBackground.TFrame')
        self.fname = "StartPage"

        for i in range(0,8):
            self.rowconfigure(i, weight=1)
        for i in range(0,5):
            self.columnconfigure(i, weight=1)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=0, column=1, rowspan=3, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)
        self.canvas.bind("<Configure>", self.resize)
        self.img  = ImageTk.PhotoImage(
            Image.open("icons/cw_a.png")
        )

        self.canvas.config(bg=self.controller._get_bg_color())
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.controller.mainapppic) 
        #self.mainapppic

        btn1frame = ttk.Frame(self)#, style='greyBackground.TFrame')
        button1 = ttk.Button(btn1frame, text=" New Data",
                            command=lambda: controller.show_frame("PageOne"))#, style='greyBackground.TButton')
        button1lbl = ttk.Label(btn1frame, text=" New Data",width=17)#, style='greyBackground.TLabel')

        btn2frame = ttk.Frame(self)#, style='greyBackground.TFrame')
        button2 = ttk.Button(btn2frame, text=" Check Progress",
                            command=lambda: controller.show_frame("PageTwo"))#, style='greyBackground.TButton')
        button2lbl = ttk.Label(btn2frame, text=" Check Progress",width=17,)# style='greyBackground.TLabel')

        btn3frame = ttk.Frame(self)#, style='greyBackground.TFrame')
        button3 = ttk.Button(btn3frame, text=" Start Analysis",
                            command=lambda: controller.show_frame("PageThree"))#, style='greyBackground.TButton')
        button3lbl = ttk.Label(btn3frame, text=" Start Analysis",width=17)#, style='greyBackground.TLabel')

        btn4frame = ttk.Frame(self)#, style='greyBackground.TFrame')
        button4 = ttk.Button(btn4frame, text=" Load Saved Waves",
                            command=lambda: controller.show_frame("PageFive"))#, style='greyBackground.TButton')
        button4lbl = ttk.Label(btn4frame, text=" Load Saved Waves",width=17)#, style='greyBackground.TLabel')
       
        btn5frame = ttk.Frame(self)#, style='greyBackground.TFrame')
        button5 = ttk.Button(btn5frame, text=" Merge Results",
                            command= self.summarize_tables)#, style='greyBackground.TButton')
        button5lbl = ttk.Label(btn5frame, text=" Merge Results",width=17)#, style='greyBackground.TLabel')

        button1.config(image=self.controller.loaddataimg, width=60)
        button2.config(image=self.controller.progresspic, width=60)
        button3.config(image=self.controller.startanalysis, width=60)
        button4.config(image=self.controller.loadpeaks, width=60)
        button5.config(image=self.controller.summtables, width=60)

        button1lbl.grid(row=0, column=1)
        button1.grid(row=0, column=0)
        btn1frame.grid(row=5, column=1)

        button2lbl.grid(row=0, column=1)
        button2.grid(row=0, column=0)
        btn2frame.grid(row=6, column=1)

        button3lbl.grid(row=0, column=1)
        button3.grid(row=0, column=0)
        btn3frame.grid(row=5, column=3)

        button4lbl.grid(row=0, column=1)
        button4.grid(row=0, column=0)
        btn4frame.grid(row=6, column=3)

        button5lbl.grid(row=0, column=1)
        button5.grid(row=0, column=0)
        btn5frame.grid(row=7, column=3)

    def resize(self, event):
        self.controller.btn_lock = True
        
        original_width = 1200
        original_height = 406

        evwidth = event.width
        evheight = event.height
        if evwidth > original_width * 0.8:
            evwidth = original_width * 0.8
        if evheight > original_height * 0.8:
            evheight = original_height * 0.8
        smallest_fac = np.min( [(evwidth/ original_width), (evheight/ original_height)] )
        new_height = int(original_height * smallest_fac)
        new_width = int(original_width * smallest_fac)

        img = Image.open("icons/cw_a.png").resize(
            (new_width, new_height), Image.ANTIALIAS
        )
        self.img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_img, image=self.img)
        self.controller.btn_lock = False

    def summarize_tables(self):
        self.controller.btn_lock = True
        SummarizeTablesDialog(self, title="Summarize Tables:")
        self.controller.btn_lock = False

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        menubar.add_cascade(label="File", menu=pageMenu)
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)

        return menubar

class PageOne(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageOne"
        self.selectedgroup = None
        self.analysisgroups = []
        for i in range(0,13):
            self.rowconfigure(i, weight=1)
        for i in range(0,4):
            self.columnconfigure(i, weight=1)
        
        label = ttk.Label(self, text="Queue Analysis Calculations", font=controller.title_font, anchor=tk.CENTER)#, style='greyBackground.TLabel')
        label.grid(row=0, column=0, rowspan =1, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)

        self.rframe = ttk.Frame(self)#, style='greyBackground.TFrame')
        self.rframeshow = False

        #1-3 rows
        rown = 0
        self.rlabel1 = ttk.Label(self.rframe, text="Name: ")#, style='greyBackground.TLabel')
        self.rlabel1.grid(row=0, column=0, columnspan=1)
        self.rlabel1_AnswerVar = tk.StringVar()
        self.rlabel1_AnswerBox = ttk.Label(self.rframe, text="")#, style='greyBackground.TLabel')
        self.rlabel1_AnswerBox.grid(row=rown, column=1, columnspan=3)

        rown+=1

        self.rLabel1c = ttk.Label(self.rframe, text="Type: ")#, style='greyBackground.TLabel')
        self.rLabel1c.grid(row=rown, column=0, columnspan=1)
        self.rLabel1d = ttk.Label(self.rframe, text="")#, style='greyBackground.TLabel')
        self.rLabel1d.grid(row=rown, column=1, columnspan=3)        

        rown+=1

        self.rlabel2a = ttk.Label(self.rframe, text="Path: ")#, style='greyBackground.TLabel')#, wraplength=200)
        self.rlabel2a.grid(row=rown, column=0, columnspan=1)
        self.rlabel2 = ttk.Label(self.rframe, text="", wraplength=400)#, style='greyBackground.TLabel')
        self.rlabel2.grid(row=rown, column=1, columnspan=3)

        rown+=1

        self.rlabel3a = ttk.Label(self.rframe, text="Frame Number: ")#, style='greyBackground.TLabel')#, wraplength=200)
        self.rlabel3a.grid(row=rown, column=0, columnspan=1)
        self.rlabel3 = ttk.Label(self.rframe, text="")#, style='greyBackground.TLabel')#, wraplength=200)
        self.rlabel3.grid(row=rown, column=1, columnspan=3)

        rown+=1

        #4th row
        self.rlabel4 = ttk.Label(self.rframe, text="FPS: ")#, style='greyBackground.TLabel')
        self.rlabel4_AnswerVar = tk.StringVar()
        self.rlabel4_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel4_AnswerVar, validate="focusout", validatecommand=lambda: self.validateinteger(self.rlabel4_AnswerBox, self.rlabel4_AnswerVar, "FPS"))
        self.rlabel4.grid(row=rown, column=0, columnspan=1)
        self.rlabel4_AnswerBox.grid(row=rown, column=1, columnspan=1)

        self.rlabel5 = ttk.Label(self.rframe, text="Pixel Size: ")
        self.rlabel5_AnswerVar = tk.StringVar()
        self.rlabel5_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel5_AnswerVar, validate="focusout", validatecommand=lambda: self.validatefloat(self.rlabel5_AnswerBox, self.rlabel5_AnswerVar, "pixelsize"))
        self.rlabel5.grid(row=rown, column=2, columnspan=1)
        self.rlabel5_AnswerBox.grid(row=rown, column=3, columnspan=1)

        rown+=1

        self.separator_adv = ttk.Separator(self.rframe, orient=tk.HORIZONTAL)
        self.separator_adv.grid(row=rown,column=0,columnspan=4, sticky="ew") 
        rown+=1

        self.separator_lbl = ttk.Label(self.rframe,  text="Advanced Settings: ")#, style='greyBackground.TLabel')
        self.separator_lbl.grid(row=rown,column=0,columnspan=4) 
        rown+=1

        #5th row
        self.rlabel6 = ttk.Label(self.rframe, text="Pyr Scale: ")#, style='greyBackground.TLabel')
        self.rlabel6_AnswerVar = tk.StringVar()
        self.rlabel6_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel6_AnswerVar, validate="focusout", validatecommand=lambda: self.validatefloat(self.rlabel6_AnswerBox, self.rlabel6_AnswerVar, "pyr_scale"))
        self.rlabel6.grid(row=rown, column=0, columnspan=1)
        self.rlabel6_AnswerBox.grid(row=rown, column=1, columnspan=1)
        
        self.rlabel7 = ttk.Label(self.rframe, text="Levels: ")#, style='greyBackground.TLabel')
        self.rlabel7_AnswerVar = tk.StringVar()
        self.rlabel7_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel7_AnswerVar, validate="focusout", validatecommand=lambda: self.validateinteger(self.rlabel7_AnswerBox, self.rlabel7_AnswerVar, "levels"))
        self.rlabel7.grid(row=rown, column=2, columnspan=1)
        self.rlabel7_AnswerBox.grid(row=rown, column=3, columnspan=1)

        rown +=1

        #6th row
        self.rlabel8 = ttk.Label(self.rframe, text="Winsize: ")#, style='greyBackground.TLabel')
        self.rlabel8_AnswerVar = tk.StringVar()
        self.rlabel8_AnswerBox = tk.Entry(self.rframe, width=5, textvariable=self.rlabel8_AnswerVar, validate="focusout", validatecommand=lambda: self.validateinteger(self.rlabel8_AnswerBox, self.rlabel8_AnswerVar, "winsize"))
        self.rlabel8.grid(row=rown, column=0, columnspan=1)
        self.rlabel8_AnswerBox.grid(row=rown, column=1, columnspan=1)

        self.rlabel9 = ttk.Label(self.rframe, text="Iterations: ")#, style='greyBackground.TLabel')
        self.rlabel9_AnswerVar = tk.StringVar()
        self.rlabel9_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel9_AnswerVar, validate="focusout", validatecommand=lambda: self.validateinteger(self.rlabel9_AnswerBox, self.rlabel9_AnswerVar, "iterations"))
        self.rlabel9.grid(row=rown, column=2, columnspan=1)  
        self.rlabel9_AnswerBox.grid(row=rown, column=3, columnspan=1)

        rown+=1

        #7th row
        self.rlabel10 = ttk.Label(self.rframe, text="Poly N: ")#, style='greyBackground.TLabel')
        self.rlabel10_AnswerVar = tk.StringVar()
        self.rlabel10_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel10_AnswerVar, validate="focusout", validatecommand=lambda: self.validateinteger(self.rlabel10_AnswerBox, self.rlabel10_AnswerVar, "poly_n"))
        self.rlabel10.grid(row=rown, column=0, columnspan=1)
        self.rlabel10_AnswerBox.grid(row=rown, column=1, columnspan=1)

        self.rlabel11 = ttk.Label(self.rframe, text="Poly Sigma: ")#, style='greyBackground.TLabel')
        self.rlabel11_AnswerVar = tk.StringVar()
        self.rlabel11_AnswerBox = ttk.Entry(self.rframe, width=5, textvariable=self.rlabel11_AnswerVar, validate="focusout", validatecommand=lambda: self.validatefloat(self.rlabel11_AnswerBox, self.rlabel11_AnswerVar, "poly_sigma"))
        self.rlabel11.grid(row=rown, column=2, columnspan=1)
        self.rlabel11_AnswerBox.grid(row=rown, column=3, columnspan=1)

        for i in range(0,rown+1):
            self.rframe.rowconfigure(i, weight=1)
        for i in range(0,4):
            self.rframe.columnconfigure(i, weight=1)

        self.menu_index = None
        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label="Close")
        self.popup_menu.add_command(label="Rename",
                                    command=self.setName)
        self.popup_menu.add_command(label="Add to Preset",
                                    command=self.open_as_preset)
        self.popup_menu.add_command(label="Apply Preset",
                                    command=self.open_preset)
        self.popup_menu.add_command(label="Delete",
                                    command=self.delete_item)
        self.popup_menu.bind("<FocusOut>", self.focus_out_menu)

        #open folder method
        self.tframe = ttk.Frame(self)#, style='greyBackground.TFrame')
        scrollbar = ttk.Scrollbar(self.tframe, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(self.tframe, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE, width=50)
        
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.listbox.bind('<<ListboxSelect>>', self.onselect_event)
        self.listbox.bind("<Button-3>", self.show_focus_menu)

        self.tframe.grid(row=1, column=0, rowspan = 8, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        self.currentselind = None
        self.rframe.grid(row=1, column=2, rowspan = 8, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        self.rframe.grid_remove()
        self.tframe.grid(row=1, column=0, rowspan = 8, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)

        b1frame = ttk.Frame(self)
        b1lbl = ttk.Label(b1frame, text="Load Data")
        b1lbl.grid(row=0, column=1)

        button_addgroup = ttk.Button(b1frame, image=self.controller.openloaddialog,
                           command=self.select_dir)
        button_addgroup.image = self.controller.openloaddialog
        button_addgroup.grid(row=0, column=0)

        b1frame.grid(row=10, column=0, columnspan=1)

        CreateToolTip(button_addgroup, \
    "Loads new data.")

        b2frame = ttk.Frame(self)
        b2lbl = ttk.Label(b2frame, text="Delete All")
        b2lbl.grid(row=0, column=1)
        
        button_deletegroup2 = ttk.Button(b2frame, image=self.controller.deleteallimg,
           command=self.clear_all)
        button_deletegroup2.image=self.controller.deleteallimg
        button_deletegroup2.grid(row=0, column=0)

        b2frame.grid(row=10, column=1, columnspan=1)

        CreateToolTip(button_deletegroup2, \
    "Removes all current data selections.")


        b3frame = ttk.Frame(self)
        b3lbl = ttk.Label(b3frame, text="Run All")
        b3lbl.grid(row=0, column=1)

        button_rungroup = ttk.Button(b3frame, image=self.controller.runallimg,
                           command=self.run_groups)
        button_rungroup.image=self.controller.runallimg
        button_rungroup.grid(row=0, column=0)
        b3frame.grid(row=10, column=2, columnspan=2)

        CreateToolTip(button_rungroup, \
    "Starts data processing.")

        b4frame = ttk.Frame(self)
        b4lbl = ttk.Label(b4frame, text="Go to the start page")
        b4lbl.grid(row=0, column=1)

        button_go_start = ttk.Button(b4frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))
        button_go_start.image =self.controller.gotostartpage
        button_go_start.grid(row=0,column=0)
        b4frame.grid(row=12, column=1, columnspan=1)

        btnprogressframe = ttk.Frame(self)
        button_go_progresslbl = ttk.Label(btnprogressframe, text="Check Progress")
        button_go_progresslbl.grid(row=0, column=1)

        button_go_progress = ttk.Button(btnprogressframe, image=self.controller.progresspic,
                           command=lambda: controller.show_frame("PageTwo"))
        button_go_progress.image=self.controller.progresspic
        button_go_progress.grid(row=0, column=0)
        btnprogressframe.grid(row=12, column=3, columnspan=1)

        CreateToolTip(button_go_progress, \
    "Checks current data processing progress.")


    def focus_out_menu(self, event=None):
        self.popup_menu.grab_release()
        self.popup_menu.unpost()
        self.listbox.focus_set()
        self.controller.popuplock = False
        self.controller.currentpopup = None

    def show_focus_menu(self, event):
        index = event.widget.nearest(event.y)
        _, yoffset, _, height = event.widget.bbox(index)
        if event.y > height + yoffset + 5: # XXX 5 is a niceness factor :)
            # Outside of widget.
            self.menu_index = None
            return
        self.menu_index = index
        item = event.widget.get(index)
        try:
            abs_coord_x = self.controller.winfo_pointerx() - self.controller.winfo_vrootx()
            abs_coord_y = self.controller.winfo_pointery() - self.controller.winfo_vrooty()
            self.controller.popuplock = True
            self.controller.currentpopup = self.popup_menu
            self.popup_menu.tk_popup(int(abs_coord_x + np.max([(self.popup_menu.winfo_width()/2) + 10, 15])), abs_coord_y)
        finally:
            self.popup_menu.grab_release()

    def open_as_preset(self, event=None):
        new_preset_dict = {
            "FPS" : self.analysisgroups[self.menu_index].FPS,
            "pixelsize" : self.analysisgroups[self.menu_index].pixelsize,
            "pyr_scale" : self.analysisgroups[self.menu_index].pyr_scale,
            "levels" : self.analysisgroups[self.menu_index].levels,
            "winsize" : self.analysisgroups[self.menu_index].winsize,
            "iterations" : self.analysisgroups[self.menu_index].iterations,
            "poly_n" : self.analysisgroups[self.menu_index].poly_n,
            "poly_sigma" : self.analysisgroups[self.menu_index].poly_sigma
        }
        d = AddPresetDialog(self, title='Add Pre-Set value:', literals=[
            ("preset_dicts", self.controller.preset_dicts),
            ("new_preset", new_preset_dict)
        ])
        if d.result != None:
            self.controller.preset_dicts[d.result] =  new_preset_dict
            try:
                filehandler = open('userprefs/userpresets.pickle' , 'wb') 
                # pickle.dump(self.controller.preset_dicts, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.controller.preset_dicts, filehandler, protocol=3)
                filehandler.close()
                messagebox.showinfo("File Saved", "User Preset File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", "Could not save Preset file\n" + str(e))



    def open_preset(self, event=None):
        d = SelectPresetDialog(self, title='Select from a previous Pre-Set:', literals=[
            ("preset_dicts", self.controller.preset_dicts)
        ])
        if d.result != None:
            self.analysisgroups[self.menu_index].set_valtype("pyr_scale", d.result["pyr_scale"])
            self.analysisgroups[self.menu_index].set_valtype("levels", d.result["levels"])
            self.analysisgroups[self.menu_index].set_valtype("winsize", d.result["winsize"])
            self.analysisgroups[self.menu_index].set_valtype("iterations", d.result["iterations"])
            self.analysisgroups[self.menu_index].set_valtype("poly_n", d.result["poly_n"])
            self.analysisgroups[self.menu_index].set_valtype("poly_sigma", d.result["poly_sigma"])
            if self.analysisgroups[self.menu_index] == self.selectedgroup:
                self.listbox.select_set(self.menu_index) #Sets focus on item
                self.listbox.event_generate("<<ListboxSelect>>")

    def clear_all(self):
        self.listbox.delete(0,'end')
        self.selectedgroup = None
        self.analysisgroups = []
        self.rframe.grid_remove()
        self.rframeshow = False
        self.tframe.grid(row=1, column=0, rowspan = 8, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)

    def delete_item(self):
        if self.controller.current_frame.fname == self.fname:
            if self.menu_index == None:
                self.listbox.delete(tk.ANCHOR)
                del self.analysisgroups[self.selectedgroup.lindex]
                self.selectedgroup = None
                self.rframe.grid_remove()
                self.rframeshow = False
            else:
                self.listbox.delete(self.menu_index)
                del self.analysisgroups[self.menu_index]
                self.selectedgroup = None
                self.rframe.grid_remove()
                self.rframeshow = False
                self.menu_index = None


    def setName(self, event=None):
        self.selectedgroup = self.analysisgroups[self.menu_index]
        d = SelectMenuItem(self, title='Edit Name', literals=[
            ("current_name", self.analysisgroups[self.menu_index].name)
        ])
        if d.result != None:
            self.listbox.delete(self.menu_index)
            self.listbox.insert(self.menu_index, d.result)
            self.analysisgroups[self.menu_index].name = d.result
            if self.selectedgroup == self.analysisgroups[self.menu_index]:
                self.rlabel1_AnswerBox['text'] = str(self.analysisgroups[self.menu_index].name)
        self.menu_index = None

    def validateinteger(self, eventsource, varthis, valtype):
        if self.controller.current_frame.fname == self.fname:
            strval = varthis.get()
            try:
                strval = int(strval)
                if default_values_bounds[valtype]:
                    prev_value = self.selectedgroup.get_valtype(valtype)
                    if strval <= default_values_bounds[valtype][0] or strval >= default_values_bounds[valtype][1]:
                        eventsource.delete(0,tk.END)
                        eventsource.insert(0,str(prev_value))
                        return False
                self.selectedgroup.set_valtype(valtype, strval)
                self.analysisgroups[self.currentselind].set_valtype(valtype, strval)
            except Exception:
                return False
            return True

    def validatefloat(self, eventsource, varthis, valtype):
        if self.controller.current_frame.fname == self.fname:
            strval = varthis.get()
            try:
                strval = float(strval)
                if default_values_bounds[valtype]:
                    prev_value = self.selectedgroup.get_valtype(valtype)
                    if strval <= default_values_bounds[valtype][0] or strval >= default_values_bounds[valtype][1]:
                        eventsource.delete(0,tk.END)
                        eventsource.insert(0,str(prev_value))
                        return False
                self.selectedgroup.set_valtype(valtype, strval)
                self.analysisgroups[self.currentselind].set_valtype(valtype, strval)
            except Exception:
                return False
            return True

    def select_dir(self):
        if self.controller.current_frame.fname == self.fname:
            #Ask here for type
            d = FolderSelectDialog(self, title='Select Input Data Type:')
            global default_values
            if d.result == "Folder":
                folder_selected = filedialog.askdirectory(title="Select Image Directory:")
                folder_selected = r'%s' %folder_selected
                if folder_selected:
                    default_name = os.path.basename(folder_selected)
                    newanalysis = AnalysisGroup(name=default_name, gpath=folder_selected, gtype="Folder")
                    newanalysis.lindex = self.listbox.size()
                    newanalysis.set_valtype("FPS", default_values["FPS"])
                    newanalysis.set_valtype("pixelsize", default_values["pixelsize"])
                    newanalysis.set_valtype("pyr_scale", default_values["pyr_scale"])
                    newanalysis.set_valtype("levels", default_values["levels"])
                    newanalysis.set_valtype("winsize", default_values["winsize"])
                    newanalysis.set_valtype("iterations", default_values["iterations"])
                    newanalysis.set_valtype("poly_n", default_values["poly_n"])
                    newanalysis.set_valtype("poly_sigma", default_values["poly_sigma"])
                    if newanalysis.framenumber >= 2:
                        self.analysisgroups.append(newanalysis)
                        self.listbox.insert(tk.END, default_name)
                        self.listbox.select_set(tk.END) #This only sets focus on the first item.
                        self.listbox.event_generate("<<ListboxSelect>>")
                    else:
                        messagebox.showwarning(
                            "Bad input",
                            "Folder has less than 2 Valid image files"
                        )
            elif d.result == "Video":
                filename = filedialog.askopenfilename(title = "Select Video File:",filetypes = (("Audio Video Interleave","*.avi"),("all files","*.*")))
                filename = r'%s' %filename
                if filename:
                    default_name = os.path.basename(filename)
                    newanalysis = AnalysisGroup(name=default_name, gpath=filename, gtype="Video")
                    newanalysis.lindex = self.listbox.size()
                    newanalysis.set_valtype("FPS", get_Video_FPS(filename))
                    newanalysis.set_valtype("pixelsize", default_values["pixelsize"])
                    newanalysis.set_valtype("pyr_scale", default_values["pyr_scale"])
                    newanalysis.set_valtype("levels", default_values["levels"])
                    newanalysis.set_valtype("winsize", default_values["winsize"])
                    newanalysis.set_valtype("iterations", default_values["iterations"])
                    newanalysis.set_valtype("poly_n", default_values["poly_n"])
                    newanalysis.set_valtype("poly_sigma", default_values["poly_sigma"])
                    if newanalysis.framenumber >= 2:
                        self.analysisgroups.append(newanalysis)
                        self.listbox.insert(tk.END, default_name)
                        self.listbox.select_set(tk.END) #This only sets focus on the first item.
                        self.listbox.event_generate("<<ListboxSelect>>")
                    else:
                        messagebox.showwarning(
                            "Bad input",
                            "Video has less than 2 Valid Frames"
                        )
            elif d.result == "Tiff Directory":
                filename = filedialog.askopenfilename(title = "Select TIFF Directory File:",filetypes = (("TIFF Files","*.tiff"),("TIF Files","*.tif"),("all files","*.*")))
                filename = r'%s' %filename
                if filename:
                    default_name = os.path.basename(filename)
                    newanalysis = AnalysisGroup(name=default_name, gpath=filename, gtype="Tiff Directory")
                    newanalysis.lindex = self.listbox.size()
                    newanalysis.set_valtype("FPS", default_values["FPS"])
                    newanalysis.set_valtype("pixelsize", default_values["pixelsize"])
                    newanalysis.set_valtype("pyr_scale", default_values["pyr_scale"])
                    newanalysis.set_valtype("levels", default_values["levels"])
                    newanalysis.set_valtype("winsize", default_values["winsize"])
                    newanalysis.set_valtype("iterations", default_values["iterations"])
                    newanalysis.set_valtype("poly_n", default_values["poly_n"])
                    newanalysis.set_valtype("poly_sigma", default_values["poly_sigma"])
                    if newanalysis.framenumber >= 2:
                        self.analysisgroups.append(newanalysis)
                        self.listbox.insert(tk.END, default_name)
                        self.listbox.select_set(tk.END) #This only sets focus on the first item.
                        self.listbox.event_generate("<<ListboxSelect>>")
                    else:
                        messagebox.showwarning(
                            "Bad input",
                            "TIFF Directory has less than 2 images"
                        )

    def onselect_event(self, event):
        if self.controller.current_frame.fname == self.fname:
            try:
                if len(self.listbox.curselection()) > 0:
                    curnamesel = self.listbox.curselection()[-1]
                    self.currentselind = curnamesel
                    self.selectedgroup = self.analysisgroups[curnamesel]
                    self.update_rframe()
            except IndexError:
                pass

    def update_rframe(self):
        if self.controller.current_frame.fname == self.fname:
            if self.rframeshow == False:
                self.rframe.grid()
                self.tframe.grid(row=1, column=0, rowspan = 8, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
                self.rframeshow = True
            self.rlabel1_AnswerBox['text'] = str(self.selectedgroup.name)
            self.rLabel1d['text'] = str(self.selectedgroup.gtype)
            self.rlabel2['text'] = str(self.selectedgroup.gpath)
            self.rlabel3['text'] = str(self.selectedgroup.framenumber)
            self.rlabel4_AnswerBox.delete(0,tk.END)
            self.rlabel4_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("FPS")))
            if self.selectedgroup.gtype == "Video":
                self.rlabel4_AnswerBox['state'] = 'disabled'
            else:
                self.rlabel4_AnswerBox['state'] = 'normal'
            self.rlabel5_AnswerBox.delete(0,tk.END)
            self.rlabel5_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("pixelsize")))
            self.rlabel6_AnswerBox.delete(0,tk.END)
            self.rlabel6_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("pyr_scale")))
            self.rlabel7_AnswerBox.delete(0,tk.END)
            self.rlabel7_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("levels")))
            self.rlabel8_AnswerBox.delete(0,tk.END)
            self.rlabel8_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("winsize")))
            self.rlabel9_AnswerBox.delete(0,tk.END)
            self.rlabel9_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("iterations")))
            self.rlabel10_AnswerBox.delete(0,tk.END)
            self.rlabel10_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("poly_n")))
            self.rlabel11_AnswerBox.delete(0,tk.END)
            self.rlabel11_AnswerBox.insert(0,str(self.selectedgroup.get_valtype("poly_sigma")))
    
    def delay_group(self):
        pass

    def run_groups(self):
        if self.listbox.size() > 0:
            MsgBox = messagebox.askyesno(title='Run Groups', message="Run all Groups in Listbox?")
            if MsgBox == True:
                global ncores
                if self.controller.queuestarted == False:
                    ncores = multiprocessing.cpu_count()
                self.controller.queuestarted = True
                # wd = WaitDialogProgress(self, title='Queueing Groups...')
                self.controller.showwd()
                # self.controller.update()
                # wd.progress_bar.start()
                for g_index in range(self.listbox.size()):
                    self.analysisgroups[g_index].saverun = True
                    addqueue(self.analysisgroups[g_index])
                    time.sleep(1)
                self.controller.checkTheQueue()
                self.controller.cancelwd()
                # wd.cancel()
                self.controller.show_frame("PageTwo")
        else:
            messagebox.showwarning(
                "Bad input",
                "No Groups selected"
            )
            return

    def run_group(self):
        global ncores
        if self.controller.current_frame.fname == self.fname:
            if len(self.listbox.curselection()) > 0:
                if self.controller.queuestarted == False:
                    ncores = multiprocessing.cpu_count()
                self.controller.queuestarted = True
                for g_index in self.listbox.curselection():
                    current_group_to_queue = self.analysisgroups[g_index]
                    MsgBox = messagebox.askyesno(title='Optical Flow Save Config.', message="Save: '" + self.analysisgroups[g_index].name + "' on disk after running?")
                    if MsgBox == True:
                        self.analysisgroups[g_index].saverun = True
                    addqueue(self.analysisgroups[g_index])
                self.controller.checkTheQueue()

            elif self.selectedgroup:
                MsgBox = messagebox.askyesno(title='Optical Flow Save Config.', message="Save: '" + self.selectedgroup.name + "' on disk after running?")
                if MsgBox == True:
                    self.selectedgroup.saverun = True
                if self.controller.queuestarted == False:
                    ncores = multiprocessing.cpu_count()
                    self.controller.queuestarted = True
                addqueue(self.selectedgroup)
                self.controller.queuestarted = True
                self.controller.checkTheQueue()
            else:
                messagebox.showwarning(
                    "Bad input",
                    "No Groups selected"
                )
                return

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        menubar.add_cascade(label="File", menu=pageMenu)
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)

        return menubar

class PageTwo(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageTwo"
        for i in range(0,10):
            self.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.columnconfigure(i, weight=1)
            
        label = ttk.Label(self, text="Current Calculations Progress", font=controller.title_font)#, style='greyBackground.TLabel')
        label.grid(row=0, column=1, rowspan=1)

        self.stamps = []
        self.stamp_dict = {}
        self.stamp_dict2 = {}
        self.n_stamp = {}
        self.timelbls = {}

        self.sframe = ttkScrollFrame(self, 'greyBackground.TFrame', self.controller.bgcolor) #add scroll bar

        self.sframe.viewPort.columnconfigure(0, weight=1)
        self.sframe.grid(row=1, column=0, rowspan=8, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        for i in range(16):
            tbgfrn = 'greyBackground.TFrame'
            tbglbl = 'greyBackground.TLabel'
            if i % 2 == 0:
                tbgfrn = 'whiteBackground.TFrame'
                tbglbl = 'whiteBackground.TLabel'
            a = ttk.Frame(self.sframe.viewPort, style=tbgfrn)
            l = ttk.Label(a, text="  ", style=tbglbl)
            l.grid(row=0, rowspan=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
            a.grid(row=i, rowspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.sframe.viewPort.rowconfigure(i, weight=1)
        self.defaultpattern = True

        btn7frame = ttk.Frame(self)
        btn7lbl =  ttk.Label(btn7frame, text="Go to the start page")
        btn7lbl.grid(row=0, column=1)

        button_go_start = ttk.Button(btn7frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))#, style='greyBackground.TButton')
        button_go_start.image=self.controller.gotostartpage
        button_go_start.grid(row=0, column=0)
        btn7frame.grid(row=9, column=1)

    def removedefault(self):
        self.controller.btn_lock = True
        if self.defaultpattern == True:
            for child in self.sframe.viewPort.winfo_children():
                child.destroy()
            self.defaultpattern = False
        self.controller.btn_lock = False

    def click_event(self, event=None, n=None, argsp=None):
        self.controller.btn_lock = True
        if self.controller.current_frame.fname == self.fname:
            event = argsp[0]
            n = event.widget.n
            r = n
            s = self.n_stamp[r]
            p = self.stamp_dict2[s].get()
            if p == 1.0:
                g = self.controller.done_groups[s]
                MsgBox = messagebox.askyesno(title='Start Analysis', message="Start Analysis on: '" + g.name + "'?")
                if MsgBox == True:
                    self.controller.current_analysis = g
                    self.controller.btn_lock = False
                    self.controller.show_frame("PageFour")
            else:
                messagebox.showerror("Error", "Group Analysis is not done.")
                self.controller.btn_lock = False

    def add_new_progressframe(self):
        global stamp_to_group, progress_tasks
        rown = 0
        for k in stamp_to_group.keys():
            if k not in self.stamps:
                try: #new group, create new row with progress bar in ttkFrame
                    progress = progress_tasks[k]
                    group = stamp_to_group[k]
                    remainingtime = tasks_time[k]
                    remainingtime2 = "Time elapsed: " + str(remainingtime[0]) + "(s), Time left: " + str(remainingtime[1]) + "(s), Estimated Finish Time: "+ str(remainingtime[2])
                    tbg = "#d3d3d3"
                    if rown % 2 == 0:
                        tbg = "#ffffff"
                    framegroup = tk.Frame(self.sframe.viewPort, background=tbg)
                    framegroup.grid(row=rown, rowspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

                    lbl1 = tk.Label(framegroup, text=group.name, background=tbg)
                    lbl1.grid(row=0, column=0)
                    
                    var_barra = tk.DoubleVar()
                    var_barra.set(progress)
                    minha_barra = ttk.Progressbar(framegroup, style="", length=300, variable=var_barra, maximum=1)
                    minha_barra.grid(row=0, column=1)
                    
                    timelbl = tk.Label(framegroup, text=remainingtime2, background=tbg)
                    timelbl.grid(row=0, column=2)

                    btn6frame = tk.Frame(framegroup, background=tbg)
                    btn6lbl = tk.Label(btn6frame, text="Analysis", background=tbg)
                    btn6lbl.grid(row=0, column=1)

                    button_go_analysis = tk.Button(btn6frame, image=self.controller.startanalysis, foreground=tbg)
                    button_go_analysis.image=self.controller.startanalysis

                    button_go_analysis.n = rown
                    button_go_analysis.grid(row=0, column=0)
                    button_go_analysis.bind("<Button-1>", lambda *args: self.click_event(n=rown, argsp=args))

                    # button_go_analysis.grid(row=0, column=3)
                    btn6frame.grid(row=0, column=3)


                    for i in range(0,1):
                        framegroup.rowconfigure(i, weight=1)
                    for i in range(0,4):
                        framegroup.columnconfigure(i, weight=1)
                    self.sframe.viewPort.rowconfigure(rown, weight=1)

                    self.stamp_dict[k] = minha_barra
                    self.stamp_dict2[k] = var_barra
                    self.timelbls[k] = timelbl
                    self.n_stamp[rown] = k
                    self.stamps.append(k)
                    rown += 1
                except KeyError:
                    pass
            else: #existing group, update row with progress bar in ttkFrame
                progress = progress_tasks[k]
                remainingtime = tasks_time[k]
                remainingtime2 = "Time elapsed: " + str(remainingtime[0]) + "(s), Time left: " + str(remainingtime[1]) + "(s), Estimated Finish Time: "+ str(remainingtime[2])
                var_barra = self.stamp_dict2[k]
                var_barra.set(progress)
                # self.controller.update()
                self.timelbls[k]['text'] = remainingtime2
                rown += 1

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        menubar.add_cascade(label="File", menu=pageMenu)
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)
        return menubar

class PageThree(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageThree"
        for i in range(0,14):
            self.rowconfigure(i, weight=1)
        for i in range(0,5):
            self.columnconfigure(i, weight=1)
            
        label = ttk.Label(self, text="Select a Group for Analysis:", font=controller.title_font)#, style='greyBackground.TLabel')
        label.grid(row=0, column=1, columnspan=3)

        self.all_list = []
        self.selectedgroup = None

        #include flow edit stuff
        tframe = ttk.Frame(self)#, style='greyBackground.TFrame')
        self.rscrollbar = ttk.Scrollbar(tframe, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(tframe, yscrollcommand=self.rscrollbar.set, selectmode=tk.SINGLE)
        
        self.rscrollbar.config(command=self.listbox.yview)
        self.rscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


        self.listbox.bind('<<ListboxSelect>>', self.onselect_event)

        tframe.grid(row=1, column=1, rowspan = 4, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        self.tabletreeframe = ttk.Frame(self)
        cols = ('Time(' + self.controller.current_timescale + ')' , 'Average Raw Speed('+self.controller.current_speedscale+')')
        self.tabletree = ttk.Treeview(self.tabletreeframe, columns=cols, show='headings')
        
        self.tabletree.tag_configure('oddrow', background='#d3d3d3')
        self.tabletree.tag_configure('evenrow', background='white')

        self.vsb = ttk.Scrollbar(self.tabletreeframe, orient="vertical", command=self.tabletree.yview)

        for col in cols:
            self.tabletree.heading(col, text=col)

        self.tabletree.configure(yscrollcommand=self.vsb.set)

        self.tabletree.pack(side='left', fill="both", expand=True)
        self.vsb.pack(side='right', fill='y')

        self.tabletreeframe.grid(row=6, column=1, rowspan = 4, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        self.data_s_index = None
        self.data_e_index = None
        self.y0 = None
        self.y1 = None
        self.current_down_data = [[],[]]
        

        self.btn1frame =  ttk.Frame(self)
        btn1lbl = ttk.Label(self.btn1frame, text="Download Table")
        btn1lbl.grid(row=0, column=1)
        self.button_download_table = ttk.Button(self.btn1frame, image=self.controller.downloadtableimg,
                           command=self.download_table)
        self.button_download_table.image=self.controller.downloadtableimg
        self.button_download_table.grid(row=0, column=0, columnspan=1)
        # self.button_download_table.grid(row=11, column=1, columnspan=1)
        self.btn1frame.grid(row=11, column=2, columnspan=1)

        CreateToolTip(self.button_download_table, \
    "Downloads current table.")

        btn2frame =  ttk.Frame(self)
        btn2lbl = ttk.Label(btn2frame, text="Start Analysis")
        btn2lbl.grid(row=0, column=1)
        button_go_run = ttk.Button(btn2frame, image=self.controller.startanalysis,
                           command=self.start_analysis)
        button_go_run.image=self.controller.startanalysis
        button_go_run.grid(row=0, column=0, columnspan=1)
        # button_go_run.grid(row=11, column=2, columnspan=1)
        btn2frame.grid(row=11, column=3, columnspan=1)

        CreateToolTip(button_go_run, \
    "Starts Analysis on a selected group.")

        btn3frame =  ttk.Frame(self)
        btn3lbl = ttk.Label(btn3frame, text="Go to the start page")
        btn3lbl.grid(row=0, column=1)
        button_go_start = ttk.Button(btn3frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))
        button_go_start.image=self.controller.gotostartpage
        button_go_start.grid(row=0, column=0, columnspan=1)
        # button_go_start.grid(row=11, column=3, columnspan=1)
        btn3frame.grid(row=11, column=1, columnspan=1)

    def update_headings(self):
        self.controller.btn_lock = True
        self.tabletree.heading('#1', text='Time(' + self.controller.current_timescale + ')')
        self.tabletree.heading('#2', text='Average Raw Speed('+self.controller.current_speedscale+')')
        self.controller.btn_lock = False

    def generate_default_table(self):
        self.controller.btn_lock = True
        self.update_headings()
        self.btn1frame.grid_remove()

        self.tabletree.delete(*self.tabletree.get_children())
        self.vsb.pack_forget()
        for i in range(20): #Rows
            cur_tag = 'oddrow'
            if i % 2 == 0:
                cur_tag = 'evenrow'
            self.tabletree.insert("", "end", values=("  ", ""), tags = (cur_tag,))
        self.controller.btn_lock = False

    def onselect_event(self, event=None):
        self.controller.btn_lock = True

        self.data_s_index = None
        self.data_e_index = None
        self.current_down_data = [[],[]]

        self.btn1frame.grid_remove()
        self.tabletree.delete(*self.tabletree.get_children())
        self.tabletree.tag_configure('oddrow', background='#d3d3d3')
        self.tabletree.tag_configure('evenrow', background='white')
        self.vsb.pack_forget()
        try:
            curnamesel = self.listbox.curselection()[0]
            self.selectedgroup = self.all_list[curnamesel]
            xdata = [float(i / self.selectedgroup.FPS) for i in range(len(self.selectedgroup.mag_means))]
            if self.controller.current_timescale == "ms":
                xdata = [a * 1000 for a in xdata]
            xdata = [float("{:.3f}".format(a)) for a in xdata]
            ydata = self.selectedgroup.mag_means.copy()
            for i in range(len(xdata)): #Rows
                cur_tag = 'oddrow'
                if i % 2 == 0:
                    cur_tag = 'evenrow'
                self.tabletree.insert("", "end", values=(xdata[i], ydata[i]), tags = (cur_tag,))
            self.btn1frame.grid()
            self.vsb.pack(side=tk.RIGHT, fill=tk.Y)

        except IndexError:
            self.vsb.grid_forget()
            self.btn1frame.grid_remove()
            self.generate_default_table()
            #hide download table button    
        self.controller.btn_lock = False

    def download_table(self):
        self.controller.btn_lock = True
        times = [float("{:.3f}".format(float(i / self.selectedgroup.FPS))) for i in range(len(self.selectedgroup.mag_means))]
        if self.controller.current_timescale == "ms":
            times = [a * 1000 for a in times]
        SaveTableDialog(self, title='Save Table', literals=[
            ("headers", ["Time(" + self.controller.current_timescale + ")", "Average Raw Speed("+self.controller.current_speedscale+")"]),
            ("data", [times, self.selectedgroup.mag_means.copy()]),
            ("data_t", "single")
            ])
        self.controller.btn_lock = False

    def start_analysis(self):
        if self.controller.btn_lock == False:
            self.controller.btn_lock = True
            if self.selectedgroup:
                MsgBox = messagebox.askyesno(title='Start Analysis', message="Start Analysis on: '" + self.selectedgroup.name + "'?")
                if MsgBox == True:
                    self.controller.current_analysis = self.selectedgroup
                    self.controller.btn_lock = False
                    self.controller.show_frame("PageFour")
            self.controller.btn_lock = False

    def listboxpopulate(self):
        self.controller.btn_lock = True
        self.listbox.delete(0, tk.END)
        disk_list = []
        if os.path.exists('savedgroups'):
            disk_groups = [x for x in os.listdir('savedgroups') if os.path.isdir(x) == False and str(x).lower().endswith("pickle")]
            if disk_groups:
                for fdg in disk_groups:
                    filehandler2 = open('savedgroups/'+ fdg, 'rb')
                    diskgroup = pickle.load(filehandler2)
                    filehandler2.close()
                    disk_list.append(diskgroup)
        memory_list = []
        memory_list = self.controller.done_groups.values()
        
        self.all_list = []
        for eg in disk_list:
            self.listbox.insert(tk.END, eg.name + " - Frame Num: " + str(eg.framenumber) + " (Disk)")
            self.all_list.append(eg)
        for eg in memory_list:
            self.listbox.insert(tk.END, eg.name + " - Frame Num: " + str(eg.framenumber) + " (Memory)")
            self.all_list.append(eg)
        self.controller.btn_lock = False

    def exportselectedtable(self):
        self.controller.btn_lock = True
        if self.selectedgroup:
            self.download_table()
            self.controller.btn_lock = True
        else:
            messagebox.showerror("Error", "No Group Selected for Exporting")
        self.controller.btn_lock = False

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)

        exportMenu = tk.Menu(menubar, tearoff=0)
        
        exportMenu.add_command(label="Export Current Table", command=self.exportselectedtable)#, command=lambda: self.controller.show_frame("StartPage"))

        menubar.add_cascade(label="File", menu=pageMenu)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_cascade(label="Export", menu=exportMenu)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)

        return menubar

class PageFour(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageFour"
        self.case = "Test"
        self.current_case = [0.04056281,0.048919737,0.04836604,0.6572696,2.2677665,1.831524,1.2725116,0.8737864,0.63842714,0.47530872,0.3679374,0.28984872,0.20926978,0.16456017,0.11686811,0.09576731, 0.20527671,0.40098408,0.714253,1.1283345,1.3859935,1.3168406,0.97422427,0.6156282,0.4179357,0.29803818,0.2616953,0.21573865,0.1791429,0.16767764,0.147474, 0.12007897,0.10805313,0.09476269,0.08185742,0.07817462,0.077601366,0.070594124,0.062177893]
        self.old_current_case = self.current_case.copy()
        self.current_case_frames = list(range(len(self.current_case)))
        self.delta = None
        self.adjustnoisevar = True
        self.prevrenoise = None
        self.usernoise = None
        self.hidedots = True
        self.decreasenoise = False
        self.userdecreasenoise = False
        self.denoising = None
        self.dotsize = None
        self.double_dotsize = None

        nargs = noise_detection(self.current_case,filter_noise_area=True, added_noise_dots=[], removed_noise_dots=[], cutoff_val=0.90)

        points, self.noises, exponential_pops, conditions = peak_detection(self.current_case, nargs=nargs)
        self.stop_condition_perc = None
        self.gvf_cutoff = None
        self.plotsettings = self.controller.plotsettings
        
        self.FPS = 2.0
        self.pixel_val = None

        for i in range(0,12):
            self.rowconfigure(i, weight=1)
        for i in range(0,5):
            self.columnconfigure(i, weight=1)

        label = ttk.Label(self, text="Wave Definition and Selection", font=controller.title_font, anchor=tk.CENTER)#, style="greyBackground.TLabel")
        label.grid(row=0, column=1, columnspan=3)

        #create top frame
        self.frame1 = ttk.Frame(self)#, style="greyBackground.TFrame")

        lbl1 = ttk.Label(self.frame1, text= 'Delta: ')#, style="greyBackground.TLabel")
        lbl1.grid(row=0, column=0)

        self.spin_deltavalue = tk.Spinbox(self.frame1, from_=0, to=9999999999999, increment=1.0, width=10, command=self.update_with_delta_freq)
        self.spin_deltavalue.grid(row=0, column=1)
        CreateToolTip(self.spin_deltavalue, \
    "Speed difference for a given Maximum and a following Minimum Plot Points to be valid. "
    "Necessary for the Wave Detection algorithm. ")

        lbl1_5 = ttk.Label(self.frame1, text= 'Noise Cutoff: ')#, style="greyBackground.TLabel")
        lbl1_5.grid(row=0, column=2)

        self.spin_cutoff = tk.Spinbox(self.frame1, from_=0, to=9999999999999, increment=1.0, width=10, command=self.update_with_delta_freq)
        self.spin_cutoff.grid(row=0, column=3)
        CreateToolTip(self.spin_cutoff, \
    "Data below this Speed threshold are considered as starter noise points. "
    "Necessary for the Noise Detection algorithm. ")

        e2 = 1
        if self.decreasenoise == False:
            e2 = 0
        self.check_decrease_value = tk.IntVar(value=e2)
        self.checkdecrease = ttk.Checkbutton(self.frame1, text = "Decrease Avg. Noise", variable = self.check_decrease_value, \
                         onvalue = 1, offvalue = 0, command=self.update_with_delta_freq)
        self.checkdecrease.grid(row=0, column=4)
        CreateToolTip(self.checkdecrease, \
    "Average starter noise points value is decreased from plot.")

        lbl2 = ttk.Label(self.frame1, text= 'Exp. Stop Freq: ')#, style="greyBackground.TLabel")
        lbl2.grid(row=0, column=5)

        self.spin_stopcondition = tk.Spinbox(self.frame1, from_=0, to=1, increment=0.05, width=10, command=self.update_with_delta_freq)
        self.spin_stopcondition.grid(row=0,column=6)
        CreateToolTip(self.spin_stopcondition, \
    "Minimum ratio between a given Data point and it's previous neighbouring point in the Exponential Regression "
    "for finding the last point of a Wave. ")

        self.frame1.grid(row=1, column=1, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        for i in range(0,1):
            self.frame1.rowconfigure(i, weight=1)
        for i in range(0,7):
            self.frame1.columnconfigure(i, weight=1)

        self.spin_deltavalue.delete(0,"end")
        self.spin_deltavalue.insert(0,0.4)

        self.spin_stopcondition.delete(0,"end")
        self.spin_stopcondition.insert(0,0.4)

        self.spin_cutoff.delete(0,"end")
        self.spin_cutoff.insert(0,0.4)

        self.spin_deltavalue.bind("<Return>",self.update_with_delta_freq)
        self.spin_stopcondition.bind("<Return>",self.update_with_delta_freq)
        self.spin_cutoff.bind("<Return>",self.update_with_delta_freq)

        self.mainplotartist = None
        self.f_points = None
        self.s_f_points = None
        self.t_points = None
        self.l_points = None
        self.points = None

        self.noise_line = None
        self.noise_line_ax2 = None

        self.regression_list = []
        self.regression_list2 = []

        self.fig = plt.figure(figsize=(5, 4), dpi=100, facecolor=self.controller.bgcolor)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.90)

        self.gs = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.3)
        self.gs2 = gridspec.GridSpec(2, 1, height_ratios=[5, 5], hspace=0.5)

        self.ax = self.fig.add_subplot(self.gs[0])
        self.ax.set_navigate(False)
        self.axbaseline = None
        self.axgrid = None

        self.ax2 = self.fig.add_subplot(self.gs2[1])
        self.ax2.set_visible(False)
        self.ax2baseline = None
        self.ax2grid = None

        self.frame3 = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame3)  # A tk.DrawingArea.

        self.selectCMenu = tk.Menu(self, tearoff=0)
        self.selectCMenu.add_command(label="Close Menu")
        self.selectCMenu.add_command(label="Add Dot", command=lambda:self.local_peak_operation(txt="add_dot"))
        self.selectCMenu.add_command(label="Change Dot Type", command=lambda:self.local_peak_operation(txt="change_dot"))
        self.selectCMenu.add_command(label="Remove Dot", command=lambda:self.local_peak_operation(txt="remove_dot"))

        self.areaCMenu = tk.Menu(self.frame3, tearoff=0)
        self.areaCMenu.add_command(label="Close Menu")
        self.areaCMenu.add_command(label="Savitzky-Golay Filter", command=lambda:self.local_peak_operation(txt="denoise_sav"))
        self.areaCMenu.add_command(label="Averaged Window Filter", command=lambda:self.local_peak_operation(txt="denoise_avg"))
        self.areaCMenu.add_command(label="Set as Current Data", command=lambda:self.local_peak_operation(txt="set_current"))
        # self.areaCMenu.add_command(label="Delete from Data", command=lambda:self.local_peak_operation(txt="del_current"))
        self.areaCMenu.add_command(label="Edit values to 0.0 (Noise)", command=lambda:self.local_peak_operation(txt="edit_current"))
        self.areaCMenu.bind("<FocusOut>", self.popupFocusOut)

        self.areaNMenu=tk.Menu(self.frame3, tearoff=0)
        self.areaNMenu.add_command(label="Close Menu")
        self.areaNMenu.add_command(label="Set as Noise", command=lambda:self.local_peak_operation(txt="add_as_noise"))
        self.areaNMenu.add_command(label="Set as Wave",  command=lambda:self.local_peak_operation(txt="remove_as_noise"))
        self.areaNMenu.bind("<FocusOut>", self.popupNFocusOut)

        self.selectCMenu.bind("<FocusOut>", self.popupCFocusOut)

        self.dragDots = MoveDragHandler(master=self.controller,currentgroup=self.controller.current_analysis, FPS=self.FPS, pixel_val=self.pixel_val, figure=self.fig, ax=self.ax, data=self.current_case, selectCMenu=self.selectCMenu, areaCMenu=self.areaCMenu, areaNMenu=self.areaNMenu, colorify=self.plotsettings.peak_plot_colors, plotconf=self.plotsettings.plotline_opts, noiseindexes=self.noises[3], dsizes=(self.dotsize, self.double_dotsize),ax2baseline=self.ax2baseline,ax2grid=self.ax2grid)
        self.dragDots.ax2_type = "Zoom"

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame3.grid(row=2, column=1, rowspan=8, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame3.grid(row=2, column=1, rowspan=8, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.canvas.draw()

        self.frame4 = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.CheckVar_dots = tk.IntVar()

        CD = ttk.Checkbutton(self.frame4, text = "Plot Dots", variable = self.CheckVar_dots, \
                         onvalue = 1, offvalue = 0, command = self.plotDots, \
                         width = 20)#, style="greyBackground.TCheckbutton")
        self.CheckVar_dots.set(0)
        CD.grid(row=0,column=0)
        CreateToolTip(CD, \
    "Plots Wave Points.")

        self.CheckVar2 = tk.IntVar()
        C2 = ttk.Checkbutton(self.frame4, text = "Plot Noise Max Line", variable = self.CheckVar2, \
                         onvalue = 1, offvalue = 0, command = self.plotMeanNoise, \
                         width = 20)#, style="greyBackground.TCheckbutton")
        C2.grid(row=0, column=1)
        CreateToolTip(C2, \
    "Plots maximum value of starter noise points.")


        self.CheckVar1 = tk.IntVar()
        C1 = ttk.Checkbutton(self.frame4, text = "Plot Exp. Regressions", variable = self.CheckVar1, \
                         onvalue = 1, offvalue = 0, command = self.plotRegressions, \
                         width = 20)#, style="greyBackground.TCheckbutton")
        C1.grid(row=0,column=2)
        CreateToolTip(C1, \
    "Plots Exponential Regessions used to find the final point of all Waves.")

        for i in range(0,1):
            self.frame4.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.frame4.columnconfigure(i, weight=1)
        self.frame4.grid(row=10, column=1, rowspan=1, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        self.runupdate()

        self.frame5 = ttk.Frame(self)#, style="greyBackground.TFrame")

        btn1frame = ttk.Frame(self.frame5)
        btn1lbl = ttk.Label(btn1frame, text="Go back")
        btn1lbl.grid(row=0, column=1)
        button_go_back = ttk.Button(btn1frame, image=self.controller.goback,
                           command=lambda: controller.show_frame("PageThree"))#, style="greyBackground.TButton")
        button_go_back.image=self.controller.goback
        button_go_back.grid(row=0, column=0)
        btn1frame.grid(row=0, column=1)

        btn2frame = ttk.Frame(self.frame5)
        btn2lbl = ttk.Label(btn2frame, text="Analyse Wave Areas")
        btn2lbl.grid(row=0, column=1)
        button_go_analysis = ttk.Button(btn2frame, image=self.controller.analysewvareas,
                           command=self.analysepeakareas)#, style="greyBackground.TButton")
        button_go_analysis.image=self.controller.analysewvareas
        button_go_analysis.grid(row=0, column=0)
        btn2frame.grid(row=0, column=2)

        CreateToolTip(button_go_analysis, \
    "Once at least a single Wave Area is selected on plot, moves to the next analysis")

        btn3frame = ttk.Frame(self.frame5)
        btn3lbl = ttk.Label(btn3frame, text="Go to the start page")
        btn3lbl.grid(row=0, column=1)
        button_go_start = ttk.Button(btn3frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))#, style="greyBackground.TButton")
        button_go_start.image=self.controller.gotostartpage
        button_go_start.grid(row=0, column=0)
        btn3frame.grid(row=0, column=0)

        for i in range(0,1):
            self.frame5.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.frame5.columnconfigure(i, weight=1)
        self.frame5.grid(row=11, column=1, rowspan=1, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)
    
    def plotDots(self, event=None):
        self.controller.btn_lock = True
        self.refreshDotList()
        if self.CheckVar_dots.get() == 1:
            self.hidedots = False
            if len(self.dots_list) > 0:
                for dot in self.dots_list:
                    dot.set_visible(True)
            if len(self.dots_list_ax2) > 0:
                for dot in self.dots_list_ax2:
                    dot.set_visible(True)
        else:
            self.hidedots = True
            if len(self.dots_list) > 0:
                for dot in self.dots_list:
                    dot.set_visible(False)
            if len(self.dots_list_ax2) > 0:
                for dot in self.dots_list_ax2:
                    dot.set_visible(False)
        self.fig.canvas.draw()
        self.controller.btn_lock = False

    def init_vars(self, bckbtn=None): #PageFour
        self.controller.btn_lock = True
        self.ax.clear()
        if self.ax2 != None:
            self.ax2.clear()
        validate = True
        if self.controller.current_analysis == None:
            MsgBox = messagebox.askyesno(title='Load Analysis', message="Load Saved Analysis?")
            if MsgBox == True:
                #LOAD GROUP FROM FILE AND SET
                filename = filedialog.askopenfilename(title = "Select Saved Analysis file",filetypes = (("pickle analysis files","*.pickle"),("all files","*.*")))
                filename = r'%s' %filename
                if filename != None:
                    try:                
                        filehandler = open(r'%s' %filename, 'rb')
                        try:
                            diskgroup = pickle.load(filehandler)
                        except pickle.UnpicklingError:
                            messagebox.showerror("Error", "Loaded File is not a Saved Analysis File")
                            validate = False
                        filehandler.close()
                        if validate == True and isinstance(diskgroup, AnalysisGroup):
                            self.controller.current_analysis = diskgroup
                        elif validate == True:
                            messagebox.showerror("Error", "Loaded File is not a Saved Analysis File")
                            validate = False
                    except Exception as e:
                        messagebox.showerror("Error", "File could not be loaded\n"+str(e))
                        validate = False
                else:
                    messagebox.showerror("Error", "File could not be loaded")
                    validate = False
            else:
                messagebox.showerror("Error", "No Analysis Found")
                validate = False
        if validate == True:
            self.mainplotartist = None

            if bckbtn is None:
                self.controller.current_analysis.delta = None
                self.controller.current_analysis.noisemin = None
                self.controller.current_analysis.stopcond = None

            self.case = self.controller.current_analysis.name
            self.current_case = self.controller.current_analysis.mag_means.copy()
            self.old_current_case = self.controller.current_analysis.mag_means.copy()
            self.current_case_frames = list(range(len(self.current_case)))

            #reset dragdots drawn rectangles:
            self.dragDots.drawnrects = []
            self.dragDots.noisedrawnrects = []
            self.dragDots.user_selected_noise =[]
            self.dragDots.user_removed_noise = []

            self.dragDots.ax2_type = "Zoom"
            self.FPS = self.controller.current_analysis.FPS
            self.dotsize = None
            self.double_dotsize = None
            self.pixel_val = self.controller.current_analysis.pixelsize
            self.delta = None
            self.denoising = None
            self.stop_condition_perc = None
            self.adjustnoisevar = True
            self.prevrenoise = None
            self.usernoise = None
            self.hidedots = True
            self.decreasenoise = False
            self.userdecreasenoise = False
            self.gvf_cutoff = None
            self.plotsettings = self.controller.plotsettings
            self.plotsettings.set_limit(len(self.current_case))

            self.noises = None
            self.add_axis = False
            self.exponential_pops = []
            self.regression_list = []
            self.regression_list2 = []
            self.noise_line = None
            self.noise_line_ax2 = None
            self.runupdate()
            
            if self.controller.current_analysis.delta is not None:
                self.spin_deltavalue.delete(0,"end")
                self.spin_deltavalue.insert(0,self.controller.current_analysis.delta)
                self.spin_stopcondition.delete(0,"end")
                self.spin_stopcondition.insert(0,self.controller.current_analysis.stopcond)
                self.spin_cutoff.delete(0,"end")
                self.spin_cutoff.insert(0,self.controller.current_analysis.noisemin)
                print(" if self.controller.current_analysis.delta is not None: update_with_delta_freq")
                self.update_with_delta_freq()

            self.controller.btn_lock = False
            return True
        self.controller.btn_lock = False
        return False

    def exportimage(self, exptype):
        self.controller.btn_lock = True
        formats = set(self.fig.canvas.get_supported_filetypes().keys())
        if exptype == "full":
            d = SaveFigureDialog(self, title='Save Figure', literals=[
                ("formats", formats),
                ("bbox", 1)
            ])
            if d.result != None:
                if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                    self.fig.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
                    messagebox.showinfo(
                        "File saved",
                        "File was successfully saved"
                    )
                else:
                    self.fig.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
                    messagebox.showinfo(
                        "File saved",
                        "File was successfully saved"
                    )
        elif exptype == "plot":
            d = SaveFigureDialog(self, title='Save Figure', literals=[
                ("formats", formats),
                ("bbox", 0)
            ])
            if d.result != None:
                extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                    self.fig.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=extent.expanded(1.1, 1.3))
                    messagebox.showinfo(
                        "File saved",
                        "File was successfully saved"
                    )
                else:
                    self.fig.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=extent.expanded(1.1, 1.3))
                    messagebox.showinfo(
                        "File saved",
                        "File was successfully saved"
                    )
        elif exptype == "sub":
            if self.ax2 != None:
                d = SaveFigureDialog(self, title='Save Figure', literals=[
                    ("formats", formats),
                    ("bbox", 0)
                ])
                if d.result != None:
                    extent = self.ax2.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                    if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                        self.fig.savefig(r'%s' %d.result["name"], quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=extent.expanded(1.1, 1.3))
                        messagebox.showinfo(
                            "File saved",
                            "File was successfully saved"
                        )
                    else:
                        self.fig.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=extent.expanded(1.1, 1.3))
                        messagebox.showinfo(
                            "File saved",
                            "File was successfully saved"
                        )
            else:
                messagebox.showerror("Error", "No Sub-Plot selected")
        self.controller.btn_lock = False

    def exportdata(self, exptype):
        self.controller.btn_lock = True
        if exptype == "plot": 
            times = [float(i / self.controller.current_analysis.FPS) for i in range(len(self.controller.current_analysis.mag_means))]
            if self.controller.current_timescale == "ms":
                times = [a * 1000 for a in times]
            SaveTableDialog(self, title='Save Table', literals=[
                ("headers", [self.ax.get_xlabel(), self.ax.get_ylabel()]),
                ("data", [times, self.controller.current_analysis.mag_means.copy()]),
                ("data_t", "single")
                ])
        elif exptype == "sub":
            if self.ax2:
                datax = self.dragDots.subplotartist[0].get_xdata()
                if self.dragDots.ax2_type == "Zoom":
                    datax = [float(i / self.controller.current_analysis.FPS) for i in datax]
                    if self.controller.current_timescale == "ms":
                        datax = [a * 1000 for a in datax]
                SaveTableDialog(self, title='Save Table', literals=[
                    ("headers", [self.ax2.get_xlabel(), self.ax2.get_ylabel()]),
                    ("data", [datax, self.dragDots.subplotartist[0].get_ydata()]),
                    ("data_t", "single")
                    ])
            else:
                messagebox.showerror("Error", "No Sub-Plot selected")
        self.controller.btn_lock = False

    def menubar(self, root):
        menu = tk.Menu(root, tearoff=0)# Plot Menu

        pageMenu = tk.Menu(menu, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        menu.add_cascade(label="File", menu=pageMenu)

        
        plotMenu = tk.Menu(menu, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)
        menu.add_cascade(label="Plot Settings", menu=plotMenu)

        export_menu = tk.Menu(menu, tearoff=0)
        export_menu.add_command(label='Export Current Image', command=lambda: self.exportimage("full"))
        export_menu.add_separator()
        export_menu.add_command(label='Export Plot Image', command=lambda: self.exportimage("plot"))
        export_menu.add_command(label='Export Plot Data', command=lambda: self.exportdata("plot"))
        export_menu.add_separator()
        export_menu.add_command(label='Export Sub Plot Image', command=lambda: self.exportimage("sub"))
        export_menu.add_command(label='Export Sub Plot Data', command=lambda: self.exportdata("sub"))
        export_menu.add_separator()

        menu.add_cascade(label='Export', menu=export_menu)

        plots_menu = tk.Menu(menu, tearoff=0)
        plots_menu.add_command(label='Peak Area Select', command=lambda:self.set_dragmode(txt="edit"))#, command=set_original)
        plots_menu.add_separator()
        plots_menu.add_command(label='Peak Point Select', command=lambda:self.set_dragmode(txt="point"))#)
        plots_menu.add_separator()
        plots_menu.add_command(label='None', command=lambda:self.set_dragmode(txt="none"))#, command=set_original)
        plots_menu.add_separator()
        menu.add_cascade(label='Plot Mouse Mode', menu=plots_menu)

        # Sub Plot Menu
        sub_plots_menu = tk.Menu(menu, tearoff=0)
        sub_plots_menu.add_command(label='None', command=lambda:self.add_a_subplot(txt="None"))#, command=set_original)
        sub_plots_menu.add_separator()
        sub_plots_menu.add_command(label='Peak Zoom', command=lambda:self.add_a_subplot(txt="Zoom"))#, command=set_original)
        sub_plots_menu.add_separator()
        sub_plots_menu.add_command(label='Noise/Peak Areas', command=lambda:self.add_a_subplot(txt="PeakNoise"))#, command=set_original)
        sub_plots_menu.add_separator()
        sub_plots_menu.add_command(label='FFT', command=lambda:self.add_a_subplot(txt="FFT"))#, command=set_original)
        sub_plots_menu.add_separator()
        menu.add_cascade(label='Sub-Plot Mode', menu=sub_plots_menu)

        # Data Menu
        data_menu = tk.Menu(menu, tearoff=0)
        data_menu.add_command(label='Restore Original', command=self.set_original)
        data_menu.add_separator()
        data_menu.add_command(label='Noise Detection Options', command=self.adjustnoise)#, command=self.set_original)
        data_menu.add_separator()

        # Smooth/Noise Sub Menu
        noise_menu = tk.Menu(data_menu, tearoff=0)

        noise_menu.add_command(label='Average Window', command=self.set_npconv)
        noise_menu.add_separator()

        noise_menu.add_command(label='Savitzky-Golay', command=self.set_savgol)
        noise_menu.add_separator()

        noise_menu.add_command(label='FFT denoise', command=self.set_fourier)
        noise_menu.add_separator()

        data_menu.add_cascade(label='Smooth/Denoise', menu=noise_menu)

        menu.add_cascade(label='Data Options', menu=data_menu)

        menu.add_command(label="About", command=self.controller.showabout)
        menu.add_command(label="Help", command=self.controller.showhelp)
        return menu

    def adjustnoise(self):
        self.controller.btn_lock = True
        e = 1
        if self.adjustnoisevar == False:
            e = 0
        e2 = 1
        if self.decreasenoise == False:
            e2 = 0
        e3 = 0.0
        if self.prevrenoise != None:
            e3 = self.prevrenoise
        e4 = 1
        if self.userdecreasenoise == False:
            e4 = 0
        d = AdjustNoiseDetectDialog(self, title='Noise Advanced Parameters', literals=[
            ("noiseareasfiltering", e),
            ("noisedecrease", e2),
            ("userdecrease", e4),
            ("noisedecreasevalue", e3)
            ])
        if d.result:
            self.adjustnoisevar = d.result["adjustnoisevar"]
            self.decreasenoise = d.result["noisedecrease"]
            if self.decreasenoise == True:
                self.check_decrease_value.set(1)
            else:
                self.check_decrease_value.set(0)
            self.userdecreasenoise = d.result["userdecrease"]
            self.usernoise = d.result["noisevalue"]
            print(" def adjustnoise update_with_delta_freq")
            self.update_with_delta_freq()
        self.controller.btn_lock = False

    def analysepeakareas(self):
        if self.controller.btn_lock == False:
            self.controller.btn_lock = True
            #first get peak areas from dragdots and validate
            #if not validation, show error
            #if validation, go to next plot
            peaks = self.dragDots.get_rectangles_data()
            if not peaks or len(peaks) == 0:
                messagebox.showerror("Error", "No Waves founds")
                self.controller.btn_lock = False
                return
            else:
                print(" def analysepeakareas update_with_delta_freq")
                self.update_with_delta_freq(doupdate=False)
                self.controller.current_analysis.delta = self.delta
                self.controller.current_analysis.noisemin = self.gvf_cutoff
                self.controller.current_analysis.stopcond = self.stop_condition_perc
                self.controller.peaks = peaks.copy()
                for epeak in self.controller.peaks:
                    epeak.switch_timescale(self.controller.current_timescale)
                self.controller.selectedframes = self.current_case_frames.copy()
                self.controller.btn_lock = False
                self.controller.show_frame("PageFive")

    def set_original(self, event=None):
        self.controller.btn_lock = True
        self.denoising = "Return"
        self.controller.btn_lock = False
        self.runupdate()

    def set_dragmode(self, event=None, txt=None):
        self.controller.btn_lock = True
        if txt:
            if self.dragDots.mode == None and txt != "none":
                self.dragDots.mode = txt
                self.dragDots.reconnect_modes()
            elif txt == "none":
                self.dragDots.mode = txt
                self.dragDots.reset_modes()
            else:
                self.dragDots.mode = txt
            if txt == "edit":
                self.ax.set_navigate(False)
                self.dragDots.ax.set_navigate(False)
            else:
                self.ax.set_navigate(True)
                self.dragDots.ax.set_navigate(True)
            self.fig.canvas.draw()
        self.controller.btn_lock = False

    def popupCFocusOut(self,event=None):
        self.controller.btn_lock = True
        if self.dragDots.selectareaopen == True:
            #close right click menu
            self.selectCMenu.grab_release()
            self.selectCMenu.unpost()
            self.focus_set()
            self.controller.popuplock = False
            self.controller.currentpopup = None

            #Reset lock and area states
            self.dragDots.selectareaopen = False
        self.controller.btn_lock = False

    def popupNFocusOut(self,event=None):
        self.controller.btn_lock = True
        if self.dragDots.noiseareaopen == True:
            #close right click menu
            self.areaNMenu.grab_release()
            self.areaNMenu.unpost()
            self.focus_set()
            self.controller.popuplock = False
            self.controller.currentpopup = None

            #Reset lock and area states
            self.dragDots.locknoiserect = False
            self.dragDots.noiseareaopen = False

            #Reset drag area rectangle xpos
            self.dragDots.noisearearect_x0 = None
            self.dragDots.noisearearect_x1 = None
        self.controller.btn_lock = False

    def popupFocusOut(self,event=None):
        self.controller.btn_lock = True
        if self.dragDots.areaopen == True:
            #close right click menu
            self.areaCMenu.grab_release()
            self.areaCMenu.unpost()
            self.focus_set()
            self.controller.popuplock = False
            self.controller.currentpopup = None

            #Reset lock and area states
            self.dragDots.lockrect = False
            self.dragDots.areaopen = False

            #Reset drag area rectangle xpos
            self.dragDots.arearect_x0 = None
            self.dragDots.arearect_x1 = None
        self.controller.btn_lock = False

    def local_peak_operation(self, event=None, txt=None):
        self.controller.btn_lock = True
        if txt == "denoise_sav":
            self.plotsettings.savgol_opts["start_x"] = self.dragDots.arearect_x0
            self.plotsettings.savgol_opts["end_x"] = self.dragDots.arearect_x1
            self.set_savgol()
            self.plotsettings.savgol_opts["start_x"] = 0
            self.plotsettings.savgol_opts["end_x"] = len(self.current_case)
        elif txt == "denoise_avg":
            self.plotsettings.np_conv["start_x"] = self.dragDots.arearect_x0
            self.plotsettings.np_conv["end_x"] = self.dragDots.arearect_x1
            self.set_npconv()
            self.plotsettings.np_conv["start_x"] = 0
            self.plotsettings.np_conv["end_x"] = len(self.current_case)
        elif txt == "set_current":
            x0 = self.dragDots.arearect_x0
            x1 = self.dragDots.arearect_x1
            if x1 < len(self.current_case):
                x1 += 1
            ncurrent_case = self.current_case[x0:x1].copy()
            self.current_case_frames = self.current_case_frames[x0:x1]
            self.current_case = []
            self.current_case.extend(ncurrent_case)
            prev_denoising = self.denoising
            self.denoising = "Fake"
            for erect in self.dragDots.drawnrects:
                erect.remove()
            self.dragDots.rect = None
            self.dragDots.drawnrects = []
            print(" def local_peak_operation update_with_delta_freq")
            self.update_with_delta_freq()
            self.denoising = prev_denoising
        elif txt == "del_current":
            x0 = self.dragDots.arearect_x0
            if x0 >= 0:
                x0 += 1
            x1 = self.dragDots.arearect_x1
            p_current_case = self.current_case[:x0].copy()
            l_current_case = self.current_case[x1:].copy()
            p_current_case_ind = self.current_case_frames[:x0].copy()
            l_current_case_ind = self.current_case_frames[x1:].copy()
            self.current_case = []
            self.current_case.extend(p_current_case)
            self.current_case.extend(l_current_case)
            self.current_case_frames = []
            self.current_case_frames.extend(p_current_case_ind)
            self.current_case_frames.extend(l_current_case_ind)
            prev_denoising = self.denoising
            self.denoising = "Fake"
            print(" def local_peak_operation update_with_delta_freq")
            self.update_with_delta_freq()
            self.denoising = prev_denoising
        elif txt == "edit_current":
            x0 = self.dragDots.arearect_x0
            x1 = self.dragDots.arearect_x1
            p_current_case = []
            if x0 >= 0:
                x0 += 1
                p_current_case = self.current_case[:x0].copy()
                x0 -= 1
            l_current_case = []
            if x1 < len(self.current_case):
                l_current_case = self.current_case[x1:].copy()
                x1 += 1
            m_current_case = [0.0 for i in self.current_case[x0:x1]]
            self.current_case = []
            self.current_case.extend(p_current_case)
            self.current_case.extend(m_current_case)
            self.current_case.extend(l_current_case)
            prev_denoising = self.denoising
            self.denoising = "Fake"
            print(" def local_peak_operation update_with_delta_freq")
            self.update_with_delta_freq()
            self.denoising = prev_denoising
        elif txt == "add_as_noise":
            # self.dragDots.user_selected_noise
            self.dragDots.set_area_as_noise()
            print(" def local_peak_operation update_with_delta_freq")
            self.update_with_delta_freq()
        elif txt == "remove_as_noise":
            # self.dragDots.user_selected_noise
            self.dragDots.unset_area_as_noise()
            self.update_with_delta_freq()
        elif txt == "add_dot":
            d = DotChangeDialog(self, title='Change Dot Type')
            if d.result != None:
                newtype = d.result
                self.dragDots.add_dot_at_last(newtype)
                self.plotDots()
        elif txt == "change_dot":
            #open dialog for dot change
            d = DotChangeDialog(self, title='Change Dot Type')
            if d.result != None:
                newtype = d.result
                self.dragDots.change_dot_at_last(newtype)
                self.plotDots()
        elif txt == "remove_dot":
            self.dragDots.remove_dot_at_last()
            self.plotDots()
        self.controller.btn_lock = False

    def refreshDotList(self):
        self.controller.btn_lock = True
        self.dots_list = []
        self.dots_list_ax2 = []
        for child in self.ax.get_children():
            if isinstance(child, Line2D) and child.get_marker() == "o":
                self.dots_list.append(child)
        #if self.ax2 != None:
        if self.dragDots.ax2 != None:
            for child2 in self.ax2.get_children():
                if isinstance(child2, Line2D) and child2.get_marker() == "o":
                    self.dots_list_ax2.append(child2)
        self.controller.btn_lock = False

    def set_fourier(self, event=None, start_x = None, end_x = None):
        self.controller.btn_lock = True
        self.denoising = "fourier"
        d = FourierConvDialog(self, title='Average Window Smoothing', literals=[
            ("maxvalues", 1.0),
            ("freqstart", self.plotsettings.fourier_opts["frequency_maintain"])
            ])
        if d.result:
            self.plotsettings.fourier_opts["frequency_maintain"] = d.result
            print(" def set_fourier update_with_delta_freq")
            self.update_with_delta_freq()
        self.controller.btn_lock = False

    def set_npconv(self, event=None, start_x = None, end_x = None):
        self.controller.btn_lock = True
        self.denoising = "convol"
        d = NpConvDialog(self, title='Scaled Window Smoothing', literals=[
            ("maxvalues", len(self.current_case)),
            ("windowstart", self.plotsettings.np_conv["window_length"]),
            ("window_type", self.plotsettings.np_conv["window_type"] )
            ])
        if d.result:
            self.plotsettings.np_conv["window_length"] = d.result[0]
            self.plotsettings.np_conv["window_type"] = d.result[1]
            print(" def set_npconv update_with_delta_freq")
            self.update_with_delta_freq()
        self.controller.btn_lock = False
    
    def set_savgol(self, event=None, start_x = None, end_x = None):
        self.controller.btn_lock = True
        self.denoising = "savgol"
        d = SavGolDialog(self, title='Savitzky-Golay Filter', literals=[
            ("maxvalues", len(self.current_case)),
            ("windowstart", self.plotsettings.savgol_opts["window_length"]),
            ("polystart", self.plotsettings.savgol_opts["polynomial_order"]),
            ])
        if d.result:
            self.plotsettings.savgol_opts["window_length"] = d.result[0]
            self.plotsettings.savgol_opts["polynomial_order"] = d.result[1]
            print(" def set_savgol update_with_delta_freq")
            self.update_with_delta_freq()
        self.controller.btn_lock = False

    def return_last_spinner(self, spintype, errormsg):
        self.controller.btn_lock = True
        messagebox.showerror("Error " + spintype, errormsg)
        if spintype == "delta":
            self.spin_deltavalue.delete(0,"end")
            self.spin_deltavalue.insert(0,self.delta)
        elif spintype == "stop_condition_perc":
            self.spin_stopcondition.delete(0,"end")
            self.spin_stopcondition.insert(0,self.stop_condition_perc)
        elif spintype == "gvf_cutoff":
            self.spin_cutoff.delete(0,"end")
            self.spin_cutoff.insert(0,self.gvf_cutoff)
        self.controller.btn_lock = False

    def update_with_delta_freq(self, event=None, doupdate=True):
        print("######")
        print("def update_with_delta_freq")
        print("######")
        self.controller.btn_lock = True
        delta_val = self.spin_deltavalue.get().replace(",",".")
        stop_condition_perc_val = self.spin_stopcondition.get().replace(",",".")
        gvf_cutoff_val = self.spin_cutoff.get().replace(",",".")
        if self.check_decrease_value.get() == 1:
            self.decreasenoise = True
        else:
            self.decreasenoise = False
        try:
            delta_val = float(delta_val)
        except ValueError:
            self.return_last_spinner("delta", "Spinner value is not float!")
            self.controller.btn_lock = False
            return
        try:
            stop_condition_perc_val = float(stop_condition_perc_val)
        except ValueError:
            self.return_last_spinner("stop_condition_perc", "Spinner value is not float!")
            self.controller.btn_lock = False
        try:
            gvf_cutoff_val = float(gvf_cutoff_val)
        except ValueError:
            self.return_last_spinner("gvf_cutoff", "Spinner value is not float!")
            self.controller.btn_lock = False
            return
        if delta_val < 0:
            self.return_last_spinner("delta", "Invalid Spinner value!")
            self.controller.btn_lock = False
            return
        if stop_condition_perc_val < 0 or stop_condition_perc_val > 1:
            self.return_last_spinner("stop_condition_perc", "Invalid Spinner value!")
            self.controller.btn_lock = False
            return
        if gvf_cutoff_val < 0:
            self.return_last_spinner("gvf_cutoff", "Invalid Spinner value!")
            self.controller.btn_lock = False
            return
        if doupdate == True:
            self.runupdate(delta_val=delta_val, stop_condition_perc_val=stop_condition_perc_val, gvf=gvf_cutoff_val)
            self.delta = delta_val
            self.stop_condition_perc = stop_condition_perc_val
            self.gvf_cutoff = gvf_cutoff_val
        self.controller.btn_lock = False

    def add_a_subplot(self, txt=None):
        self.controller.btn_lock = True
        self.dragDots.ax2_type = txt
        if txt != "None":
            self.ax.set_position(self.gs2[0].get_position(self.fig))
            
            if len(self.fig.axes) > 1:
                self.fig.delaxes(self.fig.axes[1])
                self.ax2 = None
            
            if self.ax2 == None:
                self.fig.add_subplot(self.gs2[1])
                self.ax2 = self.fig.axes[1]
            
            if self.ax2:
                self.ax2.set_visible(True)
            self.dragDots.ax2 = self.ax2
            self.dragDots.drawAx2()
        else:
            self.ax.set_position(self.gs[0].get_position(self.fig))
            if self.ax2:
                self.ax2.set_visible(False)
            
            if len(self.fig.axes) > 1:
                self.fig.delaxes(self.fig.axes[1])
                self.ax2 = None
            self.dragDots.ax2 = None
        #according to subplot type, add subplot data
        self.plotDots()
        self.fig.canvas.draw()
        self.controller.btn_lock = False

    def denoise(self, thiscase, denoising):
        finalcase = thiscase.copy()
        if self.denoising == "savgol":
            finalcase = []
            if self.plotsettings.savgol_opts["start_x"] > 0:
                finalcase.extend(thiscase[:self.plotsettings.savgol_opts["start_x"]])

            current_case_pre = savgol_filter(thiscase[self.plotsettings.savgol_opts["start_x"]: self.plotsettings.savgol_opts["end_x"]], self.plotsettings.savgol_opts["window_length"], self.plotsettings.savgol_opts["polynomial_order"], mode='nearest')
            
            if self.plotsettings.savgol_opts["above_zero"] == 0:
                for a in current_case_pre:
                    if a > 0:
                        finalcase.append(a)
                    else:
                        finalcase.append(0.0)

            if self.plotsettings.savgol_opts["end_x"] < len(thiscase):
                finalcase.extend(thiscase[self.plotsettings.savgol_opts["end_x"]:])

        elif self.denoising == "convol":
            finalcase = []

            if self.plotsettings.np_conv["start_x"] > 0:
                finalcase.extend(thiscase[:self.plotsettings.np_conv["start_x"]])

            current_case_pre = smooth_scipy(thiscase[self.plotsettings.np_conv["start_x"]: self.plotsettings.np_conv["end_x"]], self.plotsettings.np_conv["window_length"], self.plotsettings.np_conv["window_type"])
            
            for a in current_case_pre:
                if a > 0:
                    finalcase.append(a)
                else:
                    finalcase.append(0.0)

            if self.plotsettings.np_conv["end_x"] < len(thiscase):
                finalcase.extend(thiscase[self.plotsettings.np_conv["end_x"]:])

        elif self.denoising == "fourier":
            finalcase = []
            if self.plotsettings.fourier_opts["start_x"] > 0:
                finalcase.extend(thiscase[:self.plotsettings.fourier_opts["start_x"]])

            rft = np.fft.rfft(thiscase[self.plotsettings.fourier_opts["start_x"]: self.plotsettings.fourier_opts["end_x"]])
            filter_index = int(rft.shape[0] * self.plotsettings.fourier_opts["frequency_maintain"])
            rft[filter_index:] = 0
            current_case_pre = np.fft.irfft(rft)

            for a in current_case_pre:
                if a > 0:
                    finalcase.append(a)
                else:
                    finalcase.append(0.0)
            
            if self.plotsettings.fourier_opts["end_x"] < len(thiscase):
                finalcase.extend(thiscase[self.plotsettings.fourier_opts["end_x"]:])    
        return finalcase

    def runupdate(self, delta_val=False, stop_condition_perc_val=False, gvf=None, revert_case=False):
        self.controller.showwd()
        # wd = WaitDialogProgress(self, title='In Progress...')
        print("start here")
        # self.controller.update()
        # wd.progress_bar.start()
        # self.update_idletasks()
        self.controller.btn_lock = True
        #set plot fig main title
        # self.fig.suptitle(self.case)
        # self.fig.suptitle(self.case)
        
        #read noise if previously removed
        if self.prevrenoise != None:
            print("self.prevrenoise")
            print(self.prevrenoise)
            val = self.prevrenoise
            self.current_case = [a + val for a in self.current_case]
        
        #a copy is made for possibly denoising the case
        current_case_val = self.current_case.copy()

        #set limit for plotsettings variables dependent of case length        
        self.plotsettings.set_limit(len(self.current_case))

        #denoise case line
        if self.denoising == "Return":
            #Return denoise restores current case to default
            self.current_case = self.old_current_case
        else:
            # previous_case = current_case.copy()
            self.current_case = self.denoise(current_case_val, self.denoising)
        
        #set limit for plotsettings variables dependent of case length        
        self.plotsettings.set_limit(len(self.current_case))

        #detect if user settings for peak and noise detection
        # update_vars = delta_val != False or stop_condition_perc_val != False or gvf != 0.90
        update_vars = delta_val != False or stop_condition_perc_val != False or gvf != None
        
        #detect noise from case
        nargs = noise_detection(self.current_case,filter_noise_area=self.adjustnoisevar, added_noise_dots=self.dragDots.user_selected_noise, removed_noise_dots=self.dragDots.user_removed_noise, cutoff_val=gvf)
        
        if nargs == None:
            messagebox.showerror("Error:", "Noise not detected. Please adjust the Noise cutoff")    
            self.controller.cancelwd()
            return
        #decrease noise from case if selected
        if self.decreasenoise == True:
            print("decrease noise")
            val = (nargs[6]+nargs[7])
            self.current_case = [a - val  for a in self.current_case]
            self.prevrenoise = val

        elif self.userdecreasenoise == True:
            self.current_case = [a - self.usernoise  for a in self.current_case]
            self.prevrenoise = self.usernoise

        else:
            self.prevrenoise = None

        #detect points from case
        self.points, noises_vals, exponential_pops_vals, conditions = peak_detection(self.current_case, delta=delta_val, stop_condition_perc=stop_condition_perc_val, nargs=nargs)
        
        #save exponential regressions and noise positions/values
        self.exponential_pops = exponential_pops_vals
        self.noises = noises_vals

        #if user settings do not exist for peak and noise detection:
        if update_vars == False:
            #send detected conditions to window
            self.delta = conditions[0]
            self.stop_condition_perc = conditions[1]
            self.gvf_cutoff = conditions[2]

            self.spin_deltavalue.delete(0,"end")
            self.spin_deltavalue.insert(0,conditions[0])

            self.spin_stopcondition.delete(0,"end")
            self.spin_stopcondition.insert(0,conditions[1])

            self.spin_cutoff.delete(0,"end")
            self.spin_cutoff.insert(0,conditions[2])

        #points are saved by type
        self.f_points ,self.s_f_points ,self.t_points ,self.l_points = self.points
        
        #figure main ax is cleared for drawing
        self.ax.clear()
        self.ax.set_title(self.case)

        #x and y labels are set according to current timescale (default: seconds)
        self.ax.set_xlabel("Time("+ self.controller.current_timescale +")")
        self.ax.set_ylabel("Average Speed("+ self.controller.current_speedscale+")")
        
        #current case is plotted as line to ax
        self.mainplotartist = self.ax.plot(self.current_case, color=self.plotsettings.peak_plot_colors["main"])
        
        #dot list is created and plot is updated
        self.dots_list = []
        self.dots_list_ax2 = []
        self.fig.canvas.draw()

        #labels are set for the x axis and updated to plot
        labels = None
        if self.controller.current_timescale == "s":
            labels = [ float("{:.3f}".format(float(item.get_text().replace("−", "-")) / self.FPS)) for item in self.ax.get_xticklabels() ]
        elif self.controller.current_timescale == "ms":
            labels = [ float("{:.3f}".format((float(item.get_text().replace("−", "-")) / self.FPS)*1000)) for item in self.ax.get_xticklabels() ]
        self.ax.set_xticklabels(labels)

        #plot X baseline if selected by user
        if self.axbaseline != None:
            self.axbaseline.remove()
            self.axbaseline = None
        if self.plotsettings.plotline_opts["zero"] == True:
            self.axbaseline = self.ax.axhline(y=0.0, color=self.plotsettings.plotline_opts["zero_color"], linestyle='-')

        #plot Grid if selected by user
        if self.axgrid != None:
            self.axgrid.remove()
            self.axgrid = None
        if self.plotsettings.plotline_opts["grid"] == True:
            self.axgrid = self.ax.grid(linestyle="-", color=self.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.ax.grid(False)

        #plot each type of detected dot and save in dots_list
        fdots = []
        for x, y in zip(self.f_points, [self.current_case[i] for i in self.f_points]):
            dot = self.ax.plot(x, y, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["first"], picker=5)
            # dot = self.ax.plot(x, y, "o", color=self.plotsettings.peak_plot_colors["first"], picker=5)
            dot[0].pointtype = "first"
            fdots.extend(dot)        
        sfdots = []
        for x, y in zip(self.s_f_points, [self.current_case[i] for i in self.s_f_points]):
            dot = self.ax.plot(x, y, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
            # dot = self.ax.plot(x, y, "o", color=self.plotsettings.peak_plot_colors["max"], picker=5)
            dot[0].pointtype = "max"
            sfdots.extend(dot)
        mdots = []
        for x, y in zip(self.t_points, [self.current_case[i] for i in self.t_points]):
            dot = self.ax.plot(x, y, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["min"], picker=5)
            # dot = self.ax.plot(x, y, "o", color=self.plotsettings.peak_plot_colors["min"], picker=5)
            dot[0].pointtype = "min"
            mdots.extend(dot)
        ldots = []
        for x, y in zip(self.l_points, [self.current_case[i] for i in self.l_points]):
            dot = self.ax.plot(x, y, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)
            # dot = self.ax.plot(x, y, "o", color=self.plotsettings.peak_plot_colors["last"], picker=5)
            dot[0].pointtype = "last"
            ldots.extend(dot)

        #dots list size is saved for clicking functions
        if len(fdots) > 0:
            self.dotsize = fdots[0].get_markersize()
            self.double_dotsize = fdots[0].get_markersize() * 2
        else:
            forced_dot = self.ax.plot(0, 0, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)
            self.dotsize = forced_dot[0].get_markersize()
            self.double_dotsize = forced_dot[0].get_markersize() * 2
            forced_dot[0].remove()
            forced_dot = None

        #dots lists is filled with each type of dots
        self.dots_list.extend(fdots)
        self.dots_list.extend(sfdots)
        self.dots_list.extend(mdots)
        self.dots_list.extend(ldots)

        #previous ax2_type, and drawn noise rectanles are saved
        txt = self.dragDots.ax2_type
        prevndr = self.dragDots.noisedrawnrects.copy()
        prevndt = self.dragDots.user_selected_noise.copy()
        prevurn = self.dragDots.user_removed_noise.copy()
        rect = self.dragDots.rect
        drawnrects = self.dragDots.drawnrects.copy()

        #previous drag class is destroyed and a new one is created, preserving some things such as the noise rectangles
        del self.dragDots
        self.dragDots = MoveDragHandler(master=self.controller,currentgroup=self.controller.current_analysis, FPS=self.FPS, pixel_val=self.pixel_val, figure=self.fig, ax=self.ax, data=self.current_case, selectCMenu=self.selectCMenu, areaCMenu=self.areaCMenu, areaNMenu=self.areaNMenu, colorify=self.plotsettings.peak_plot_colors, plotconf=self.plotsettings.plotline_opts, noiseindexes=self.noises[3], dsizes=(self.dotsize, self.double_dotsize),ax2baseline=self.ax2baseline,ax2grid=self.ax2grid)
        self.dragDots.noisedrawnrects = prevndr
        self.dragDots.user_selected_noise = prevndt
        self.dragDots.user_removed_noise = prevurn

        #readds previous selection by user on plot updates with adjusted height and y positioning
        self.dragDots.rect = rect
        if rect != None:
            curlims = (self.ax.get_xlim(), self.ax.get_ylim())
            # nheight = np.max(self.current_case) + abs(np.min(self.current_case)) + 0.5
            nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5
            rect.set_height(nheight)
            # rect.set_y(np.min(self.current_case))
            rect.set_y(curlims[1][0])
            self.ax.add_patch(rect)
            self.ax.set_xlim(curlims[0])
            self.ax.set_ylim(curlims[1])

        self.dragDots.drawnrects = drawnrects
        for erect in drawnrects:
            curlims = (self.ax.get_xlim(), self.ax.get_ylim())
            # nheight = np.max(self.current_case) + abs(np.min(self.current_case)) + 0.5
            nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5
            erect.set_height(nheight)
            # erect.set_y(np.min(self.current_case))
            erect.set_y(curlims[1][0])
            self.ax.add_patch(erect)
            self.ax.set_xlim(curlims[0])
            self.ax.set_ylim(curlims[1])


        #checks if user wants to draw a subplot and which type
        self.add_a_subplot(txt=txt)

        #regressions and noise line is plotted if selected
        if self.CheckVar1.get() == 1:
            self.plotThoseRegressions()
        if self.CheckVar2.get() == 1:
            self.plotNoiseLine()

        #dots are hidden/shown according to user selection
        self.plotDots()

        #canvas is finally updated to user
        self.fig.canvas.draw()
        self.controller.btn_lock = False
        self.controller.cancelwd()

    def plotThoseRegressions(self):
        self.controller.btn_lock = True
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if self.dragDots.ax2_type == "Zoom":
            xlim2 = self.ax2.get_xlim()
            ylim2 = self.ax2.get_ylim()

        #reset both regression lists
        for a in range(0, len(self.regression_list)):
            self.regression_list[a][0].remove()
        self.regression_list = []
        for a in range(0, len(self.regression_list2)):
            self.regression_list2[a][0].remove()
        self.regression_list2 = []

        for exp_pops in self.exponential_pops:
            this_pop = self.ax.plot(exp_pops[0] , exponential_fit(exp_pops[0], *exp_pops[1]), color=self.plotsettings.peak_plot_colors["last"])
            self.regression_list.append(this_pop)
            if self.dragDots.ax2_type == "Zoom":
                this_pop2 = self.ax2.plot(exp_pops[0] , exponential_fit(exp_pops[0], *exp_pops[1]), color=self.plotsettings.peak_plot_colors["last"])
                self.regression_list2.append(this_pop2)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if self.dragDots.ax2_type == "Zoom":
            self.ax2.set_xlim(xlim2)
            self.ax2.set_ylim(ylim2)
        self.controller.btn_lock = False

    def plotNoiseLine(self):
        self.controller.btn_lock = True
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xlim2 = self.ax2.get_xlim()
        ylim2 = self.ax2.get_ylim()

        if self.noise_line is not None:
            self.noise_line[0].remove()
            self.noise_line = None
        if self.noise_line_ax2 is not None:
            self.noise_line_ax2[0].remove()
            self.noise_line_ax2 = None

        val = 0.0
        if self.prevrenoise is not None:
            val = self.prevrenoise

        self.noise_line = self.ax.plot([self.noises[2] - val for i in range(len(self.current_case))], color=self.plotsettings.peak_plot_colors["min"])
        if self.dragDots.ax2_type == "Zoom":
            self.noise_line_ax2 = self.ax2.plot([self.noises[2] - val for i in range(len(self.current_case))], color=self.plotsettings.peak_plot_colors["min"])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if self.dragDots.ax2_type == "Zoom":
            self.ax2.set_xlim(xlim2)
            self.ax2.set_ylim(ylim2)
        self.controller.btn_lock = False

    def plotRegressions(self, event=None):
        self.controller.btn_lock = True
        C1_val = self.CheckVar1.get()
        if C1_val == 1:
            self.plotThoseRegressions()
        else:
            for a in range(0, len(self.regression_list)):
                self.regression_list[a][0].remove()
            self.regression_list = []
            for a in range(0, len(self.regression_list2)):
                self.regression_list2[a][0].remove()
            self.regression_list2 = []
        self.fig.canvas.draw()
        self.controller.btn_lock = False

    def plotMeanNoise(self, event=None):
        self.controller.btn_lock = True
        C2_val = self.CheckVar2.get()
        if C2_val == 1:
            self.plotNoiseLine()
        else:
            if self.noise_line is not None:
                self.noise_line[0].remove()
            if self.noise_line_ax2 is not None:
                self.noise_line_ax2[0].remove()
            self.noise_line = None
            self.noise_line_ax2 = None
        self.fig.canvas.draw()
        self.controller.btn_lock = False

class PageFive(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageFive"
        self.peaks = []
        self.current_tabletree = None
        self.current_subtabletree = None
        self.current_tablevsb = None
        self.current_tablevsb2 = None
        self.plotsettings = self.controller.plotsettings

        for i in range(0,12):
            self.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.columnconfigure(i, weight=1)

        label = ttk.Label(self, text="Wave Parameters Plotting", font=controller.title_font, anchor=tk.CENTER)#, style="greyBackground.TLabel")
        label.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        #row 1

        self.changeval = tk.IntVar()
        self.changeval.set(0)

        self.frame1 = ttk.Frame(self)

        self.radio1 = ttk.Radiobutton(self.frame1, text = "Time", variable = self.changeval, value = 0, command=self.settabletype)#, style='greyBackground.TRadiobutton')
        self.radio1.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)

        CreateToolTip(self.radio1, \
    "Upper Table is updated with Time related variables for each Wave Point. Average Values for each Time related variable are shown in the Lower Table")

        self.radio2 = ttk.Radiobutton(self.frame1, text = "Speed", variable = self.changeval, value = 1, command=self.settabletype)#, style='greyBackground.TRadiobutton')
        self.radio2.grid(row=0, column=1, sticky=tk.W+tk.E+tk.N+tk.S)

        CreateToolTip(self.radio2, \
    "Upper Table is updated with Speed related variables for each Wave Point. Average Values for each Speed related variable are shown in the Lower Table")


        self.radio3 = ttk.Radiobutton(self.frame1, text = "Area", variable = self.changeval, value = 2, command=self.settabletype)#, style='greyBackground.TRadiobutton')
        self.radio3.grid(row=0, column=2, sticky=tk.W+tk.E+tk.N+tk.S)

        CreateToolTip(self.radio3, \
    "Upper Table is updated with Area related variables for each Wave Point. Average Values for each Area related variable are shown in the Lower Table")

        self.frame1.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)


        for i in range(0,1):
            self.frame1.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.frame1.columnconfigure(i, weight=1)


        #row 2 to row 6
        #https://stackoverflow.com/questions/14359906/horizontal-scrolling-wont-activate-for-ttk-treeview-widget
        #https://stackoverflow.com/questions/33375489/how-can-i-attach-a-vertical-scrollbar-to-a-treeview-using-tkinter/33376178#33376178

        self.tabletreeframe = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.subtabletreeframe = ttk.Frame(self)#, style="greyBackground.TFrame")

        self.peakRow_iid = None
        self.peakRowSelectMenu=tk.Menu(self, tearoff=0)
        self.peakRowSelectMenu.add_command(label="Close Menu")
        self.peakRowSelectMenu.add_command(label="Remove Peak", command=self.remove_item)
        self.peakRowSelectMenu.bind("<FocusOut>", self.focus_out_menu)

        self.timecols = ("Contraction-Relaxation Time (CRT)", "Contraction Time (CT)", "Relaxation Time (RT)", "Contraction time-to-peak (CTP)", "Contraction time from peak to minimum speed (CTPMS)", "Relaxation time-to-peak (RTP)", "Relaxation time from peak to Basaline (RTPB)", "Time between Contraction-Relaxation maximum speed (TBC-RMS)")
        self.speedcols = ('Maximum Contraction Speed (MCS)','Maximum Relaxation Speed (MRS)','MCS/MRS Difference Speed (MCS/MRS-DS)')
        self.areacols = ('Contraction-Relaxation Area (CRA)', 'Shortening Area (SA)')

        #table time
        self.tabletimetree = ttk.Treeview(self.tabletreeframe, columns=self.timecols, selectmode='browse', show='headings')
        self.subtabletimetree = ttk.Treeview(self.subtabletreeframe, columns=self.timecols, selectmode='none', height=1, show='headings')
        
        self.tabletimetree.grid(row=0, column=0, sticky=tk.NSEW)
        self.subtabletimetree.grid(row=0, column=0, sticky=tk.NSEW)

        self.tabletimetree.tag_configure('oddrow', background='#d3d3d3')
        self.tabletimetree.tag_configure('evenrow', background='white')
        #fill table with time as default columns
        for col in self.timecols:
            self.tabletimetree.heading(col, text=col)
            self.subtabletimetree.heading(col, text='Avg. '+col)

        self.vsbtime = ttk.Scrollbar(self.tabletreeframe, orient="vertical", command=self.tableTimeYScroll)
        self.vsbtimehorizontal = ttk.Scrollbar(self.tabletreeframe, orient="horizontal", command=self.tableTimeXScroll)
        
        self.vsbtime.grid(row=0,column=1,sticky=tk.NS)
        self.vsbtimehorizontal.grid(row=1,column=0,sticky=tk.EW)

        self.tabletimetree.configure(yscrollcommand=self.vsbtime.set, xscrollcommand=self.vsbtimehorizontal.set)
        self.subtabletimetree.configure(yscrollcommand=self.vsbtime.set, xscrollcommand=self.vsbtimehorizontal.set)
        self.tabletimetree.bind("<<TreeviewSelect>>", self.tabletreeselection)
        self.tabletimetree.bind("<Button-3>", self.show_focus_menu)

        self.tabletimetree.grid_forget()
        self.subtabletimetree.grid_forget()
        self.vsbtime.grid_forget()
        self.vsbtimehorizontal.grid_forget()

        #table speed
        self.tablespeedtree = ttk.Treeview(self.tabletreeframe, columns=self.speedcols, selectmode='browse', show='headings')
        self.subtablespeedtree = ttk.Treeview(self.subtabletreeframe, columns=self.speedcols, selectmode='none', height=1, show='headings')
        
        self.tablespeedtree.grid(row=0, column=0, sticky=tk.NSEW)
        self.subtablespeedtree.grid(row=0, column=0, sticky=tk.NSEW)

        self.tablespeedtree.tag_configure('oddrow', background='#d3d3d3')
        self.tablespeedtree.tag_configure('evenrow', background='white')
        for col in self.speedcols:
            self.tablespeedtree.heading(col, text=col)
            self.subtablespeedtree.heading(col, text='Avg. '+col)

        self.vsbspeed = ttk.Scrollbar(self.tabletreeframe, orient="vertical", command=self.tableSpeedYScroll)
        self.vsbspeedhorizontal = ttk.Scrollbar(self.tabletreeframe, orient="horizontal", command=self.tableSpeedXScroll)


        self.vsbspeed.grid(row=0,column=1,sticky=tk.NS)
        self.vsbspeedhorizontal.grid(row=1,column=0,sticky=tk.EW)

        self.tablespeedtree.configure(yscrollcommand=self.vsbspeed.set, xscrollcommand=self.vsbspeedhorizontal.set)
        self.subtablespeedtree.configure(yscrollcommand=self.vsbspeed.set, xscrollcommand=self.vsbspeedhorizontal.set)
        self.tablespeedtree.bind("<<TreeviewSelect>>", self.tabletreeselection)
        self.tablespeedtree.bind("<Button-3>", self.show_focus_menu)
        

        self.tablespeedtree.grid_forget()
        self.subtablespeedtree.grid_forget()
        self.vsbspeed.grid_forget()
        self.vsbspeedhorizontal.grid_forget()

        #table area
        self.tableareatree = ttk.Treeview(self.tabletreeframe, columns=self.areacols, selectmode='browse', show='headings')
        self.subtableareatree = ttk.Treeview(self.subtabletreeframe, columns=self.areacols, selectmode='none', height=1, show='headings')
        
        self.tableareatree.grid(row=0, column=0, sticky=tk.NSEW)
        self.subtableareatree.grid(row=0, column=0, sticky=tk.NSEW)

        self.tableareatree.tag_configure('oddrow', background='#d3d3d3')
        self.tableareatree.tag_configure('evenrow', background='white')
        for col in self.areacols:
            self.tableareatree.heading(col, text=col)
            self.subtableareatree.heading(col, text='Avg. '+col)

        self.vsbarea = ttk.Scrollbar(self.tabletreeframe, orient="vertical", command=self.tableAreaYScroll)
        self.vsbareahorizontal = ttk.Scrollbar(self.tabletreeframe, orient="horizontal", command=self.tableAreaXScroll)

        self.vsbarea.grid(row=0,column=1,sticky=tk.NS)
        self.vsbareahorizontal.grid(row=1,column=0,sticky=tk.EW)

        self.tableareatree.configure(yscrollcommand=self.vsbarea.set, xscrollcommand=self.vsbareahorizontal.set)
        self.subtableareatree.configure(yscrollcommand=self.vsbarea.set, xscrollcommand=self.vsbareahorizontal.set)
        self.tableareatree.bind("<<TreeviewSelect>>", self.tabletreeselection)
        self.tableareatree.bind("<Button-3>", self.show_focus_menu)
        
        self.tableareatree.grid_forget()
        self.subtableareatree.grid_forget()
        self.vsbarea.grid_forget()
        self.vsbareahorizontal.grid_forget()

        self.tabletreeframe.grid(row=2, column=0, rowspan=4, columnspan=3, sticky=tk.NSEW)
        self.tabletreeframe.columnconfigure(0, weight=1)
        self.tabletreeframe.rowconfigure(0, weight=1)

        self.subtabletreeframe.grid(row=6, column=0, rowspan=1, columnspan=3, sticky=tk.NSEW)
        
        self.subtabletreeframe.columnconfigure(0, weight=1)

        #row 6 to row 10
        self.fig = plt.figure(figsize=(2.5, 2), dpi=100, facecolor=self.controller.bgcolor)


        self.fig.tight_layout()
        self.gs = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2)
        self.frame_canvas = ttk.Frame(self)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_canvas)  # A tk.DrawingArea.
        self.mainplotartist = None
        self.ax = self.fig.add_subplot()
        self.axbaseline = None
        self.axgrid = None
        self.ax.set_xlabel("Time("+self.controller.current_timescale+")")
        self.ax.set_ylabel("Average Speed("+self.controller.current_speedscale+")")
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame_canvas.grid(row=7, column=0, rowspan=4, columnspan=3, sticky=tk.NSEW)

        self.frame_canvas.grid(row=7, column=0, rowspan=4, columnspan=3, sticky=tk.NSEW)
        
        #row 11

        #row 12
        self.frame5 = ttk.Frame(self)

        # self.gotostartpage = tk.PhotoImage(file="icons/refresh-sharp.png")
        # self.goback = tk.PhotoImage(file="icons/arrow-back-sharp.png")
        # self.jetquiverpltimg = tk.PhotoImage(file="icons/layers-sharp.png")

        btn1frame = ttk.Frame(self.frame5)
        btn1lbl = ttk.Label(btn1frame, text="Go back")
        btn1lbl.grid(row=0, column=1)
        button_go_back = ttk.Button(btn1frame, image=self.controller.goback,
                           command=lambda: controller.show_frame("PageFour", bckbtn=True))
        button_go_back.image =self.controller.goback
        button_go_back.grid(row=0, column=0)
        btn1frame.grid(row=0, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

        
        btn2frame = ttk.Frame(self.frame5)
        btn2lbl = ttk.Label(btn2frame, text="Quiver/Jet Plots")
        btn2lbl.grid(row=0, column=1)
        button_go_analysis = ttk.Button(btn2frame, image=self.controller.jetquiverpltimg,
                           command=self.quiverjetgo)#, style="greyBackground.TButton")
        button_go_analysis.image =self.controller.jetquiverpltimg
        button_go_analysis.grid(row=0, column=0)
        btn2frame.grid(row=0, column=2, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

        CreateToolTip(button_go_analysis, \
    "Allows moving to the last analysis once a single Wave row is selected by clicking.")

        btn3frame = ttk.Frame(self.frame5)
        btn3lbl = ttk.Label(btn3frame, text="Go to the start page")
        btn3lbl.grid(row=0, column=1)
        button_go_start = ttk.Button(btn3frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))
        button_go_start.image =self.controller.gotostartpage
        button_go_start.grid(row=0, column=0)
        btn3frame.grid(row=0, column=0, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

        for i in range(0,1):
            self.frame5.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.frame5.columnconfigure(i, weight=1)
        self.frame5.grid(row=11, column=0, rowspan=1, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)
    
    def init_vars(self):
        self.controller.btn_lock = True
        self.ax.clear()
        self.fig.canvas.draw()
        validate = True
        self.mainplotartist = None
        if self.controller.peaks == None:
            MsgBox = messagebox.askyesno(title='Load Analysis', message="Load Saved Waves?")
            if MsgBox == True:
                filename = filedialog.askopenfilename(title = "Select Saved Waves file", initialdir="./savedwaves/",filetypes = (("pickle waves files","*.pickle"),("all files","*.*")))
                filename = r'%s' %filename
                if filename != None:
                    try:                
                        filehandler = open(r'%s' %filename, 'rb')
                        try:
                            diskgroup = pickle.load(filehandler)
                        except pickle.UnpicklingError:
                            messagebox.showerror("Error", "Loaded File is not a Saved Waves File")
                            validate = False
                        filehandler.close()
                        if validate == True and isinstance(diskgroup, PeaksObj):
                            self.controller.peaks = diskgroup.peaks
                            for epeak in self.controller.peaks:
                                epeak.switch_timescale(self.controller.current_timescale)
                            self.controller.current_analysis = diskgroup.thisgroup
                            self.controller.selectedframes = diskgroup.thisframes
                        elif validate == True:
                            messagebox.showerror("Error", "Loaded File is not a Saved Waves File")
                            validate = False
                    except Exception as e:
                        messagebox.showerror("Error", "File could not be loaded\n"+str(e))
                        validate = False
                else:
                    messagebox.showerror("Error", "File could not be loaded")
                    validate = False
            else:
                messagebox.showerror("Error", "No Waves Found")
                validate = False
        #CHECK FOR PEAKS EXISTANCE IN CONTROLLER
        if validate == True:
            self.peaks = self.controller.peaks.copy()
            self.controller.btn_lock = False
            return True
        self.controller.btn_lock = False
        return False

    def show_focus_menu(self, event=None):
        self.controller.btn_lock = True
        self.peakRow_iid = self.current_tabletree.identify_row(event.y)
        try:
            abs_coord_x = self.controller.winfo_pointerx() - self.controller.winfo_vrootx()
            abs_coord_y = self.controller.winfo_pointery() - self.controller.winfo_vrooty()
            self.controller.popuplock = True
            self.controller.currentpopup = self.peakRowSelectMenu
            self.peakRowSelectMenu.tk_popup(int(abs_coord_x + np.max([(self.peakRowSelectMenu.winfo_width()/2) + 10, 15])), abs_coord_y, 0)
        finally:
            self.peakRowSelectMenu.grab_release()
        self.controller.btn_lock = False

    def focus_out_menu(self, event=None):
        self.controller.btn_lock = True

        self.peakRowSelectMenu.grab_release()
        self.peakRowSelectMenu.unpost()
        self.current_tabletree.focus_set()
        self.controller.popuplock = False
        self.controller.currentpopup = None

        self.controller.btn_lock = False

    #get groups
    def tableTimeXScroll(self, *args):
        self.tabletimetree.xview(*args)
        self.subtabletimetree.xview(*args)

    def tableSpeedXScroll(self, *args):
        self.tablespeedtree.xview(*args)
        self.subtablespeedtree.xview(*args)

    def tableAreaXScroll(self, *args):
        self.tableareatree.xview(*args)
        self.subtableareatree.xview(*args)

    def tableTimeYScroll(self, *args):
        self.tabletimetree.yview(*args)
        self.subtabletimetree.yview(*args)

    def tableSpeedYScroll(self, *args):
        self.tablespeedtree.yview(*args)
        self.subtablespeedtree.yview(*args)

    def tableAreaYScroll(self, *args):
        self.tableareatree.yview(*args)
        self.subtableareatree.yview(*args)

    def settabletype(self, event=None):
        self.controller.btn_lock = True
        if self.changeval.get() == 0:
            self.filltabletype("time")

        elif self.changeval.get() == 1:
            self.filltabletype("speed")

        elif self.changeval.get() == 2:
            self.filltabletype("area")
        self.controller.btn_lock = False

    def filltabletype(self, type_cols):
        self.controller.btn_lock = True
        tabletree = None
        subtabletree = None
        vsb = None
        vsb2 = None
        cols = []

        if type_cols == "time":
            tabletree = self.tabletimetree
            subtabletree = self.subtabletimetree
            vsb = self.vsbtime
            vsb2 = self.vsbtimehorizontal
            cols = self.timecols

        elif type_cols == "speed":
            tabletree = self.tablespeedtree
            subtabletree = self.subtablespeedtree
            vsb = self.vsbspeed
            vsb2 = self.vsbspeedhorizontal
            cols = self.speedcols

        elif type_cols == "area":
            tabletree = self.tableareatree
            subtabletree = self.subtableareatree
            vsb = self.vsbarea
            vsb2 = self.vsbareahorizontal
            cols = self.areacols

        child_row = None
        if self.current_tabletree:
            curItem = self.current_tabletree.focus()
            if curItem:
                curItemDecomp = self.current_tabletree.item(curItem)
                child_row = int(curItemDecomp["text"])
            self.current_tabletree.grid_forget()
            self.current_subtabletree.grid_forget()
            self.current_tablevsb.grid_forget()
            self.current_tablevsb2.grid_forget()
        else:
            if len(tabletree.selection()) > 0:
                tabletree.selection_remove(tabletree.selection()[0])
            if len(subtabletree.selection()) > 0:
                subtabletree.selection_remove(subtabletree.selection()[0])

        tabletree.delete(*tabletree.get_children())
        subtabletree.delete(*subtabletree.get_children())

        tabletree.grid_forget()
        subtabletree.grid_forget()
        vsb.grid_forget()
        vsb2.grid_forget()

        tabletree.grid(row=0, column=0, sticky=tk.NSEW)
        subtabletree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0,column=1,sticky=tk.NS)
        vsb2.grid(row=1,column=0,sticky=tk.EW)

        self.current_tabletree = tabletree
        self.current_subtabletree = subtabletree
        self.current_tablevsb = vsb
        self.current_tablevsb2 = vsb2
        i = 0
        avg_array = [0.0 for col in cols]
        for peak in self.peaks:
            print(peak)
            value_array = []
            j = 0
            for col in cols:
                value_array.append(peak.parameters[col])
                avg_array[j] += peak.parameters[col]
                j += 1
            value_array = tuple(value_array)
            cur_tag = "oddrow"
            if i % 2 == 0:
                cur_tag = "evenrow"
            self.current_tabletree.insert("", "end", values=value_array, text=str(i), tags = (cur_tag,))
            i += 1
        if len(self.peaks) > 0:
            avg_array = [float("{:.3f}".format(a / len(self.peaks))) for a in avg_array]
            self.current_subtabletree.insert("", "end", values=avg_array)
        if child_row != None:
            for child_item in self.current_tabletree.get_children():
                curItemDecomp = self.current_tabletree.item(child_item)
                current_row = int(curItemDecomp["text"])
                if current_row == child_row:
                    self.current_tabletree.focus(child_item)
                    self.current_tabletree.selection_set(child_item)
        self.controller.btn_lock = False

    def tabletreeselection(self, event=None):
        self.controller.btn_lock = True
        #https://www.tcl.tk/man/tcl8.5/TkCmd/ttk_treeview.htm#M-selectmode
        self.ax.clear()
        self.ax.set_xlabel("Time("+self.controller.current_timescale+")")
        self.ax.set_ylabel("Average Speed("+self.controller.current_speedscale+")")

        self.mainplotartist = None
        try:
            curItem = self.current_tabletree.focus()
            curItemDecomp = self.current_tabletree.item(curItem)
            curRow = int(curItemDecomp["text"])
            if self.plotsettings.plotline_opts["absolute_time"] == True:
                self.mainplotartist = self.ax.plot(self.peaks[curRow].peaktimes, self.peaks[curRow].peakdata, color=self.plotsettings.peak_plot_colors["main"])
                self.ax.plot(self.peaks[curRow].firsttime, self.peaks[curRow].firstvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["first"], picker=5)
                self.ax.plot(self.peaks[curRow].secondtime, self.peaks[curRow].secondvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
                self.ax.plot(self.peaks[curRow].thirdtime, self.peaks[curRow].thirdvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["min"], picker=5)
                self.ax.plot(self.peaks[curRow].fourthtime, self.peaks[curRow].fourthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
                self.ax.plot(self.peaks[curRow].fifthtime, self.peaks[curRow].fifthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)
            else:
                zerotime = self.peaks[curRow].firsttime
                self.mainplotartist = self.ax.plot([ttime - zerotime for ttime in self.peaks[curRow].peaktimes], self.peaks[curRow].peakdata, color=self.plotsettings.peak_plot_colors["main"])
                self.ax.plot(self.peaks[curRow].firsttime - zerotime, self.peaks[curRow].firstvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["first"], picker=5)
                self.ax.plot(self.peaks[curRow].secondtime - zerotime, self.peaks[curRow].secondvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
                self.ax.plot(self.peaks[curRow].thirdtime - zerotime, self.peaks[curRow].thirdvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["min"], picker=5)
                self.ax.plot(self.peaks[curRow].fourthtime - zerotime, self.peaks[curRow].fourthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
                self.ax.plot(self.peaks[curRow].fifthtime - zerotime, self.peaks[curRow].fifthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)
        except ValueError:
            pass

        curlims = (self.ax.get_xlim(), self.ax.get_ylim())
        self.fig.tight_layout()

        if self.axbaseline != None:
            self.axbaseline.remove()
            self.axbaseline = None
        if self.plotsettings.plotline_opts["zero"] == True:
            self.axbaseline = self.ax.axhline(y=0.0, color=self.plotsettings.plotline_opts["zero_color"], linestyle='-')

        if self.axgrid != None:
            self.axgrid.remove()
            self.axgrid = None
        if self.plotsettings.plotline_opts["grid"] == True:
            self.axgrid = self.ax.grid(linestyle="-", color=self.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.ax.grid(False)
        
        self.ax.set_xlim(curlims[0])
        self.ax.set_ylim(curlims[1])

        self.fig.canvas.draw()
        self.controller.btn_lock = False
    
    def remove_item(self):
        self.controller.btn_lock = True
        MsgBox = messagebox.askyesno(title='Remove Confirmation', message="Confirm Removing Wave? (This action cannot be undone on this page)")
        if MsgBox == True:
            selected_items = self.current_tabletree.selection()
            if len(selected_items) > 0:
                if selected_items[0] and selected_items[0] == self.peakRow_iid:
                    self.ax.clear()
                    self.fig.tight_layout()
                    self.fig.canvas.draw()
                    #child_row = None
            current_row = int(self.current_tabletree.item(self.peakRow_iid)["text"])
            self.current_tabletree.delete(self.peakRow_iid)
            del self.peaks[current_row]
            self.settabletype()
        self.controller.btn_lock = False

    def savepeaks(self):
        self.controller.peaks = self.peaks.copy()
        MsgBox = messagebox.askyesno(title='Save Selected Waves', message="Save Waves on disk?")
        if MsgBox == True:
            pksobj = PeaksObj()
            pksobj.thisgroup = self.controller.current_analysis
            pksobj.thisframes = self.controller.selectedframes
            pksobj.peaks = self.controller.peaks
            if not os.path.exists('savedwaves'):
                os.makedirs('savedwaves/')
            f = filedialog.asksaveasfile(title = "Save Selected Waves", mode='w', initialdir="./savedwaves/")
            if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            f.close()
            try:
                fname = str(f.name)
                fname = r'%s' %fname
                fname2 = fname.split(".")[0] + ".pickle"
                filehandler = open(r'%s' %fname2 , 'wb') 
                # pickle.dump(pksobj, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(pksobj, filehandler, protocol=3)
                filehandler.close()
            except Exception as e:
                messagebox.showerror("Error", "Could not save Selected Waves file\n" + str(e))

    def quiverjetgo(self):
        if self.controller.btn_lock == False:
            self.controller.btn_lock = True
            #Configure export Menus and Options
            selected_items = self.current_tabletree.selection()
            if len(selected_items) > 0:
                if selected_items[0]:
                    #QuiverJetProgress
                    self.savepeaks()

                    self.controller.showwd()
                    # self.controller.update()
                    # wd.progress_bar.start()
                    # self.update_idletasks()

                    current_row = int(self.current_tabletree.item(selected_items[0])["text"])
                    self.controller.current_peak = self.peaks[current_row]
                    self.controller.current_framelist = []
                    self.controller.current_maglist = []
                    self.controller.current_anglist = []
                    if self.controller.current_analysis.gtype == "Folder":
                        global img_opencv
                        files_grabbed = [x for x in os.listdir(self.controller.current_analysis.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
                        framelist = sorted(files_grabbed)
                        files_grabbed_now = framelist[self.controller.selectedframes[self.controller.current_peak.first]:(self.controller.selectedframes[self.controller.current_peak.last+1])+1]
                        files_grabbed_now = [self.controller.current_analysis.gpath + "/" + a for a in files_grabbed_now]

                        for j in range(len(files_grabbed_now)-1):
                            frame1 = cv2.imread(r'%s' %files_grabbed_now[0+j])
                            frame2 = cv2.imread(r'%s' %files_grabbed_now[1+j])
                    
                            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                            prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.controller.current_analysis.pyr_scale, self.controller.current_analysis.levels, self.controller.current_analysis.winsize, self.controller.current_analysis.iterations, self.controller.current_analysis.poly_n, self.controller.current_analysis.poly_sigma, 0)

                            U=flow[...,0]
                            V=flow[...,1]

                            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                            self.controller.current_framelist.append(frame1)
                            self.controller.current_maglist.append(mag * self.controller.current_peak.FPS * self.controller.current_peak.pixel_val)
                            self.controller.current_anglist.append((U,V))

                        self.controller.current_framelist = np.array(self.controller.current_framelist)
                        self.controller.current_maglist = np.array(self.controller.current_maglist)
                        self.controller.current_anglist = np.array(self.controller.current_anglist)
                        self.controller.btn_lock = False
                        self.controller.cancelwd()
                        self.controller.show_frame("PageSix")
                        return
                    elif self.controller.current_analysis.gtype == "Video":
                        vc = cv2.VideoCapture(r'%s' %self.controller.current_analysis.gpath)

                        count = self.controller.selectedframes[self.controller.current_peak.first]
                        vc.set(1, count-1)
                        _, frame1 = vc.read()

                        while(vc.isOpened() and count < (self.controller.selectedframes[self.controller.current_peak.last]+1)):
                            _, frame2 = vc.read()
                            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                            prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.controller.current_analysis.pyr_scale, self.controller.current_analysis.levels, self.controller.current_analysis.winsize, self.controller.current_analysis.iterations, self.controller.current_analysis.poly_n, self.controller.current_analysis.poly_sigma, 0)

                            U=flow[...,0]
                            V=flow[...,1]

                            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                            self.controller.current_framelist.append(frame1)
                            self.controller.current_maglist.append(mag * self.controller.current_peak.FPS * self.controller.current_peak.pixel_val)
                            self.controller.current_anglist.append((U,V))

                            frame1 = frame2.copy()
                            count += 1
                        self.controller.current_framelist = np.array(self.controller.current_framelist)
                        self.controller.current_maglist = np.array(self.controller.current_maglist)
                        self.controller.current_anglist = np.array(self.controller.current_anglist)
                        self.controller.btn_lock = False
                        self.controller.cancelwd()
                        self.controller.show_frame("PageSix")
                        return
                    elif self.controller.current_analysis.gtype == "Tiff Directory" or self.controller.current_analysis.gtype == "CTiff":
                        _, images = cv2.imreadmulti(r'%s' %self.controller.current_analysis.gpath, None, cv2.IMREAD_COLOR)

                        images = images[self.controller.selectedframes[self.controller.current_peak.first]:self.controller.selectedframes[(self.controller.current_peak.last+1)+1]]
                        for j in range(len(images)-1):
                            frame1 = images[0+j]
                            frame2 = images[1+j]
                    
                            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                            prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                            flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.controller.current_analysis.pyr_scale, self.controller.current_analysis.levels, self.controller.current_analysis.winsize, self.controller.current_analysis.iterations, self.controller.current_analysis.poly_n, self.controller.current_analysis.poly_sigma, 0)

                            U=flow[...,0]
                            V=flow[...,1]

                            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                            self.controller.current_framelist.append(frame1)
                            self.controller.current_maglist.append(mag * self.controller.current_peak.FPS * self.controller.current_peak.pixel_val)
                            self.controller.current_anglist.append((U,V))

                        self.controller.current_framelist = np.array(self.controller.current_framelist)
                        self.controller.current_maglist = np.array(self.controller.current_maglist)
                        self.controller.current_anglist = np.array(self.controller.current_anglist)
                        self.controller.btn_lock = False
                        self.controller.cancelwd()
                        self.controller.show_frame("PageSix")
                        return
            messagebox.showerror("Error", "No Waves selected")
            self.controller.btn_lock = False
    
    def exportplotimage(self):
        self.controller.btn_lock = True
        formats = set(self.fig.canvas.get_supported_filetypes().keys())
        d = SaveFigureDialog(self, title='Save Figure', literals=[
            ("formats", formats),
            ("bbox", 1)
        ])
        if d.result != None:
            if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                self.fig.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
            else:
                self.fig.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
        self.controller.btn_lock = False

    def genexportdata(self, cols):
        rows = [[] for col in cols]
        avg_array = [0.0 for col in cols]
        for peak in self.peaks:
            j = 0
            for col in cols:
                rows[j].append(peak.parameters[col])
                avg_array[j] += peak.parameters[col]
                j += 1
        avg_array = [[float("{:.3f}".format(a / len(self.peaks)))] for a in avg_array]
        return [rows, avg_array]

    def exportdata(self, exptype):
        self.controller.btn_lock = True
        if exptype == "plot":
            try:
                SaveTableDialog(self, title='Save Table', literals=[
                    ("headers", [self.ax.get_xlabel(), self.ax.get_ylabel()]),
                    ("data", [self.mainplotartist[0].get_xdata(), self.mainplotartist[0].get_ydata()]),
                    ("data_t", "single")
                    ])
            except TypeError:
                messagebox.showerror("Error", "No Data in plot")
        elif exptype == "time":
            cols = self.timecols
            timeavgcols = ["Avg." + a for a in cols]
            rows, avg_array = self.genexportdata(cols)
            SaveTableDialog(self, title='Save Table', literals=[
                ("headers", [timeavgcols,cols]),
                ("data", [avg_array, rows]),
                ("sheetnames", ["Avg. Time","Time"]),
                ("data_t", "multiple")
                ])
        elif exptype == "speed":
            cols = self.speedcols
            speedavgcols = ["Avg." + a for a in cols]
            rows, avg_array = self.genexportdata(cols)
            SaveTableDialog(self, title='Save Table', literals=[
                ("headers", [speedavgcols,cols]),
                ("data", [avg_array, rows]),
                ("sheetnames", ["Avg. Speed","Speed"]),
                ("data_t", "multiple")
                ])
        elif exptype == "area":
            cols = self.areacols
            areaavgcols = ["Avg." + a for a in cols]
            rows, avg_array = self.genexportdata(cols)
            SaveTableDialog(self, title='Save Table', literals=[
                ("headers", [areaavgcols,cols]),
                ("data", [avg_array, rows]),
                ("sheetnames", ["Avg. Area","Area"]),
                ("data_t", "multiple")
                ])
        elif exptype == "all":
            timecols = self.timecols
            timeavgcols = ["Avg." + a for a in timecols]
            timerows, timeavgrows= self.genexportdata(timecols)
            speedcols = self.speedcols
            speedavgcols = ["Avg." + a for a in speedcols]
            speedrows, speedavgrows = self.genexportdata(speedcols)
            areacols = self.areacols
            areaavgcols = ["Avg." + a for a in areacols]
            arearows, areaavgrows = self.genexportdata(areacols)
            cols = [timeavgcols, speedavgcols, areaavgcols, timecols, speedcols, areacols]
            rows = [timeavgrows, speedavgrows, areaavgrows, timerows, speedrows, arearows]
            SaveTableDialog(self, title='Save Table', literals=[
                ("headers", cols),
                ("data", rows),
                ("sheetnames", ["Avg. Time","Avg. Speed","Avg. Area", "Time", "Speed","Area"]),
                ("data_t", "multiple")
                ])
        self.controller.btn_lock = False

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)

        exportMenu = tk.Menu(menubar, tearoff=0)
        exportMenu.add_command(label="Export Data", command=lambda: self.exportdata("all"))
        exportMenu.add_command(label="Export Plot Data", command=lambda: self.exportdata("plot"))
        exportMenu.add_command(label="Export Plot Figure", command=self.exportplotimage)

        menubar.add_cascade(label="File", menu=pageMenu)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_cascade(label="Export", menu=exportMenu)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)
        return menubar

class PageSix(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        # self.configure(bg=self.controller.bgcolor)
        # self.configure(style='greyBackground.TFrame')
        self.fname = "PageSix"
        self.current_peak = controller.current_peak
        self.plotsettings = controller.plotsettings
        self.mainplotartist = None
        self.rect = None
        self.maximize = False
        self.maximizeplot = None
        self.dblclickcid = None
        self.advsettings = None

        self.current_frame = None
        self.current_image = None
        self.current_mag = None
        self.current_ang = None

        self.current_framelist = []
        self.current_maglist = None
        self.current_anglist = None
        self.anijob = None
        self.current_jetscalemax = None
        self.current_jetscalemin = None
        self.magfilter = None

        self.current_windowX = None
        self.current_windowY = None
        
        self.cb = None
        self.blur_size = None
        self.kernel_dilation = None
        self.kernel_erosion = None
        self.kernel_smoothing_contours = None
        self.border_thickness = None
        self.plot = None

        self.rect = None
        self.half_time = None

        self.timeanimation = 100
        self.animation_status = False
        self.pressmovecid = None

        for i in range(0,15):
            self.rowconfigure(i, weight=1)
        for i in range(0,3):
            self.columnconfigure(i, weight=1)

        label = ttk.Label(self, text="Wave Quiver/Jet Plottting", font=controller.title_font)#, style="greyBackground.TLabel")
        label.grid(row=0, column=0, columnspan=3)

        self.frame_checks = ttk.Frame(self)

        self.CheckMerge = tk.IntVar()
        self.CheckJet = tk.IntVar()
        self.CheckQuiver = tk.IntVar()
        self.CheckLegend = tk.IntVar()
        self.CheckContour = tk.IntVar()

        self.CheckMerge.set(0)
        self.CheckJet.set(0)
        self.CheckQuiver.set(0)

        self.CheckLegend.set(0)
        self.CheckContour.set(0)

        self.CheckBtnMerge = ttk.Checkbutton(self.frame_checks, text = "Plot. Image", variable = self.CheckMerge, \
                         onvalue = 1, offvalue = 0, command = self.init_viz)#, style="greyBackground.TCheckbutton")

        CreateToolTip(self.CheckBtnMerge , \
    "Adds current image to Figure.")

        self.CheckBtnJet = ttk.Checkbutton(self.frame_checks, text = "Plot. Jet", variable = self.CheckJet, \
                         onvalue = 1, offvalue = 0, command = self.init_viz)#, style="greyBackground.TCheckbutton")

        CreateToolTip(self.CheckBtnJet, \
    "Adds Speed Heatmap to Figure.")

        self.CheckBtnQuiver = ttk.Checkbutton(self.frame_checks, text = "Plot. Quiver", variable = self.CheckQuiver, \
                         onvalue = 1, offvalue = 0, command = self.init_viz)#, style="greyBackground.TCheckbutton")

        CreateToolTip(self.CheckBtnQuiver, \
    "Adds Quiver Plot to Figure.")

        self.CheckBtnLegend = ttk.Checkbutton(self.frame_checks, text = "Add. Legend", variable = self.CheckLegend, \
                         onvalue = 1, offvalue = 0, command = self.init_viz)#, style="greyBackground.TCheckbutton")


        CreateToolTip(self.CheckBtnLegend, \
    "Adds Legend to Figure if Jet or Quiver Plots are selected.")

        self.CheckBtnContour = ttk.Checkbutton(self.frame_checks, text = "Contour Figure", variable = self.CheckContour, \
                         onvalue = 1, offvalue = 0, command = self.init_viz)#, style="greyBackground.TCheckbutton")

        CreateToolTip(self.CheckBtnContour, \
    "Contours current Cell Jet/Quiver Plots. Contouring can be edited in the 'Advanced Configs' Menu at the Top Bar")

        self.CheckBtnMerge.grid(row=0, column=0, sticky=tk.NSEW)
        self.CheckBtnJet.grid(row=0, column=1, sticky=tk.NSEW)
        self.CheckBtnQuiver.grid(row=0, column=2, sticky=tk.NSEW)

        self.CheckBtnLegend.grid(row=0, column=3, sticky=tk.NSEW)
        self.CheckBtnContour.grid(row=0, column=4, sticky=tk.NSEW)

        self.CheckMerge.set(1)
        
        for i in range(0,2):
            self.frame_checks.rowconfigure(i, weight=1)
        for i in range(0,4):
            self.frame_checks.columnconfigure(i, weight=1)
        
        self.frame_checks.grid(row=1, column=0, rowspan=2, columnspan=3, sticky=tk.NSEW)

        self.fig = plt.figure(figsize=(4, 3), dpi=100 ,facecolor=self.controller.bgcolor, edgecolor="None")
        self.gs = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2, left=None, bottom=None, right=None, top=None)

        self.gs_noborder = gridspec.GridSpec(1, 1, height_ratios=[5], left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.frame_canvas = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_canvas)  # A tk.DrawingArea.
        self.ax = self.fig.add_subplot(self.gs[0])
        self.orilocator = self.ax.get_axes_locator()
        self.cax = None

        self.fpos = self.gs[0].get_position(self.fig)

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame_canvas.grid(row=3, column=0, rowspan=4, columnspan=3, sticky=tk.NSEW)

        self.fig2 = plt.figure(figsize=(3, 2), dpi=100, facecolor=self.controller.bgcolor)
        self.fig2.tight_layout()
        self.fig2.subplots_adjust(top=0.85, bottom=0.25)
        # self.fig2.tight_layout(rect=[0, 0.03, 0.8, 0.95])
        # self.fig2.tight_layout(rect=[0, 0.03, 0.8, 0.95])
        self.gs2 = gridspec.GridSpec(1, 1, height_ratios=[3], hspace=0.5)

        self.frame_canvas2 = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame_canvas2)  # A tk.DrawingArea.
        self.ax2 = self.fig2.add_subplot()
        self.axbaseline = None
        self.axgrid = None
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.frame_canvas2.grid(row=7, column=0, rowspan=4, columnspan=3, sticky=tk.NSEW)

        self.slidervar = tk.IntVar()
        self.slider = ttk.Scale(self, from_=1, to=len(self.current_framelist), variable=self.slidervar, orient=tk.HORIZONTAL, command=self.update_frame)#, style="greyBackground.Horizontal.TScale")
        self.slider.grid(row=12, column=0, columnspan=3, sticky=tk.NSEW)

        self.frame_3 = ttk.Frame(self)#, style="greyBackground.TFrame")
        
        self.startstopanimationbtn = ttk.Button(self.frame_3, command=self.startstopanimation)#, style="greyBackground.TButton")
        self.startstopanimationbtn.config(image=self.controller.playstopicon)#, width=30, height=30)

        CreateToolTip(self.startstopanimationbtn, \
    "Starts Figure animation for whole wave. Warning: Might lag at high FPS values")

        tlbl1 = ttk.Label(self.frame_3, text="Play/Stop:")#, style="greyBackground.TLabel")
        tlbl1.grid(row=0, column=0, sticky=tk.NSEW)

        self.frame_3.grid(row=13, column=1, sticky=tk.NSEW)
        for i in range(0,1):
            self.frame_3.rowconfigure(i, weight=1)
        
        self.startstopanimationbtn.grid(row=0, column=1)

        self.frame_4 = ttk.Frame(self)#, style="greyBackground.TFrame")
        self.fpsspin = tk.StringVar()
        self.fpsspin.set("1")
        self.fps_spinner = tk.Spinbox(self.frame_4, from_=1, to=10000000, textvariable=self.fpsspin, increment=1, width=10, command=self.set_framerate)

        CreateToolTip(self.fps_spinner, \
    "Sets Figure animation FPS. Warning: High FPS values might lag animation")

        tlbl2 =ttk.Label(self.frame_4, text="FPS:")#, style="greyBackground.TLabel")
        tlbl2.grid(row=0, column=0, sticky=tk.NSEW)

        self.fps_spinner.grid(row=0, column=1,pady=20, sticky=tk.NSEW)
        self.frame_4.grid(row=13, column=2, sticky=tk.NSEW)
        for i in range(0,1):
            self.frame_4.rowconfigure(i, weight=1)

        #row 12

        # self.gotostartpage = tk.PhotoImage(file="icons/refresh-sharp.png")
        # self.goback = tk.PhotoImage(file="icons/arrow-back-sharp.png")

        self.frame5 = ttk.Frame(self)
        btn1frame = ttk.Frame(self.frame5)
        btn1lbl = ttk.Label(btn1frame, text="Go back")
        btn1lbl.grid(row=0, column=1)
        button_go_back = ttk.Button(btn1frame, image=self.controller.goback,
                           command=lambda: controller.show_frame("PageFive"))
        button_go_back.image=self.controller.goback
        button_go_back.grid(row=0, column=0)
        btn1frame.grid(row=0, column=3)

        btn2frame = ttk.Frame(self.frame5)
        btn2lbl = ttk.Label(btn2frame, text="Go to the start page")
        btn2lbl.grid(row=0, column=1)
        button_go_start = ttk.Button(btn2frame, image=self.controller.gotostartpage,
                           command=lambda: controller.show_frame("StartPage"))
        button_go_start.image=self.controller.gotostartpage
        button_go_start.grid(row=0, column=0)
        btn2frame.grid(row=0, column=1)

        for i in range(0,1):
            self.frame5.rowconfigure(i, weight=1)
        for i in range(0,5):
            self.frame5.columnconfigure(i, weight=1)
        self.frame5.grid(row=14, column=0, rowspan=1, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

    def update_config(self, etype, eval1):
        if etype == "current_windowX":
            self.current_windowX = eval1
        if etype == "current_windowY":
            self.current_windowY = eval1
        if etype == "blur_size":
            self.blur_size = eval1
        if etype == "kernel_dilation":
            self.kernel_dilation = eval1
        if etype == "kernel_erosion":
            self.kernel_erosion = eval1
        if etype == "kernel_smoothing_contours":
            self.kernel_smoothing_contours = eval1
        if etype == "border_thickness":
            self.border_thickness = eval1
        if etype == "minscale":
            self.current_jetscalemin = eval1
        if etype == "maxscale":
            self.current_jetscalemax = eval1
        self.update_frame()

    def update_all_settings(self, resultobj):
        print("about to update window")
        self.current_windowX = resultobj["current_windowX"]
        self.current_windowY = resultobj["current_windowY"]
        self.blur_size = resultobj["blur_size"]
        self.kernel_dilation = resultobj["kernel_dilation"]
        self.kernel_erosion = resultobj["kernel_erosion"]
        self.kernel_smoothing_contours = resultobj["kernel_smoothing_contours"]
        self.border_thickness = resultobj["border_thickness"]
        self.current_jetscalemin = resultobj["minscale"]
        self.current_jetscalemax = resultobj["maxscale"]
        self.update_frame()
        self.controller.btn_lock = True
        self.delete_settings()
        self.controller.btn_lock = False

    def delete_settings(self):
        self.advsettings = None

    def change_settings(self, event=None):
        self.controller.btn_lock = True
        global default_values_bounds

        if self.advsettings == None:
            self.advsettings = QuiverJetSettings(self, title='Edit Advanced Configs', literals=[
                ("config", default_values_bounds),
                ("current_windowX", ("Quiver Window X", self.current_windowX)),
                ("current_windowY", ("Quiver Window Y", self.current_windowY)),
                ("blur_size", ("Blur Size", self.blur_size)),
                ("kernel_dilation", ("Kernel Dilation", self.kernel_dilation)),
                ("kernel_erosion", ("Kernel Erosion", self.kernel_erosion)),
                ("kernel_smoothing_contours", ("Kernel Smoothing Contours", self.kernel_smoothing_contours)),
                ("border_thickness", ("Border Thickness", self.border_thickness)),
                ("minscale", ("Scale Min.", self.current_jetscalemin)),
                ("maxscale", ("Scale Max.", self.current_jetscalemax)),
                ("updatable_frame", self)
                ])
            self.controller.btn_lock = False

    def update_frame(self, *args):
        global img_opencv

        self.controller.btn_lock = True
        current_fnum = self.slidervar.get() - 1
        self.current_frame = current_fnum
        
        if self.rect != None:
            self.rect.remove()
        self.rect = None
        curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())
        new_rect = Rectangle((0,0), 1, 1)
        new_rect.set_width(2 * self.half_time)
        # new_rect.set_height(np.max(self.current_peak.peakdata) + abs(np.min(self.current_peak.peakdata)) + 0.5)
        new_rect.set_height(curlims[1][1] + abs(curlims[1][0]) + 0.5)

        if self.plotsettings.plotline_opts["absolute_time"] == False:
            zerotime = self.current_peak.firsttime
            # new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time - zerotime, np.min(self.current_peak.peakdata)))
            new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time - zerotime, curlims[1][0]))
        else:
            # new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time, np.min(self.current_peak.peakdata)))
            new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time, curlims[1][0]))
        
        new_rect.set_facecolor(self.plotsettings.peak_plot_colors['rect_color'])

        if self.controller.current_analysis.gtype == "Folder":
            files_grabbed = [x for x in os.listdir(self.controller.current_analysis.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
            framelist = sorted(files_grabbed)
            files_grabbed_now = framelist[self.controller.current_peak.first:(self.controller.current_peak.last+1)+1]
            self.ax2.title.set_text(files_grabbed_now[self.current_frame])

        elif self.controller.current_analysis.gtype == "Video":
            self.ax2.title.set_text(os.path.basename(self.controller.current_analysis.gpath) + " Frame: " + str(self.controller.current_peak.first+self.current_frame))

        elif self.controller.current_analysis.gtype == "Tiff Directory" or self.controller.current_analysis.gtype == "CTiff":
            self.ax2.title.set_text(os.path.basename(self.controller.current_analysis.gpath) + " Img: " + str(self.controller.current_peak.first+self.current_frame))

        self.ax2.set_xlim(curlims[0])
        self.ax2.set_ylim(curlims[1])
        self.rect = self.ax2.add_patch(new_rect)
        # self.fig2.tight_layout()
        self.fig2.canvas.draw()
        self.init_viz()
        self.controller.btn_lock = False
        return True

    def animate_frame(self):
        new_val = self.slidervar.get()+1
        if new_val <= len(self.current_framelist):
            self.slidervar.set(new_val)
        else:
            self.slidervar.set(1)
        a = self.update_frame()
        if a and self.animation_status == True:
            self.anijob = self.after(self.timeanimation, self.animate_frame)

    def startstopanimation(self):
        self.animation_status = not self.animation_status
        if self.animation_status == True:
            self.animate_frame()
        elif self.anijob != None:
            self.after_cancel(self.anijob)
            self.anijob = None

    def set_framerate(self):
        try:
            third = int(self.fps_spinner.get())
            if third < 1 or third > 9999999:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
            else:
                #all good set frame rate
                self.timeanimation = int(1000 / third)
        except Exception as e:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )

    def slowanimation(self):
        self.timeanimation = self.timeanimation + 10

    def speedanimation(self):
        newdelta = self.timeanimation - 10
        if newdelta > 0:
            self.timeanimation = newdelta

    def clear_maximize(self):
        self.maximizeplot = None
        self.maximize = False

    def init_vars(self):
        self.controller.btn_lock = True
        validate = True
        if self.controller.peaks == None:
            messagebox.showerror("Error", "Waves/Flow Objects not found")
            validate = False
        if len(self.controller.current_framelist) != len(self.controller.current_peak.peaktimes) or len(self.controller.current_peak.peakdata) != len(self.controller.current_peak.peaktimes):
            messagebox.showerror("Error", "Error in Flow Objects generation")
            validate = False
        if validate == True:
            self.anijob = None
            self.current_peak = self.controller.current_peak
            self.current_frame = 0
            self.current_framelist = self.controller.current_framelist
            self.current_maglist = self.controller.current_maglist
            self.current_anglist = self.controller.current_anglist
            self.mainplotartist = None
            self.plot = None
            self.maximize = False
            self.maximizeplot = None
            self.dblclickcid = None
            self.advsettings = None

            #calculate X and Y window from image automatically
            self.current_jetscalemax = np.max(self.current_maglist.flatten())
            if self.controller.current_analysis.noisemin is None:
                self.current_jetscalemin = np.min(self.current_maglist.flatten())
            else:
                self.current_jetscalemin = self.controller.current_analysis.noisemin


            Y,X=self.current_anglist[0][0].shape

            global default_values_bounds
            default_values_bounds["current_windowX"][1] = int(X)
            default_values_bounds["current_windowY"][1] = int(Y)
            default_values_bounds["kernel_dilation"][1] = int(np.min([X,Y]))
            default_values_bounds["kernel_erosion"][1] = int(np.min([X,Y]))
            default_values_bounds["kernel_smoothing_contours"][1] = int(np.min([X,Y]))
            default_values_bounds["minscale"] = [0, 100000000000]
            default_values_bounds["maxscale"] = [0, 100000000000]
            
            gcdt =  np.gcd(X,Y)
            self.current_windowX = np.max([int(X / gcdt), 8])
            self.current_windowY = np.max([int(Y / gcdt), 15])

            self.magfilter = None

            self.slider["to"] = len(self.current_framelist)
            self.slider.grid_forget()
            self.slider.grid(row=11, column=0, columnspan=3, sticky=tk.NSEW)

            self.CheckMerge.set(1)
            self.CheckJet.set(0)
            self.CheckQuiver.set(0)

            self.slidervar.set(self.current_frame+1)

            self.cb = None
            self.blur_size = 5
            self.kernel_dilation = 3
            self.kernel_erosion = 11
            self.kernel_smoothing_contours = 5
            self.border_thickness = 60

            if self.rect != None:
                self.rect.remove()

            self.rect = None
            self.half_time = None

            self.timeanimation = 100
            self.animation_status = False
            self.pressmovecid = None
            self.controller.btn_lock = False
            return True
        self.controller.btn_lock = False
        return False

    def dblclickev(self, event):
        if event.button == 1 and event.dblclick == True and self.maximize == False:
            if self.maximizeplot == None:
                self.maximizeplot = QuiverJetMaximize(self, title="", literals=[
                    ("updatable_frame", self)
                    ])
                U = self.controller.current_anglist[self.current_frame][0]
                Y,X=U.shape
                self.maximizeplot.set_figure(X, Y)
                self.init_viz()

    def hide_ax(self, tax):
        tax.set_xticks([], [])
        tax.set_yticks([], [])
        tax.set_xticklabels([])
        tax.set_yticklabels([])

    def init_viz(self):
        self.controller.btn_lock = True
        #here viz is reset and plotted according to selection
        self.ax.clear()
        if self.maximizeplot != None:
            self.maximizeplot.axmax.clear()

        self.fig.canvas.mpl_disconnect(self.dblclickcid)
        self.dblclickcid = None
        self.fig.canvas.mpl_connect("button_press_event", self.dblclickev)
        
        self.hide_ax(self.ax)
        self.ax.set_facecolor('black')
        if self.maximizeplot != None:
            self.maximizeplot.axmax.set_xticks([], [])
            self.maximizeplot.axmax.set_yticks([], [])
            self.maximizeplot.axmax.set_xticklabels([])
            self.maximizeplot.axmax.set_yticklabels([])
            self.maximizeplot.axmax.set_facecolor('black')
        
        if self.CheckMerge.get() == 0 and self.CheckJet.get() == 0 and self.CheckQuiver.get() == 0:
            self.ax.axis('off')
            if self.maximizeplot != None:
                self.maximizeplot.axmax.axis('off')
        else:
            self.ax.axis('on')
            if self.maximizeplot != None:
                self.maximizeplot.axmax.axis('on')

        jetalpha = 1.0
        img = self.current_framelist[self.current_frame]
        mag = np.array(self.controller.current_maglist[self.current_frame])
        mag = mag
        
        # mag = np.ma.masked_where(mag < 0.08, mag)
        mag = np.ma.masked_where(mag < self.current_jetscalemin, mag)

        self.plot = None

        cmap = plt.cm.jet
        cmap.set_bad(color=(1, 1, 1, 0.0))

        if self.magfilter != None:
            fil = self.magfilter
            mag[mag<fil]=0 
        if self.CheckContour.get() == 1:
            mask = self.get_contour(img)
            mag = np.ma.masked_where(mask, mag)
        if self.CheckMerge.get() == 1:
            self.ax.imshow(img)
            if self.maximizeplot != None:
                self.maximizeplot.axmax.imshow(img)
            jetalpha = 0.5
        if self.CheckJet.get() == 1:
            if self.CheckQuiver.get() == 1:
                jetalpha = 0.5
            self.plot = self.ax.imshow(mag, clim=[self.current_jetscalemin,self.current_jetscalemax],norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap,alpha=jetalpha)
            if self.maximizeplot != None:
                self.maximizeplot.axmax.imshow(mag, clim=[self.current_jetscalemin,self.current_jetscalemax],norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap,alpha=jetalpha)
        if self.CheckQuiver.get() == 1:
            U = self.controller.current_anglist[self.current_frame][0]
            V = self.controller.current_anglist[self.current_frame][1]

            Y,X=U.shape

            x, y = np.meshgrid(np.arange(X),np.arange(Y))

            skip=(slice(None,None,int(self.current_windowX)),slice(None,None,int(self.current_windowY)))
            
            u_norm = U / np.sqrt(U ** 2.0 + V ** 2.0)
            v_norm = V / np.sqrt(U ** 2.0 + V ** 2.0)
            # u_norm *= 0.8
            a = x[skip]
            b = y[skip]
            p1 = None
            angles_run = 'xy'
            # v_norm *= 0.8

            if self.plot == None:
                self.plot = self.ax.quiver(a, b, u_norm[skip], v_norm[skip],mag[skip],clim=[self.current_jetscalemin,self.current_jetscalemax],angles=angles_run,scale_units='xy',units='xy',pivot='middle',
                        width=2,scale=0.05,norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap)


                if self.maximizeplot != None:
                    self.maximizeplot.axmax.quiver(a, b, u_norm[skip], v_norm[skip],mag[skip],clim=[self.current_jetscalemin,self.current_jetscalemax],angles=angles_run,scale_units='xy',units='xy',pivot='middle',
                        width=2,scale=0.05,norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap)

            else:
                zp = self.ax.quiver(a, b, u_norm[skip], v_norm[skip],mag[skip],clim=[self.current_jetscalemin,self.current_jetscalemax],angles=angles_run,scale_units='xy',units='xy',pivot='middle',
                        width=2,scale=0.05,norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap)

                if self.maximizeplot != None:
                    self.maximizeplot.axmax.quiver(a, b, u_norm[skip], v_norm[skip],mag[skip],clim=[self.current_jetscalemin,self.current_jetscalemax],angles=angles_run,scale_units='xy',units='xy',pivot='middle',
                        width=2,scale=0.05,norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax),cmap=cmap)

            if self.CheckMerge.get() == 0:
                self.ax.set_xlim(0, X)
                self.ax.set_ylim(Y, 0)
                if self.maximizeplot != None:
                    self.maximizeplot.axmax.set_xlim(0, X)
                    self.maximizeplot.axmax.set_ylim(Y, 0)

        cbarticks = None
        if self.CheckLegend.get() == 1:
            if self.cb != None:
                self.cb.remove()
                self.cb = None
            if self.plot != None:
                self.divider = make_axes_locatable(self.ax)
                self.cax = self.divider.append_axes("right", size="2.5%", pad=0.05)
                self.cb = self.fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=self.current_jetscalemin,vmax=self.current_jetscalemax), cmap=cmap), fraction=0.046, pad=0.04, cax=self.cax)

            else:
                self.cb = None
        elif self.cb:
            self.cax.clear()
            self.cb.remove()
            self.divider = None
            self.ax.set_axes_locator(self.orilocator)
            self.ax.reset_position()
            self.cb = None

        self.hide_ax(self.ax)
        self.fig.canvas.draw()
        if self.maximizeplot != None:
            self.hide_ax(self.maximizeplot.axmax)
            self.maximizeplot.figmax.canvas.draw()
            U = self.controller.current_anglist[self.current_frame][0]
            Y,X=U.shape
            self.maximizeplot.set_figure(X, Y)
        self.controller.btn_lock = False

    def auto_canny(self, image, sigma=0.15):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.1 - sigma) * v))
        upper = int(min(255, (0.1 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged

    def get_contour(self, img):
        #ENCONTRAR CONTORNO
        #GAUSSIAN LOW PASS FILTERING
        blur = cv2.GaussianBlur(img,(self.blur_size,self.blur_size),0)
        #CANNY EDGE DETECTION
        auto = self.auto_canny(blur)
        #DILATE BINARY GRADIENT
        edges2 = auto
        kernel = np.ones((self.kernel_dilation,self.kernel_dilation),np.uint8)
        dilation = cv2.dilate(edges2,kernel,iterations = 1)
        #FILLING HOLES
        im_floodfill = dilation.copy()
        h, w = dilation.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8) 
        cv2.floodFill(im_floodfill, mask, (0,0), 255); 
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        FillingHoles = dilation | im_floodfill_inv
        
        #EROSION - EDGE SMOOTHING
        kernel = np.ones((self.kernel_erosion,self.kernel_erosion),np.uint8)
        erosion = cv2.erode(FillingHoles,kernel,iterations = 1)
        #IMAGE OPENING (smoothing contours)
        kernel2 = np.ones((self.kernel_smoothing_contours,self.kernel_smoothing_contours),np.uint8)
        smoothCont = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel2)

        #CHOOSING OBJECT OF INTEREST
        ret, smoothCont = cv2.threshold(smoothCont, 250, 255,0)
        contours2, hierarchy = cv2.findContours(smoothCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(smoothCont.shape, np.uint8)
        largest_areas = sorted(contours2, key=cv2.contourArea)
        
        cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), self.border_thickness)
        cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255))
        cv2.fillPoly(mask, [largest_areas[-1]], (255,255,255,255))
        mask = np.logical_not(mask)
        return mask

    def init_ax2(self):
        global img_opencv
        self.controller.btn_lock = True
        self.mainplotartist = None
        self.ax2.clear()

        if self.controller.current_analysis.gtype == "Folder":
            files_grabbed = [x for x in os.listdir(self.controller.current_analysis.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
            framelist = sorted(files_grabbed)
            files_grabbed_now = framelist[self.controller.current_peak.first:(self.controller.current_peak.last+1)+1]
            self.ax2.set_title(files_grabbed_now[self.current_frame])
            # pass

        elif self.controller.current_analysis.gtype == "Video":
            self.ax2.set_title(os.path.basename(self.controller.current_analysis.gpath) + " Frame: " + str(self.controller.current_peak.first+self.current_frame))
            # pass

        elif self.controller.current_analysis.gtype == "Tiff Directory" or self.controller.current_analysis.gtype == "CTiff":
            self.ax2.set_title(os.path.basename(self.controller.current_analysis.gpath) + " Img: " + str(self.controller.current_peak.first+self.current_frame))
            # pass

        self.ax2.set_xlabel("Time("+self.controller.current_timescale+")")
        self.ax2.set_ylabel("Average Speed("+self.controller.current_speedscale+")")

        if self.plotsettings.plotline_opts["absolute_time"] == True:
            self.mainplotartist = self.ax2.plot(self.current_peak.peaktimes, self.current_peak.peakdata, color=self.plotsettings.peak_plot_colors["main"])
            self.ax2.plot(self.current_peak.firsttime, self.current_peak.firstvalue, , linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["first"], picker=5)
            self.ax2.plot(self.current_peak.secondtime, self.current_peak.secondvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
            self.ax2.plot(self.current_peak.thirdtime, self.current_peak.thirdvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["min"], picker=5)
            self.ax2.plot(self.current_peak.fourthtime, self.current_peak.fourthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
            self.ax2.plot(self.current_peak.fifthtime, self.current_peak.fifthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)
        else:
            zerotime = self.current_peak.firsttime
            self.mainplotartist = self.ax2.plot([ttime - zerotime for ttime in self.current_peak.peaktimes], self.current_peak.peakdata, color=self.plotsettings.peak_plot_colors["main"])
            self.ax2.plot(self.current_peak.firsttime - zerotime, self.current_peak.firstvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["first"], picker=5)
            self.ax2.plot(self.current_peak.secondtime - zerotime, self.current_peak.secondvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
            self.ax2.plot(self.current_peak.thirdtime - zerotime, self.current_peak.thirdvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["min"], picker=5)
            self.ax2.plot(self.current_peak.fourthtime - zerotime, self.current_peak.fourthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["max"], picker=5)
            self.ax2.plot(self.current_peak.fifthtime - zerotime, self.current_peak.fifthvalue, "o", linewidth=2, fillstyle='none', color=self.plotsettings.peak_plot_colors["last"], picker=5)

        self.half_time = self.current_peak.peaktimes[1] - self.current_peak.peaktimes[0]

        curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())

        if self.rect != None:
            self.rect.remove()
        self.rect = None
        new_rect = Rectangle((0,0), 1, 1)
        new_rect.set_width(2 * self.half_time)
        # new_rect.set_height(np.max(self.current_peak.peakdata) + abs(np.min(self.current_peak.peakdata)) + 0.5)
        new_rect.set_height(curlims[1][1] + abs(curlims[1][0]) + 0.5)

        if self.plotsettings.plotline_opts["absolute_time"] == True:
            # new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time, np.min(self.current_peak.peakdata)))
            new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time, curlims[1][0]))
        else:
            zerotime = self.current_peak.firsttime
            # new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time - zerotime, np.min(self.current_peak.peakdata)))
            new_rect.set_xy((self.current_peak.peaktimes[self.current_frame] - self.half_time - zerotime, curlims[1][0]))

        new_rect.set_facecolor(self.plotsettings.peak_plot_colors['rect_color'])

        self.rect = self.ax2.add_patch(new_rect)
        # self.fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.axbaseline != None:
            self.axbaseline.remove()
            self.axbaseline = None
        if self.plotsettings.plotline_opts["zero"] == True:
            self.axbaseline = self.ax2.axhline(y=0.0, color=self.plotsettings.plotline_opts["zero_color"], linestyle='-')
        
        if self.axgrid != None:
            self.axgrid.remove()
            self.axgrid = None
        if self.plotsettings.plotline_opts["grid"] == True:
            self.axgrid = self.ax2.grid(linestyle="-", color=self.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.ax2.grid(False)

        self.ax2.set_xlim(curlims[0])
        self.ax2.set_ylim(curlims[1])
        self.fig2.canvas.draw()

        if self.pressmovecid != None:
            self.fig2.canvas.mpl_disconnect(self.pressmovecid)
            self.pressmovecid = None
        self.pressmovecid = self.fig2.canvas.mpl_connect("button_press_event", self.on_press_event_slider)
        self.fig2.canvas.draw()
        self.controller.btn_lock = False

    def on_press_event_slider(self, event):
        array = None
        if self.plotsettings.plotline_opts["absolute_time"] == True:
            array = np.asarray(self.current_peak.peaktimes)
        else:
            zerotime = self.current_peak.firsttime
            array = np.asarray([ttime - zerotime for ttime in self.current_peak.peaktimes])
        try:
            idx = (np.abs(array - event.xdata)).argmin()
            self.slidervar.set(idx+1)
            a = self.update_frame()
            if a == True:
                return a
        except TypeError:
            pass
    
    def exportplotdata(self):
        self.controller.btn_lock = True
        try:
            d = SaveTableDialog(self, title='Save Table', literals=[
                ("headers", [self.ax2.get_xlabel(), self.ax2.get_ylabel()]),
                ("data", [self.mainplotartist[0].get_xdata(), self.mainplotartist[0].get_ydata()]),
                ("data_t", "single")
                ])
        except TypeError:
            messagebox.showerror("Error", "No Data in plot")
        self.controller.btn_lock = False

    def exportcurrentfig(self):
        self.controller.btn_lock = True
        formats = set(self.fig.canvas.get_supported_filetypes().keys())
        d = SaveFigureDialog(self, title='Save Figure', literals=[
            ("formats", formats),
            ("bbox", 0)
        ])
        if d.result != None:
            if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=extent,pad_inches=0)
            else:
                extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=extent,pad_inches=0)
        self.controller.btn_lock = False

    def exportplotimage(self):
        self.controller.btn_lock = True
        formats = set(self.fig2.canvas.get_supported_filetypes().keys())
        d = SaveFigureDialog(self, title='Save Figure', literals=[
            ("formats", formats),
            ("bbox", 1)
        ])
        if d.result != None:
            if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                self.fig2.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])#,pad_inches=0)
            else:
                self.fig2.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])#,pad_inches=0)
        self.controller.btn_lock = False
    
    def exportlegend(self):
        self.controller.btn_lock = True
        formats = set(self.fig.canvas.get_supported_filetypes().keys())
        d = SaveFigureDialog(self, title='Save Legend', literals=[
            ("formats", formats),
            ("bbox", 0)
        ])
        if d.result != None:
            previous_state = [self.CheckMerge.get(), self.CheckJet.get(), self.CheckQuiver.get()]
            self.CheckMerge.set(0)
            self.CheckJet.set(1)
            self.CheckQuiver.set(0)
            previous_state_legend = self.CheckLegend.get()
            if self.CheckLegend.get() == 0:
                self.CheckLegend.set(1)
                self.update_frame()
            figb,bax = plt.subplots(figsize=(5, 4), dpi=100)
            figb.colorbar(self.plot, fraction=0.046, pad=0.04,ax=bax)
            bax.remove()
            if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                figb.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches="tight",pad_inches=0)
            else:
                figb.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches="tight",pad_inches=0)
            self.CheckMerge.set(previous_state[0])
            self.CheckJet.set(previous_state[1])
            self.CheckQuiver.set(previous_state[2])
            self.CheckLegend.set(previous_state_legend)
            self.update_frame()
        self.controller.btn_lock = False

    def exportfig(self, ftype, framel):
        self.controller.btn_lock = True
        #open save dialog with img/video type
        formats = set(self.fig.canvas.get_supported_filetypes().keys())
        d = SaveFigureVideoDialog(self, title='Save Figure/Video', literals=[
            ("formats", formats),
            ("bbox", 0)
        ])
        if d.result != None:
            #https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
            previous_animation = self.animation_status
            self.animation_status = False
            if self.anijob != None:
                self.after_cancel(self.anijob)
                self.anijob = None
            previous_framenum = self.slidervar.get()
            previous_state = [self.CheckMerge.get(), self.CheckJet.get(), self.CheckQuiver.get()]
            previous_state_legend = self.CheckLegend.get()
            if "merge" in ftype:
                self.CheckMerge.set(1)
            else:
                self.CheckMerge.set(0)
            if "jet" in ftype:
                self.CheckJet.set(1)
            else:
                self.CheckJet.set(0)
            if "quiver" in ftype:
                self.CheckQuiver.set(1)
            else:
                self.CheckQuiver.set(0)
            if self.CheckLegend.get() == 1:
                self.CheckLegend.set(0)
            svideo = None
            if d.result["outtype"] == "video":
                size = self.fig.get_size_inches()*self.fig.dpi
                svideo = cv2.VideoWriter(r'%s' %d.result["name"],cv2.VideoWriter_fourcc(*'MJPG'),d.result["fps"],(int(size[0]), int(size[1])))
            for i in framel:
                self.slidervar.set(i+1)
                self.update_frame()
                if d.result["outtype"] == "image":
                    imgname = d.result["name"].split(".")[0] + "_" + str(i+1) + "." + d.result["name"].split(".")[1]
                    if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                        # self.fig.savefig(imgname,quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
                        self.fig.savefig(r'%s' %imgname,quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=extend, pad_inches=0)
                    else:
                        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                        self.fig.savefig(r'%s' %imgname, dpi=d.result["dpi"], bbox_inches=extent, pad_inches=0)
                if d.result["outtype"] == "video":
                    prevmargins = plt.margins()
                    self.ax.set_position(self.gs_noborder[0].get_position(self.fig))
                    plt.margins(0,0)
                    self.fig.canvas.draw()
                    mat = np.array(self.fig.canvas.renderer._renderer)
                    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                    svideo.write(mat)
            if d.result["outtype"] == "video":
                svideo.release()
                plt.margins(prevmargins[0], prevmargins[1])
                self.ax.set_position(self.fpos)
                self.fig.canvas.draw()
            #reset to previous state
            self.slidervar.set(previous_framenum)
            self.CheckMerge.set(previous_state[0])
            self.CheckJet.set(previous_state[1])
            self.CheckQuiver.set(previous_state[2])
            self.CheckLegend.set(previous_state_legend)
            self.update_frame()
            if previous_animation == True:
                self.startstopanimation()
        self.controller.btn_lock = False

    def menubar(self, root):
        menubar = tk.Menu(root, tearoff=0)
        pageMenu = tk.Menu(menubar, tearoff=0)
        pageMenu.add_command(label="Start Page", command=lambda: self.controller.reset_and_show("StartPage"))
        pageMenu.add_command(label="New Data", command=lambda: self.controller.reset_and_show("PageOne"))
        pageMenu.add_command(label="Check Progress", command=lambda: self.controller.reset_and_show("PageTwo"))
        pageMenu.add_command(label="Load Analysis", command=lambda: self.controller.reset_and_show("PageFour"))
        pageMenu.add_command(label="Load Saved Waves", command=lambda: self.controller.reset_and_show("PageFive"))
        
        plotMenu = tk.Menu(menubar, tearoff=0)
        plotMenu.add_command(label="Edit Plot Settings", command=self.controller.configplotsettings)
        plotMenu.add_command(label="Save Plot Settings", command=self.controller.saveplotsettings)
        plotMenu.add_command(label="Load Plot Settings", command=self.controller.loadplotsettings)

        exportMenu = tk.Menu(menubar, tearoff=0)
        exportMenu.add_command(label="Export Plot Data", command=self.exportplotdata)
        exportMenu.add_command(label="Export Plot Image", command=self.exportplotimage)
        subexportMenu = tk.Menu(exportMenu, tearoff=0)
        subexportMenu.add_command(label="Export Current Figure", command=self.exportcurrentfig)
        subexportMenu.add_command(label="Export Legend", command=self.exportlegend)
        jetexportMenu = tk.Menu(subexportMenu, tearoff=0)
        jetexportMenu.add_command(label="Export Current Jet", command=lambda: self.exportfig("jet", [self.current_frame]))
        jetexportMenu.add_command(label="Export All Jet", command=lambda: self.exportfig("jet", list(range(len(self.current_framelist))) ) )
        quiverexportMenu = tk.Menu(subexportMenu, tearoff=0)
        quiverexportMenu.add_command(label="Export Current Quiver", command=lambda: self.exportfig("quiver", [self.current_frame]) )
        quiverexportMenu.add_command(label="Export All Quiver", command=lambda: self.exportfig("quiver", list(range(len(self.current_framelist))) ) )
        mergeexportMenu = tk.Menu(subexportMenu, tearoff=0)
        mergeexportMenu.add_command(label="Export Current Image/Jet", command=lambda: self.exportfig("mergejet", [self.current_frame]))
        mergeexportMenu.add_command(label="Export All Image/Jet", command=lambda: self.exportfig("mergejet", list(range(len(self.current_framelist))) ))
        mergeexportMenu.add_command(label="Export Current Image/Quiver", command=lambda: self.exportfig("mergequiver", [self.current_frame]))
        mergeexportMenu.add_command(label="Export All Image/Quiver", command=lambda: self.exportfig("mergequiver", list(range(len(self.current_framelist))) ))
        mergeexportMenu.add_command(label="Export Current Jet/Quiver", command=lambda: self.exportfig("jetquiver", [self.current_frame]))
        mergeexportMenu.add_command(label="Export All Jet/Quiver", command=lambda: self.exportfig("jetquiver", list(range(len(self.current_framelist))) ))
        mergeexportMenu.add_command(label="Export Current Image/Jet/Quiver", command=lambda: self.exportfig("mergejetquiver", [self.current_frame]))
        mergeexportMenu.add_command(label="Export All Image/Jet/Quiver", command=lambda: self.exportfig("mergejetquiver", list(range(len(self.current_framelist))) ))
        subexportMenu.add_cascade(label="Jet", menu=jetexportMenu)
        subexportMenu.add_cascade(label="Quiver", menu=quiverexportMenu)
        subexportMenu.add_cascade(label="Merge", menu=mergeexportMenu)
        exportMenu.add_cascade(label="Figure", menu=subexportMenu)
        menubar.add_cascade(label="File", menu=pageMenu)
        menubar.add_cascade(label="Plot Settings", menu=plotMenu)
        menubar.add_cascade(label="Export", menu=exportMenu)
        menubar.add_command(label="Advanced Configs", command=self.change_settings)
        menubar.add_command(label="About", command=self.controller.showabout)
        menubar.add_command(label="Help", command=self.controller.showhelp)
        return menubar

if __name__ == "__main__":
    if _platform == "win32" or _platform == "win64":
        multiprocessing.freeze_support()
    if _platform == "linux" or _platform == "linux2":
        # linux
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    elif _platform == "darwin":
        # MAC OS X
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        # if locale.getlocale()[0] is None:
            # locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    elif _platform == "win32":
        # Windows
        locale.setlocale(locale.LC_ALL, "en-US")
    elif _platform == "win64":
        # Windows 64-bit
        locale.setlocale(locale.LC_ALL, "en-US")
    globalq = multiprocessing.Queue()
    # qmanager = Manager()
    # qmanagerflows = qmanager.dict()
    # progress_tasks = qmanager.dict()
    # tasks_time = qmanager.dict()
    qmanagerflows = {}
    progress_tasks = {}
    tasks_time = {}
    processingdeque = deque()
    ncores = 2
    running_tasks = []
    stamp_to_group = {}
    app = SampleApp()
    app.mainloop()
