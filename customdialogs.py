import tkinter as tk
from tkinter import font  as tkfont # python 3
import tkDialog
from tkinter import messagebox
from tkinter import colorchooser
from tkinter import filedialog
from tkinter import ttk
#import ttk2
from PIL import ImageTk
from PIL import Image
import pandas as pd
import os, cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from customframe import ttkScrollFrame
from sys import platform as _platform
import scipy.ndimage as ndimage
from scipy.linalg import norm
from compareimage import CompareImage
import xlsxwriter
# from skimage.filters import threshold_otsu
# from jenksdetectpack import getJenksBreaks
# from scipy import stats
#from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

img_opencv = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")

def load_img(imgpath, evwidth):
    img = Image.open(imgpath)
    original_width, original_height = img.size
    if evwidth > original_width:
        evwidth = original_width
    smallest_fac = evwidth / original_width
    new_height = int(original_height * smallest_fac)
    new_width = int(original_width * smallest_fac)
    img = img.resize(
        (new_width, new_height), Image.ANTIALIAS
    )
    return ImageTk.PhotoImage(img)

class AboutDialog(tkDialog.Dialog):
    def body(self, master):
        # print("class AboutDialog def body creation")
        ttk.Label(master, text='CONTRACTIONWAVE: open software to process, analyze and visualize cellular contractility\n', wraplength=800, anchor="w", justify=tk.LEFT ).grid(row=0, column=1)
        ttk.Label(master, text='Version: 1.27c\n', wraplength=800, anchor="w", justify=tk.LEFT).grid(row=1, column=1)
        # ttk.Label(master, text='Sergio Scalzo; Marcelo Afonso; Neli Fonseca; Itamar Couto Guedes de Jesus; Ana Paula Alves; Carolina A. T. F. Mendonça, Vanessa Pereira Teixeira; Anderson Kenedy Santos; Diogo Biagi; Estela Crunivel; Maria Jose Campagnole-Santos.; Flavio Marques; Oscar Mesquita; Christopher Kushmerick; Lucas Bleicher; Ubirajara Agero; Silvia Guatimosim\n', wraplength=800, anchor="w", justify=tk.LEFT).grid(row=2, column=1)
        names = "Sérgio Scalzo; Marcelo Q. L. Afonso; Néli J. da Fonseca Jr; Itamar Guedes de Jesus; Ana Paula Alves; Carolina A. T. F. Mendonça, Vanessa Pereira Teixeira; Diogo Biagi; Estela Cruvinel; Anderson Kenedy Santos; Kiany Miranda; Flavio A.M. Marques; Oscar Mesquita; Christopher Kushmerick; Maria José Campagnole-Santos; Ubirajara Agero; Silvia Guatimosim"
        ttk.Label(master, text=names+'\n', wraplength=800, anchor="w", justify=tk.LEFT).grid(row=2, column=1)
        # ttk.Label(master, text='Scalzo, S.; Lima Afonso, M.Q.L.; Fonseca, N.J.;Jesus, I. C. G.; Alves, A. P.;Teixeira,V.P.;\nMarques, F.A.M.;Mesquita, O. N.; Kushmerick, C; Bleicher, L.;Agero, U.; Guatimosim, S.\n').grid(row=2, column=1)
        ttk.Label(master, text='To be published\n', wraplength=800, anchor="w", justify=tk.LEFT ).grid(row=3, column=1)
        ttk.Label(master, text='This software is registered under GPL3.0.\nContact: contractionwave at gmail.com\n', wraplength=800, anchor="w", justify=tk.LEFT ).grid(row=4, column=1)
        ttk.Label(master, wraplength=800, anchor="w", justify=tk.LEFT, text='Third party copyrights are property of their respective owners.\n\nCopyright (C) 2020, Ionicons, Released under MIT License.\nCopyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, All rights reserved.\nCopyright (c) 2012-2013 Matplotlib Development Team; All Rights Reserved\nCopyright (C) 2000-2019, Intel Corporation, all rights reserved.\nCopyright (C) 2009-2011, Willow Garage Inc., all rights reserved.\nCopyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.\nCopyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.\nCopyright (C) 2015-2016, OpenCV Foundation, all rights reserved.\nCopyright (C) 2015-2016, Itseez Inc., all rights reserved.\nCopyright (C) 2019-2020, Xperience AI, all rights reserved.\nCopyright (C) 2018 Uwe Klimmek\nCopyright Regents of the University of California, Sun Microsystems, Inc., Scriptics Corporation, and other parties\nCopyright © 1997-2011 by Secret Labs AB\nCopyright © 1995-2011 by Fredrik Lundh\nCopyright © 2010-2020 by Alex Clark and contributors\nCopyright (c) 2013-2020, John McNamara <jmcnamara@cpan.org> All rights reserved.\nCopyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola, All rights reserved. \nCopyright (c) 2005, NumPy Developers\nCopyright (c) 2010-2020, PyInstaller Development Team\nCopyright (c) 2005-2009, Giovanni Bajo\n Based on previous work under copyright (c) 2002 McMillan Enterprises, Inc.\nCopyright © 2001, 2002 Enthought, Inc. All rights reserved. \nCopyright © 2003-2019 SciPy Developers. All rights reserved.\n ').grid(row=5, column=1)

        self.packbtns = False

    def validate(self):
        # print("class AboutDialog def validate start")
        self.valid = False
        return 0

    def apply(self):
        # print("class AboutDialog def apply start")
        pass

class FolderSelectDialog(tkDialog.Dialog):
    def body(self, master):
        # print("class FolderSelectDialog def body creation")
        # tkDialog.Dialog.okbtn.pack_forget()
        # tkDialog.Dialog.cnbtn.pack_forget()
        # ttk.Label(master, text='Select Input Type:').grid(row=0, column=1)

        #PageOne Load Data Dialog
        self.loadimgs = tk.PhotoImage(file="icons/images-sharp.png")
        self.loadvids = tk.PhotoImage(file="icons/film-sharp.png")
        self.loadtiff = tk.PhotoImage(file="icons/duplicate-sharp.png")

        btn1cont = ttk.Frame(master)
        ttk.Label(btn1cont, text="Image Folder").grid(row=0, column=1)
        tbtn1 = ttk.Button(btn1cont, image=self.loadimgs, command=lambda: self.setresult("Folder"))
        tbtn1.image = self.loadimgs
        tbtn1.grid(row=0, column=0)
        btn1cont.grid(row=0, column=0)
        
        btn2cont = ttk.Frame(master)
        ttk.Label(btn2cont, text="Video File").grid(row=0, column=1)
        tbtn2 = ttk.Button(btn2cont, image=self.loadvids, command=lambda: self.setresult("Video"))
        tbtn2.image=self.loadvids
        tbtn2.grid(row=0, column=0)
        btn2cont.grid(row=0, column=1)

        btn3cont = ttk.Frame(master)
        ttk.Label(btn3cont, text="Tiff Directory").grid(row=0, column=1)
        tbtn3 = ttk.Button(btn3cont, image=self.loadtiff, command=lambda: self.setresult("Tiff Directory"))
        tbtn3.image=self.loadtiff
        tbtn3.grid(row=0, column=0)
        btn3cont.grid(row=0, column=2)

        self.packbtns = False
        self.result = None

    def setresult(self, txt):
        self.result = txt
        self.ok()

    def validate(self):
        # print("class FolderSelectDialog def validate start")
        if self.result != None:
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        # print("class FolderSelectDialog def apply start")
        pass

class CoreAskDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class CoreAskDialog def body creation")
        ttk.Label(master, text='Start Processing Queue:').grid(row=0, column=0)

        self.formatvar = tk.StringVar(master)
        self.formatchoices = range(1, self.literals["maxcores"]+1)
        self.formatvar.set(self.literals["maxcores"])
        self.optmenu = ttk.OptionMenu(master, self.formatvar, self.literals["maxcores"], *self.formatchoices)
        ttk.Label(master, text='Core Number:').grid(row=1, column=0)
        self.optmenu.grid(row = 1, column = 1)

    def validate(self):
        # print("class CoreAskDialog def validate start")
        #nothing to validate
        if int(self.formatvar.get()) > 0:
            self.valid = True
            return 1
        else:
            self.valid = False
            return 0

    def apply(self):
        # print("class CoreAskDialog def apply start")
        #save configs
        self.result = int(self.formatvar.get())

class SelectMenuItem(tkDialog.Dialog):

    def body(self, master):
        # print("class SelectMenuItemN def body creation")
        ttk.Label(master, text='Type New Name:').grid(row=0, column=1)
        self.AnswerVar = tk.StringVar()
        self.AnswerVar.set(self.literals["current_name"])
        ttk.Entry(master, width=10, textvariable=self.AnswerVar).grid(row=1,column=1)
        self.AnswerVar.set(self.literals["current_name"])

    def validate(self):
        # print("class SelectMenuItem def validate start")
        #nothing to validate
        if len(str(self.AnswerVar.get())) > 0:
            self.valid = True
            return 1
        else:
            self.valid = False
            return 0

    def apply(self):
        # print("class SelectMenuItemN def apply start")
        #save configs
        self.result = str(self.AnswerVar.get())

class AddPresetDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class AddPresetDialog def body creation")
        ttk.Label(master, text='New Preset Name:').grid(row=0, column=1)
        self.AnswerVar = tk.StringVar()
        self.AnswerVar.set("")
        ttk.Entry(master, width=10, textvariable=self.AnswerVar).grid(row=1,column=1)
        self.result = None

    def validate(self):
        # print("class AddPresetDialog def validate start")
        #nothing to validate
        if self.AnswerVar.get() != "":
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        # print("class AddPresetDialog def apply start")
        #save configs
        if self.valid == True:
            self.result = self.AnswerVar.get()
        else:
            self.result = None

class DotChangeDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class DotChangeDialog def body creation")
        ttk.Label(master, text='Dot Type:').grid(row=0, column=1)
        self.DType = tk.StringVar()
        dot_types = ["First", "Max", "Min", "Last"]
        fkey = sorted(dot_types)[0]
        self.DType.set(fkey)
        self.optmenu = ttk.OptionMenu(master, self.DType,fkey, *sorted(dot_types), command=self.set_dtype)
        self.optmenu.grid(row=1, column=1)
        self.DType.set(fkey)

        self.result = None

    def set_dtype(self, args=None):
        self.DType.set(args)

    def validate(self):
        # print("class DotChangeDialog def validate start")
        #nothing to validate
        if self.DType.get() != "":
            self.master.dragDots.tempresult = True
            self.valid = True
            return 1
        self.valid = False
        self.master.dragDots.tempresult = False
        self.master.popupCFocusOut()
        return 0

    def apply(self):
        # print("class DotChangeDialog def apply start")
        #save configs
        if self.valid == True:
            self.master.dragDots.tempresult = True
            self.result = self.DType.get().lower()
        else:
            self.master.dragDots.tempresult = False
            self.result = None

class SelectPresetDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class SelectPresetDialog def body creation")
        ttk.Label(master, text='Select from Presets:').grid(row=0, column=1)
        self.PresetSelect = tk.StringVar()
        fkey = sorted(self.literals["preset_dicts"].keys())[0]
        self.PresetSelect.set(fkey)
        self.optmenu = ttk.OptionMenu(master, self.PresetSelect,fkey, *sorted(self.literals["preset_dicts"].keys()), command=self.set_preset)
        self.optmenu.grid(row=1, column=1)
        self.PresetSelect.set(fkey)
        rown = 1
        rown += 1
        ttk.Label(master, text="pyr_scale".replace("_", " ")).grid(row=rown, column=0)
        self.pyr_scale_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["pyr_scale"]))
        self.pyr_scale_lbl.grid(row=rown, column=1)
        rown += 1
        ttk.Label(master, text="levels".replace("_", " ")).grid(row=rown, column=0)
        self.levels_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["levels"]))
        self.levels_lbl.grid(row=rown, column=1)
        rown += 1
        ttk.Label(master, text="winsize".replace("_", " ")).grid(row=rown, column=0)
        self.winsize_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["winsize"]))
        self.winsize_lbl.grid(row=rown, column=1)
        rown += 1
        ttk.Label(master, text="iterations".replace("_", " ")).grid(row=rown, column=0)
        self.iterations_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["iterations"]))
        self.iterations_lbl.grid(row=rown, column=1)
        rown += 1
        ttk.Label(master, text="poly_n".replace("_", " ")).grid(row=rown, column=0)
        self.poly_n_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["poly_n"]))
        self.poly_n_lbl.grid(row=rown, column=1)
        rown += 1
        ttk.Label(master, text="poly_sigma".replace("_", " ")).grid(row=rown, column=0)
        self.poly_sigma_lbl = ttk.Label(master, text=str(self.literals["preset_dicts"][fkey]["poly_sigma"]))
        self.poly_sigma_lbl.grid(row=rown, column=1)
        self.result = None

    def set_preset(self, args=None):
        self.PresetSelect.set(args)
        self.pyr_scale_lbl['text'] =  str(self.literals["preset_dicts"][self.PresetSelect.get()]["pyr_scale"])
        self.levels_lbl['text'] =  str(self.literals["preset_dicts"][self.PresetSelect.get()]["levels"])
        self.winsize_lbl['text'] =  str(self.literals["preset_dicts"][self.PresetSelect.get()]["winsize"])
        self.iterations_lbl['text'] = str(self.literals["preset_dicts"][self.PresetSelect.get()]["iterations"])
        self.poly_n_lbl['text'] =  str(self.literals["preset_dicts"][self.PresetSelect.get()]["poly_n"])
        self.poly_sigma_lbl['text'] = str(self.literals["preset_dicts"][self.PresetSelect.get()]["poly_sigma"])

    def validate(self):
        # print("class SelectPresetDialog def validate start")
        #nothing to validate
        if self.PresetSelect.get() != "":
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        # print("class SelectPresetDialog def apply start")
        #save configs
        if self.valid == True:
            self.result = self.literals["preset_dicts"][self.PresetSelect.get()]
        else:
            self.result = None

class CustomYesNo(tkDialog.Dialog):
    def body(self, master):
        ttk.Label(master, text=self.thistitle, font=('Helvetica', 12)).grid(row=0, column=0)
        self.result = False
        self.btn1_name = "Yes"
        self.btn2_name = "No"
        pass

    def validate(self):
        self.valid = True
        return 1
    
    def apply(self):
        self.result = True

class PlotSettingsProgress(tkDialog.Dialog):

    def body(self, master):
        # print("class PlotSettingsProgress def body creation")
        self.current_settings = self.literals["plotsettingscolors"]
        self.current_lineopts = self.literals["plotsettingsline"]
        rowindex = 0

        ttk.Label(master, text='Graph Settings:', font=('Helvetica', 14)).grid(row=rowindex, column=0, columnspan=2)
        rowindex += 1

        checklvarstatus = 0
        if self.current_lineopts["zero"] == True:
            checklvarstatus = 1
        self.thisCheck_line_var = tk.IntVar()
        self.thisCheck_line_var.set(checklvarstatus)
        self.thisCheck_line = ttk.Checkbutton(master, text="Plot X Axis Baseline", variable = self.thisCheck_line_var,width=20)
        self.thisCheck_line.texttype = "zero"
        self.thisCheck_line.grid(row=rowindex, column=0)

        self.zeroButton = ttk.Button(master, text="Baseline Color",width=20)
        self.zeroButton.texttype = "zero_color"
        self.zeroButton.bind("<Button-1>", self.change_color2)
        self.zeroButton.grid(row=rowindex, column=1)
        rowindex += 1

        checkgvarstatus = 0
        if self.current_lineopts["grid"] == True:
            checkgvarstatus = 1
        self.thisCheck_grid_var = tk.IntVar()
        self.thisCheck_grid_var.set(checkgvarstatus)
        self.thisCheck_grid = ttk.Checkbutton(master, text="Plot Gridlines", variable = self.thisCheck_grid_var,width=20)
        self.thisCheck_grid.texttype = "grid"
        self.thisCheck_grid.grid(row=rowindex, column=0)

        self.gridButton = ttk.Button(master, text="Gridlines Color",width=20)
        self.gridButton.texttype = "grid_color"
        self.gridButton.bind("<Button-1>", self.change_color2)
        self.gridButton.grid(row=rowindex, column=1)
        rowindex += 1

        self.TimeSelect = tk.StringVar()
        self.full_time_dict = {
            "Seconds": "s",
            "Milliseconds": "ms"
        }
        self.rev_full_time_dict = {
            "s": "Seconds",
            "ms": "Milliseconds"
        }
        fkey = self.rev_full_time_dict[self.literals["current_time"]]
        self.TimeSelect.set(fkey)
        self.optmenu = ttk.OptionMenu(master, self.TimeSelect, "Seconds", *sorted(self.full_time_dict.keys()), command=self.set_timetype)
        self.optmenu.texttype = "time_unit"
        ttk.Label(master, text="Used Time Unit:").grid(row=rowindex, column=0, columnspan=2)
        rowindex+=1
        self.optmenu.grid(row=rowindex, column=0, columnspan=2)
        self.TimeSelect.set(fkey)
        rowindex+=1

        checkavarstatus = 0
        if self.current_lineopts["absolute_time"] == True:
            checkavarstatus = 1
        self.thisCheck_absolute_var = tk.IntVar()
        self.thisCheck_absolute_var.set(checkavarstatus)
        self.thisCheck_absolute = ttk.Checkbutton(master, text="Wave absolute time", variable = self.thisCheck_absolute_var,width=20)
        self.thisCheck_absolute.texttype = "absolute_time"
        self.thisCheck_absolute.grid(row=rowindex, column=0, columnspan=2)
        rowindex+=1

        checkavarstatus2 = 0
        if self.current_lineopts["show_dots"] == True:
            checkavarstatus2 = 1
        self.thisCheck_dots_var = tk.IntVar()
        self.thisCheck_dots_var.set(checkavarstatus2)
        self.thisCheck_dots = ttk.Checkbutton(master, text="Plot dots of interest", variable = self.thisCheck_dots_var,width=20)
        self.thisCheck_dots.texttype = "show_dots"
        self.thisCheck_dots.grid(row=rowindex, column=0, columnspan=2)
        rowindex+=1

        sep = ttk.Separator(master, orient='horizontal')
        sep.grid(row=rowindex, column=0, columnspan=2, sticky="ew", ipadx=20, ipady=1)
        rowindex += 1

        ttk.Label(master, text='Color Settings:', font=('Helvetica', 14)).grid(row=rowindex, column=0, columnspan=2)
        rowindex += 1

        keytextconversion = {
            "main": 'Set Plot Line Color',
            "first":'Set First Dots Color',
            "max":'Set Max Dots Color',
            "min":'Set Min Dots Color',
            "last":'Set Last Dots Color',
            "fft": 'Set FFT Dots Color',
            "fft_selection": 'Set FFT Sel. Dot Color',
            "noise_true": 'Set Noise Dots Color',
            "noise_false": 'Set Wave Dots Color',
            "rect_color": 'Set Sel. Areas Color',
            "gvf": 'Set Max. Filtering Line Color'
        }
        for j, k in enumerate(sorted(self.current_settings.keys())):
            coln = 1
            if j % 2 == 0:
                rowindex += 1
                coln = 0
            thisButton = ttk.Button(master, text=keytextconversion[k],width=20)
            thisButton.texttype = k
            thisButton.bind("<Button-1>", self.change_color)
            thisButton.grid(row=rowindex, column=coln)
        rowindex += 1

        # print("class PlotSettingsProgress def body created")

    def change_color(self, event):
        # print("class PlotSettingsProgress def change_color started")
        k = event.widget.texttype
        newcolor = colorchooser.askcolor(title="Select Color for " + k, color=self.current_settings[k])
        # print("class PlotSettingsProgress def change_color current color is: " + self.current_settings[k])
        if newcolor[1] != None:
            # print("class PlotSettingsProgress def change_color color changed!")
            self.current_settings[k] = newcolor[1]
            # print("class PlotSettingsProgress def change_color new color is:" + newcolor[1])
        # print("class PlotSettingsProgress def change_color done")
    
    def change_color2(self, event):
        # print("class PlotSettingsProgress def change_color2 started")
        k = event.widget.texttype
        newcolor = colorchooser.askcolor(title="Select Color for " + k, color=self.current_lineopts[k])
        # print("class PlotSettingsProgress def change_color2 current color is: " + self.current_lineopts[k])
        if newcolor[1] != None:
            # print("class PlotSettingsProgress def change_color2 color changed!")
            self.current_lineopts[k] = newcolor[1]
            # print("class PlotSettingsProgress def change_color2 new color is:" + newcolor[1])
        # print("class PlotSettingsProgress def change_color2 done")
    
    def set_timetype(self, event):
        # print("class PlotSettingsProgress def set_timetype started")
        self.current_lineopts["time_unit"] = self.full_time_dict[self.TimeSelect.get()]
        # print("class PlotSettingsProgress def set_timetype done")
    
    def validate(self):
        # print("class PlotSettingsProgress def validate start")
        #nothing to validate
        self.valid = True
        if self.thisCheck_line_var.get() == 1:
            self.current_lineopts["zero"] = True
        else:
            self.current_lineopts["zero"] = False
        if self.thisCheck_grid_var.get() == 1:
            self.current_lineopts["grid"] = True
        else:
            self.current_lineopts["grid"] = False
        if self.thisCheck_absolute_var.get() == 1:
            self.current_lineopts["absolute_time"] = True
        else:
            self.current_lineopts["absolute_time"] = False
        if self.thisCheck_dots_var.get() == 1:
            self.current_lineopts["show_dots"] = True
        else:
            self.current_lineopts["show_dots"] = False
        # print("class PlotSettingsProgress def validate done")
        return 1

    def apply(self):
        # print("class PlotSettingsProgress def apply start")
        #save configs
        self.result = {}
        # print(self.current_settings.copy())
        # print(self.current_lineopts.copy())
        self.result["peak_plot_colors"] = self.current_settings.copy()
        self.result["plotline_opts"] = self.current_lineopts.copy()
        # print("class PlotSettingsProgress def apply done")
        # return self.current_settings[k]

class ProgressBarDialog(tkDialog.Dialog):
    def body(self, master):
        #literals:
        #obj
        #aux
        aux = self.literals["aux"]
        #create main from literals
        #create main label
        frameuno = ttk.Frame(self)
        frameuno.grid_rowconfigure(0, weight=1)
        for ic in range(5):
            frameuno.columnconfigure(ic, weight=1)

        lbl1 = ttk.Label(frameuno, text="Running... ", font=self.master.controller.title_font)
        lbl1.grid(row=0, column=0, rowspan=1, columnspan=3)
        
        self.var_bar = tk.DoubleVar()
        self.var_bar.set(0.0)
        #create main bar
        self.pbar = ttk.Progressbar(frameuno, style="", length=500, variable=self.var_bar, maximum=1)
        self.pbar.grid(row=1, column=0, rowspan=1, columnspan=3)

        frameuno.grid(row=0, column=0, rowspan=2, columnspan=5, sticky=tk.NSEW)
        # frameuno.pack()

        #start running background process
        self.packbtns = False
        self.master.controller.progress_bar = self
        self.protocol("WM_DELETE_WINDOW", self.customCancel)
        self.bind("<Escape>", self.customCancel)
        # try:
            # self.protocol("WM_DELETE_WINDOW", self.customCancel)
            # self.bind("<Escape>", self.customCancel)
        # except Exception as e:
            # print("Exception!")
            # print(e)
        time.sleep(1)
        self.master.after(100, lambda: self.master.frames["PageFour"].queuePreProcess(aux) )
        # self.master.frames["PageFour"].queuePreProcess(aux)

    def refreshProgress(self):
        #check for any progress and update bar
        progress, etask = self.master.frames["PageFour"].retrievePreProcess()
        print("refreshProgress progress")
        print("current stamp: ")
        print(etask)
        print(progress)
        self.var_bar.set(progress)
        if progress == 1.0:
            #trigger close bar
            # if etask not in self.master.frames["PageFour"].done_pdiffs:
            #     self.master.frames["PageFour"].done_pdiffs.append(etask)
            self.ok()

    def validate(self):
        #segmentation checking for disturbances
        self.valid = True
        return 1

    def customCancel(self):
        progress, etask = self.master.frames["PageFour"].retrievePreProcess()
        print("apply progress")
        print(progress)
        if progress < 1.0:
            self.master.frames["PageFour"].killTasks(etask)
        self.master.controller.progress_bar = None
        self.cancel()

    def apply(self):
        progress, etask = self.master.frames["PageFour"].retrievePreProcess()
        print("apply progress")
        print(progress)
        if progress < 1.0:
            self.master.frames["PageFour"].killTasks(etask)
        else:
            if etask not in self.master.frames["PageFour"].done_pdiffs:
                self.master.frames["PageFour"].done_pdiffs.append(etask)
        self.master.controller.progress_bar = None
        self.result = True

class ReferenceDefinitionDialog(tkDialog.Dialog):
    def body(self, master):
        frameuno = ttk.Frame(self)
        for ir in range(2):
            frameuno.grid_rowconfigure(ir, weight=1)
        for ic in range(3):
            frameuno.columnconfigure(ic, weight=1)
        ttk.Label(frameuno,text="Image subtraction reference frames:",  font=('Helvetica', 14)).grid(row=0, column=0, columnspan=3)
        # ttk.Label(frameuno,text="(frame shift)",  font=('Helvetica', 10)).grid(row=1, column=1, columnspan=3)
        self.current_spin = tk.Spinbox(frameuno, from_=(-1 * self.literals["framenumber"]/2), to=self.literals["framenumber"]/2, increment=1, width=4)
        self.current_spin.thistype = "frame_shift" 
        self.current_spin.grid(row=1, column=2)
        self.current_spin.delete(0,"end")
        self.current_spin.insert(0,-2)
        ttk.Label(frameuno,text="Contraction Start + Shift:",  font=('Helvetica', 12)).grid(row=1, column=0, columnspan=1)
        self.current_spin.grid(row=1, column=1, columnspan=1)
        frameuno.grid(row=0, column=0, rowspan=2, columnspan=5, sticky=tk.NSEW)
        # self.current_spin.bind('<Return>', lambda *args: self.set_segmentation(0))
        self.packbtns = False
        self.packbtnokonly = True
        pass

    def validate(self):
        try:
            current_ref_spin_val = int(self.current_spin.get().replace(",", "."))
            if current_ref_spin_val < (-1 * self.literals["framenumber"]/2) or current_ref_spin_val > self.literals["framenumber"]/2:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0 
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0
        self.valid = True
        return 1
    
    def apply(self):
        self.result = int(self.current_spin.get().replace(",", "."))


class DiffComparisionDialog(tkDialog.DialogMax):
    def body(self, master):
        #literals:
        #data
        data = self.literals["data"]
        data2 = self.literals["data2"]
        #data2
        self.mframe = self.literals["mframe"]

        #create plot with zoom
        self.plotFrame = ttk.Frame(self)
        self.subPlotFrame = ttk.Frame(self.plotFrame)

        self.frameLabel = ttk.Label(self.plotFrame, text="Comparison View:", font=self.master.controller.subtitle_font)

        self.figFrame = plt.figure(figsize=(6, 3), dpi=100, facecolor=self.master.controller.bgcolor)
        self.figFrame.tight_layout()
        self.figFrame.subplots_adjust(top=0.85, bottom=0.25)
        self.gsWave = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2)
        self.canvasFrame = FigureCanvasTkAgg(self.figFrame, master=self.subPlotFrame)  # A tk.DrawingArea.
        self.canvasFrame.draw()
        # pack_toolbar=False will make it easier to use a layout manager later on.
        self.toolbar = NavigationToolbar2Tk(self.canvasFrame, self.subPlotFrame)
        self.toolbar.update()
        
        self.mainplotartistFrame = None
        self.mainplotartistFrame2 = None

        self.axFrame = self.figFrame.add_subplot()
        self.axFrame2 = self.axFrame.twinx()

        self.axbaselineFrame = None
        self.axgridFrame = None
        
        self.axFrame.set_xlabel("Time ("+self.master.controller.current_timescale+")")
        self.axFrame.set_ylabel("Average Speed ("+self.master.controller.current_speedscale+")")# ▢")
        self.axFrame2.set_ylabel("Contraction Amplitude (a.u)")# ▢")
        # self.axFrame2.set_ylabel("Average Pixel Intensity ("+self.literals["controller"].controller.current_pixscale+")")
        
        self.canvasFrame.draw()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvasFrame.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        # ttk.Label(self.adjustfiltersframe,text="Blur size:",  font=('Helvetica', 10)).grid(row=1, column=1)
        # self.blur_size_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["blur_size"][0], to=self.literals["config"]["blur_size"][1], increment=2, width=4, command=lambda: self.set_segmentation(0))
        self.spinnerLowLimLabel = ttk.Label(self.plotFrame, text="Low Limit (Contraction Amplitude):")
        self.spinnerLowLim = tk.Spinbox(self.plotFrame, increment=1, from_=np.min(data)/10, to=np.max(data)*10, width=6, command=self.updatePlot)
        self.spinnerHighLimLabel = ttk.Label(self.plotFrame, text="High Limit (Contraction Amplitude):")
        self.spinnerHighLim = tk.Spinbox(self.plotFrame, increment=1, from_=np.min(data)/10, to=np.max(data)*10, width=6, command=self.updatePlot)
        self.spinnerLowLim.bind('<Return>',lambda *args: self.updatePlot())
        self.spinnerHighLim.bind('<Return>',lambda *args: self.updatePlot())

        self.button_plot = ttk.Button(self.plotFrame, text="Export Figure", command=self.exportPlotFig, width=15)
        self.button_data = ttk.Button(self.plotFrame, text="Export Data", command=self.exportPlotData, width=15)
        for ic in range(6):
            self.plotFrame.columnconfigure(ic, weight=1)
        for ir in range(8):
            self.plotFrame.rowconfigure(ir, weight=1)
        
        self.frameLabel.grid(row=0, column=2, columnspan=1, sticky=tk.NSEW)
        self.subPlotFrame.grid(row=1, column=0, rowspan=5, columnspan=5, sticky=tk.NSEW)
        # ttk.Label(self.adjustfiltersframe,text="Scale adjustment:",  font=('Helvetica', 10)).grid(row=1, column=1)
        self.spinnerLowLimLabel.grid(row=6, column=1)
        self.spinnerLowLim.grid(row=6, column=2)
        self.spinnerHighLimLabel.grid(row=6, column=3)
        self.spinnerHighLim.grid(row=6, column=4)
        # self.setLimsFrame.grid(row=6, column=1, columnspan=2)
        self.button_plot.grid(row=7, column=1, columnspan=1)
        self.button_data.grid(row=7, column=3, columnspan=1)

        self.figFrame.canvas.draw()

        xdata = None
        if self.master.controller.current_timescale == "s":
            xdata = [i / self.master.FPS  for i in range(len(data))]
        elif self.master.controller.current_timescale == "ms":
            xdata = [(i / self.master.FPS ) * 1000.0 for i in range(len(data))]
        # self.mainplotartistFrame = self.axFrame.plot(data, color=self.master.controller.plotsettings.peak_plot_colors["main"])
        self.mainplotartistFrame = self.axFrame.plot(xdata, data, color=self.master.controller.plotsettings.peak_plot_colors["main"],label="Average Speed ("+self.master.controller.current_speedscale+")")
        # self.mainplotartistFrame2 = self.axFrame2.plot(data2, color=self.master.controller.plotsettings.peak_plot_colors["max"])
        self.mainplotartistFrame2 = self.axFrame2.plot(xdata, data2, color=self.master.controller.plotsettings.peak_plot_colors["max"],label="Contraction Amplitude (a.u)")
        #get smallest axFrame2 data
        data2arr = np.array(data2)
        mindata2_above_zero = np.min(data2arr[data2arr > 0])
        maxdata2 = np.max(data2arr)
        lim_yax2 = self.axFrame2.get_ylim()
        # maxdata2_val = np.round(maxdata2 * 1.1)
        maxdata2_val = maxdata2 * 1.1
        # self.axFrame2.set_ylim(mindata2_above_zero, lim_yax2[1])
        print("mindata2_above_zero")
        print(mindata2_above_zero)
        print('maxdata2_val')
        print(maxdata2_val)
        self.axFrame2.set_ylim(mindata2_above_zero, maxdata2_val)
        self.spinnerLowLim.delete(0,"end")
        self.spinnerLowLim.insert(0,mindata2_above_zero)
        self.spinnerHighLim.delete(0,"end")
        self.spinnerHighLim.insert(0,maxdata2_val)


        # self.axFrame.legend()
        # self.axFrame2.legend()
        h1, l1 = self.axFrame.get_legend_handles_labels()
        h2, l2 = self.axFrame2.get_legend_handles_labels()
        # self.axFrame.legend(h1+h2, l1+l2, loc=2)
        self.axFrame.legend(h1+h2, l1+l2, loc='upper left', fontsize='xx-small')
        self.figFrame.canvas.draw()

        # self.fxlabels = None
        # if self.master.controller.current_timescale == "s":
        #     self.fxlabels = [ float("{:.3f}".format(float(item.get_text().replace("−", "-")) / self.master.FPS)) for item in self.axFrame.get_xticklabels() ]
        # elif self.master.controller.current_timescale == "ms":
        #     self.fxlabels = [ float("{:.3f}".format((float(item.get_text().replace("−", "-")) / self.master.FPS)*1000)) for item in self.axFrame.get_xticklabels() ]
        # self.axFrame.set_xticklabels(self.fxlabels)
        
        curlims = (self.axFrame.get_xlim(), self.axFrame.get_ylim())

        if self.axbaselineFrame != None:
            self.axbaselineFrame.remove()
            self.axbaselineFrame = None
        if  self.master.controller.plotsettings.plotline_opts["zero"] == True:
            self.axbaselineFrame = self.axFrame.axhline(y=0.0, color= self.master.controller.plotsettings.plotline_opts["zero_color"], linestyle='-')

        if self.axgridFrame != None:
            self.axgridFrame.remove()
            self.axgridFrame = None
        if  self.master.controller.plotsettings.plotline_opts["grid"] == True:
            self.axgridFrame = self.axFrame.grid(linestyle="-", color= self.master.controller.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.axFrame.grid(False)
        
        self.axFrame.set_xlim(curlims[0])
        self.axFrame.set_ylim(curlims[1])

        self.figFrame.canvas.draw()
        
        # self.plotFrame.pack(expand=True)
        self.plotFrame.grid(row=0, column=0, rowspan=1, columnspan=3, sticky=tk.NSEW)
        
        #create export button
        pass

    def updatePlot(self):
        highlim = float(self.spinnerHighLim.get().replace(",", "."))
        lowlim = float(self.spinnerLowLim.get().replace(",", "."))
        self.axFrame2.set_ylim(lowlim, highlim)
        self.figFrame.canvas.draw()

    def exportPlotData(self):
        try:
            xlbldata = []
            if self.master.controller.current_timescale == "s":
                xlbldata= [float("{:.3f}".format(float(a) / self.master.FPS)) for a in self.mainplotartistFrame[0].get_xdata()]
            elif self.master.controller.current_timescale == "ms":
                xlbldata= [float("{:.3f}".format(float(a) * 1000.0 / self.master.FPS)) for a in self.mainplotartistFrame[0].get_xdata()]
            SaveTableDialog(self, title='Save Length Data', literals=[
                ("headers", [self.axFrame.get_xlabel(), self.axFrame2.get_ylabel()]),
                ("data", [xlbldata, self.mainplotartistFrame2[0].get_ydata()]),
                ("data_t", "single")
                ], parent2=self.mframe)
        except TypeError:
            messagebox.showerror("Error", "No Data in plot")

    def exportPlotFig(self):
        #export current plot figure
        formats = set(self.figFrame.canvas.get_supported_filetypes().keys())
        d = SaveFigureDialog(self, title='Save Figure', literals=[
                ("formats", formats),
                ("bbox", 1)
            ], parent2=self.mframe)
        if d.result != None:
            if d.result["format"] == ".jpg" or d.result["format"] == ".jpeg":
                self.figFrame.savefig(r'%s' %d.result["name"],quality=d.result["quality"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
                messagebox.showinfo(
                    "File saved",
                    "File was successfully saved"
                )
            else:
                self.figFrame.savefig(r'%s' %d.result["name"], dpi=d.result["dpi"], bbox_inches=d.result["bbox"])
                messagebox.showinfo(
                    "File saved",
                    "File was successfully saved"
                )

    def validate(self):
        self.valid = True
        return 1

    def apply(self):
        self.result = True

class NewCellLengthDialog(tkDialog.DialogMax):
    def body(self, master):

        #todo: make update on Analyse click btn
        #update peak num when clicking on new peak
        #area_label_calc addition

        #
        #Information Parsing
        #

        self.peaks = self.literals["peaks_obj"]
        self.current_peak_index = self.literals["peak_index"]
        self.index_of_peak = 0
        self.current_peak = self.peaks[self.current_peak_index]
        self.mframe = self.literals["mframe"]
        self.segmentation_default = self.literals["segdef"]
        self.segmentation_dict = {}
        for i, peak in enumerate(self.peaks):
            self.segmentation_dict[i] = []
            for j in range(len(peak.peaktimes)):
                #i (peakindex), j(imageindex), k(segmentation setting index)
                self.segmentation_dict[i].append(self.segmentation_default.copy())
        self.set_image_list()
        
        #
        #Master Frame Declaration
        #
        
        self.masterframe = ttkScrollFrame(master, 'greyBackground.TFrame', self.literals["controller"].bgcolor) #add scroll bar
        #6 columns
        self.masterframe.viewPort.columnconfigure(0, weight=1)
        self.masterframe.viewPort.columnconfigure(1, weight=1)
        self.masterframe.viewPort.columnconfigure(2, weight=1)
        self.masterframe.viewPort.columnconfigure(3, weight=1)
        self.masterframe.viewPort.columnconfigure(4, weight=1)
        self.masterframe.viewPort.columnconfigure(5, weight=1)

        #
        # First Big Title creation
        #

        rown = 0 #first row number
        #Add title frame
        self.firstBigTitleFrame = ttk.Frame(self.masterframe.viewPort)
        # self.firstBigTitleLabel = ttk.Label(self.firstBigTitleFrame, text="", font=self.literals["controller"].subtitle_font)
        self.firstBigTitleLabel = ttk.Label(self.firstBigTitleFrame, text="Contraction-relaxation Wave "+str(self.current_peak_index), font=self.literals["controller"].title_font)
        rspan = 1
        self.firstBigTitleLabel.grid(row=0,column=2,rowspan=1,columnspan=2, sticky=tk.NSEW)
        
        self.firstBigTitleFrame.grid_rowconfigure(0, weight=1)
        for ic in range(5):
            self.firstBigTitleFrame.columnconfigure(ic, weight=1)
        self.firstBigTitleFrame.grid(row=rown, column=0, rowspan=rspan, columnspan=6, sticky=tk.NSEW)
        rown += rspan

        #
        # Table with all Waves creation
        #
        
        # self.table_wave_frame = ttk.Frame(self.masterframe.viewPort)
        self.treePrevSelectedNames = []
        self.tableFrame = ttk.Frame(self.masterframe.viewPort) 
        self.tableTreeLabel = ttk.Label(self.tableFrame, text="Wave Selection:", font=self.literals["controller"].subtitle_font)
        self.tableTreeLabel.grid(row=0, column=0, sticky=tk.NSEW)
        self.tableCols = ("Wave Index", )
        # self.tableTree = ttk.Treeview(self.tableFrame, columns=self.tableCols, selectmode='extended', show='headings')
        self.tableTree = ttk.Treeview(self.tableFrame, columns=self.tableCols, selectmode='browse', show='headings')
        self.tableTree.grid(row=1, column=0, sticky=tk.NSEW)
        self.tableTree.tag_configure('oddrow', background='#d3d3d3')
        self.tableTree.tag_configure('evenrow', background='white')
        for col in self.tableCols:
            self.tableTree.heading(col, text=col)
        self.vsbTable = ttk.Scrollbar(self.tableFrame, orient="vertical", command=self.tableYScroll)
        self.vsbTableHorizontal = ttk.Scrollbar(self.tableFrame, orient="horizontal", command=self.tableXScroll)
        self.vsbTable.grid(row=1,column=1,sticky=tk.NS)
        self.vsbTableHorizontal.grid(row=2,column=0,sticky=tk.EW)
        self.tableTree.configure(yscrollcommand=self.vsbTable.set, xscrollcommand=self.vsbTableHorizontal.set)
        self.tableTree.bind("<<TreeviewSelect>>", self.tabletreeselection)

        for ic in range(2):
            self.tableFrame.columnconfigure(ic, weight=1)
        for ir in range(3):
            self.tableFrame.rowconfigure(ir, weight=1)
        rspan = 6
        self.tableFrame.grid(row=rown, column=0, rowspan=rspan, columnspan=2, sticky=tk.NSEW)

        #
        # Wave Plot Creation
        #
        self.waveFrame = ttk.Frame(self.masterframe.viewPort)
        self.waveFrame2 = ttk.Frame(self.waveFrame)
        # self.waveLabel = ttk.Label(self.waveFrame, text="Wave View:", font=self.literals["controller"].subtitle_font)
        self.rectWave = None
        self.rectContour = None
        self.rectCellLength = None
        self.pressmovecidWave = None
        self.figWave = plt.figure(figsize=(6, 3), dpi=100, facecolor=self.literals["controller"].bgcolor)
        self.figWave.tight_layout()
        self.figWave.subplots_adjust(top=0.85, bottom=0.25)
        self.gsWave = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2)
        self.canvasWave = FigureCanvasTkAgg(self.figWave, master=self.waveFrame2)  # A tk.DrawingArea.
        self.mainplotartistWave = None
        self.axWave = self.figWave.add_subplot()
        self.axbaselineWave = None
        self.axgridWave = None
        self.axWave.set_xlabel("Time ("+self.literals["controller"].controller.current_timescale+")")
        self.axWave.set_ylabel("Average Speed ("+self.literals["controller"].controller.current_speedscale+")")
        self.canvasWave.draw()
        self.canvasWave.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        for ic in range(6):
            self.waveFrame.columnconfigure(ic, weight=1)
        for ir in range(6):
            self.waveFrame.rowconfigure(ir, weight=1)

        # self.waveLabel.grid(row=0, column=1, columnspan=3, sticky=tk.NSEW)
        self.waveFrame2.grid(row=0, column=0, rowspan=5, columnspan=5, sticky=tk.NSEW)
        self.waveFrame.grid(row=rown, column=2, rowspan=rspan, columnspan=4, sticky=tk.NSEW)
        self.figWave.canvas.draw()

        for i, peak in enumerate(self.peaks):
            ptxt = str(i+1)
            # if i == self.current_peak_index:
                # ptxt = "Current"
            cur_tag = "oddrow"
            if i % 2 == 0:
                cur_tag = "evenrow"
            self.tableTree.insert("", "end", values=(ptxt), text=str(i), tags = (cur_tag,))

        for child_item in self.tableTree.get_children():
            curItemDecomp = self.tableTree.item(child_item)
            current_row = int(curItemDecomp["text"])
            if current_row == self.current_peak_index:
                self.tableTree.focus(child_item)
                self.tableTree.selection_set(child_item)

        rown += rspan

        #
        # Second Big Title creation
        #

        #Add title frame
        self.secondBigTitleFrame = ttk.Frame(self.masterframe.viewPort)
        # self.firstBigTitleLabel = ttk.Label(self.firstBigTitleFrame, text="", font=self.literals["controller"].subtitle_font)
        self.secondBigTitleLabel = ttk.Label(self.secondBigTitleFrame, text="Cell segmentation", font=self.literals["controller"].title_font)
        rspan = 1
        self.secondBigTitleLabel.grid(row=0,column=2,rowspan=1,columnspan=2, sticky=tk.NSEW)
        
        self.secondBigTitleFrame.grid_rowconfigure(0, weight=1)
        for ic in range(5):
            self.secondBigTitleFrame.columnconfigure(ic, weight=1)
        self.secondBigTitleFrame.grid(row=rown, column=0, rowspan=rspan, columnspan=6, sticky=tk.NSEW)

        rown += rspan

        #
        # Configuration frame creation
        #
        self.configframe = ttk.Frame(self.masterframe.viewPort)
        
        self.adjustfiltersframe = ttk.Frame(self.configframe)

        # ttk.Label(self.configframe, text='Image Filtering:',  font=('Helvetica', 14)  ).grid(row=0, column=0, rowspan=1, columnspan=2, sticky=tk.NSEW)
        ttk.Label(self.configframe, text='Image Filtering:',  font=('Helvetica', 14)  ).grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        # ttk.Label(self.configframe,text="Blur size:",  font=('Helvetica', 8)).grid(row=1, column=0)
        # self.blur_size_spin = tk.Spinbox(self.configframe, from_=self.literals["config"]["blur_size"][0], to=self.literals["config"]["blur_size"][1], increment=2, width=8, command=lambda: self.set_segmentation(0))
        
        ttk.Label(self.adjustfiltersframe,text="Blur size:",  font=('Helvetica', 10)).grid(row=1, column=1)
        self.blur_size_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["blur_size"][0], to=self.literals["config"]["blur_size"][1], increment=2, width=4, command=lambda: self.set_segmentation(0))
        self.blur_size_spin.thistype = "blur_size" 
        self.blur_size_spin.grid(row=1, column=2)
        self.blur_size_spin.delete(0,"end")
        self.blur_size_spin.insert(0,self.literals["segdef"][0])
        self.blur_size_spin.bind('<Return>', lambda *args: self.set_segmentation(0))

        # ttk.Label(self.configframe,text="Kernel dilation:",  font=('Helvetica', 8)).grid(row=2, column=0)
        # self.kernel_dilation_spin = tk.Spinbox(self.configframe, from_=self.literals["config"]["kernel_dilation"][0], to=self.literals["config"]["kernel_dilation"][1], increment=2, width=8, command=lambda: self.set_segmentation(1))
    
        ttk.Label(self.adjustfiltersframe,text="Kernel dilation:",  font=('Helvetica', 10)).grid(row=2, column=1)
        self.kernel_dilation_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["kernel_dilation"][0], to=self.literals["config"]["kernel_dilation"][1], increment=2, width=4, command=lambda: self.set_segmentation(1))
        self.kernel_dilation_spin.thistype = "kernel_dilation" 
        self.kernel_dilation_spin.grid(row=2, column=2)
        self.kernel_dilation_spin.delete(0,"end")
        self.kernel_dilation_spin.insert(0,self.literals["segdef"][1])
        self.kernel_dilation_spin.bind('<Return>', lambda *args: self.set_segmentation(1))
       
        # ttk.Label(self.configframe,text="Kernel erosion:",  font=('Helvetica', 8)).grid(row=3, column=0)
        # self.kernel_erosion_spin = tk.Spinbox(self.configframe, from_=self.literals["config"]["kernel_erosion"][0], to=self.literals["config"]["kernel_erosion"][1], increment=2, width=8, command=lambda: self.set_segmentation(2))
    
        ttk.Label(self.adjustfiltersframe,text="Kernel erosion:",  font=('Helvetica', 10)).grid(row=3, column=1)
        self.kernel_erosion_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["kernel_erosion"][0], to=self.literals["config"]["kernel_erosion"][1], increment=2, width=4, command=lambda: self.set_segmentation(2))
        self.kernel_erosion_spin.thistype = "kernel_erosion" 
        self.kernel_erosion_spin.grid(row=3, column=2)
        self.kernel_erosion_spin.delete(0,"end")
        self.kernel_erosion_spin.insert(0,self.literals["segdef"][2])
        self.kernel_erosion_spin.bind('<Return>', lambda *args: self.set_segmentation(2))

        # ttk.Label(self.configframe,text="Smoothing contours:",  font=('Helvetica', 8)).grid(row=4, column=0)
        # self.kernel_smoothing_contours_spin = tk.Spinbox(self.configframe, from_=self.literals["config"]["kernel_smoothing_contours"][0], to=self.literals["config"]["kernel_smoothing_contours"][1], increment=1, width=8, command=lambda: self.set_segmentation(3))
       
        ttk.Label(self.adjustfiltersframe,text="Smoothing contours:",  font=('Helvetica', 10)).grid(row=4, column=1)
        self.kernel_smoothing_contours_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["kernel_smoothing_contours"][0], to=self.literals["config"]["kernel_smoothing_contours"][1], increment=1, width=4, command=lambda: self.set_segmentation(3))
        self.kernel_smoothing_contours_spin.thistype = "kernel_smoothing_contours" 
        self.kernel_smoothing_contours_spin.grid(row=4, column=2)
        self.kernel_smoothing_contours_spin.delete(0,"end")
        self.kernel_smoothing_contours_spin.insert(0,self.literals["segdef"][3])
        self.kernel_smoothing_contours_spin.bind('<Return>', lambda *args: self.set_segmentation(3))

        # ttk.Label(self.configframe,text="Border thickness:",  font=('Helvetica', 8)).grid(row=5, column=0)
        # self.border_thickness_spin = tk.Spinbox(self.configframe, from_=self.literals["config"]["border_thickness"][0], to=self.literals["config"]["border_thickness"][1], increment=1, width=8, command=lambda: self.set_segmentation(4))

        ttk.Label(self.adjustfiltersframe,text="Border thickness:",  font=('Helvetica', 10)).grid(row=5, column=1)
        self.border_thickness_spin = tk.Spinbox(self.adjustfiltersframe, from_=self.literals["config"]["border_thickness"][0], to=self.literals["config"]["border_thickness"][1], increment=1, width=4, command=lambda: self.set_segmentation(4))
        self.border_thickness_spin.thistype = "border_thickness"
        self.border_thickness_spin.grid(row=5, column=2)
        self.border_thickness_spin.delete(0,"end")
        self.border_thickness_spin.insert(0,self.literals["segdef"][4])
        self.border_thickness_spin.bind('<Return>', lambda *args: self.set_segmentation(4))

        self.adjustfiltersframe.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)
        # self.adjustfiltersframe.grid(row=1, rowspan=5, column=0, columnspan=2)
        # self.adjustfiltersframe.grid(row=1, rowspan=2, column=0, columnspan=2)
        #
        # Configuration buttons frame creation

        for ic in range(3):
            self.adjustfiltersframe.columnconfigure(ic, weight=1)
        for ir in range(6):
            self.adjustfiltersframe.rowconfigure(ir, weight=1)
        #

        self.segbtnsframe = ttk.Frame(self.configframe) 

        # self.groupchecklist = ttkScrollFrame(self.frameCellLengthPlot, 'greyBackground.TFrame', self.literals["controller"].bgcolor)


        self.seg_for_all = ttk.Button(self.segbtnsframe, text="Set for Wave", command=lambda: self.set_segmentation("all"), width=11, style="small.TButton")
        # self.seg_for_all.grid(row=0, column=1, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
        # self.seg_for_all.grid(row=1, column=1, columnspan=1)
        self.seg_for_all.grid(row=1, column=0, columnspan=1)
    
        self.seg_for_allW = ttk.Button(self.segbtnsframe, text="Set for All", command=lambda: self.set_segmentation("allwaves"), width=11, style="small.TButton")
        # self.seg_for_allW.grid(row=0, column=2, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S) 
        # self.seg_for_allW.grid(row=1, column=2, columnspan=1) 
        self.seg_for_allW.grid(row=1, column=1, columnspan=1) 

        self.seg_for_allP = ttk.Button(self.segbtnsframe, text="Analyse", command=self.calculate_length_current, width=11, style="small.TButton")
        # self.seg_for_allP.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S) 
        # self.seg_for_allP.grid(row=2, column=1, columnspan=2)
        self.seg_for_allP.grid(row=1, column=2, columnspan=2)
        
        self.segbtnsframe.grid_rowconfigure(0, weight=1)
        self.segbtnsframe.grid_rowconfigure(1, weight=1)
        self.segbtnsframe.grid_rowconfigure(2, weight=1)
        self.segbtnsframe.grid_columnconfigure(0, weight=1)
        self.segbtnsframe.grid_columnconfigure(1, weight=1)
        self.segbtnsframe.grid_columnconfigure(2, weight=1)
        self.segbtnsframe.grid_columnconfigure(3, weight=1)
        # self.segbtnsframe.grid_columnconfigure(5, weight=1)
        # self.segbtnsframe.grid(row=7, column=0, columnspan=2, rowspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
        # self.segbtnsframe.grid(row=6, column=0, columnspan=2, rowspan=1)
        # self.segbtnsframe.grid(row=5, column=0, columnspan=2, rowspan=1)
        self.segbtnsframe.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)



        #
        # Cell segmentation view creation
        #
        self.frameSegment = ttk.Frame(self.configframe)
        # self.segmentLabel = ttk.Label(self.frameSegment, text="Segmentation Box View:", font=self.literals["controller"].subtitle_font)
        self.frameSegment2 = ttk.Frame(self.frameSegment)
        self.figSegment = plt.figure(figsize=(6, 2), dpi=100 ,facecolor=self.literals["controller"].bgcolor, edgecolor="None")
        self.gsSegment = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2, left=None, bottom=None, right=None, top=None)
        self.canvasSegment = FigureCanvasTkAgg(self.figSegment, master=self.frameSegment2)  # A tk.DrawingArea.
        self.axSegment = self.figSegment.add_subplot(self.gsSegment[0])
        self.canvasSegment.draw()
        self.canvasSegment.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        for ic in range(6):
            self.frameSegment.columnconfigure(ic, weight=1)
        # for ir in range(7):
        for ir in range(1):
            self.frameSegment.rowconfigure(ir, weight=1)

        # self.segmentLabel.grid(row=0, column=3, columnspan=1, sticky=tk.NSEW)
        # self.frameSegment2.grid(row=0, column=0, rowspan=6, columnspan=5, sticky=tk.NSEW)
        self.frameSegment2.grid(row=0, column=0, columnspan=5, sticky=tk.NSEW)
        self.frameSegment2.bind("<Configure>", self.resize)
        # self.frameSegment.grid(row=0, column=2, ro
        # wspan=5, columnspan=8, sticky=tk.NSEW)
        # self.frameSegment.grid(row=0, column=2, rowspan=5, columnspan=8, sticky=tk.NSEW)
        # self.frameSegment.grid(row=0, column=2, rowspan=5, columnspan=8, sticky=tk.NSEW)
        self.frameSegment.grid(row=1, column=2, columnspan=8, sticky=tk.NSEW)

        #
        # Height, Width Frame creation
        #
        self.info_calc_frame = ttk.Frame(self.configframe)
        self.height_label_calc = ttk.Label(self.info_calc_frame, text="Height: ", font=self.literals["controller"].subtitle_font)
        self.height_label_calc.grid(row=0, column=1, columnspan=1, sticky=tk.NSEW)
        self.width_label_calc = ttk.Label(self.info_calc_frame, text="Width: ", font=self.literals["controller"].subtitle_font)
        self.width_label_calc.grid(row=0, column=2, columnspan=1, sticky=tk.NSEW)
        self.area_label_calc = ttk.Label(self.info_calc_frame, text="Area: ", font=self.literals["controller"].subtitle_font)
        # self.area_label_calc.grid(row=0, column=3, columnspan=1, sticky=tk.NSEW)
        for ic in range(5):
            self.info_calc_frame.columnconfigure(ic, weight=1)
        for ir in range(1):
            self.info_calc_frame.rowconfigure(ir, weight=1)
        # self.info_calc_frame.grid(row=5, column=2, columnspan=8, sticky=tk.NSEW)
        self.info_calc_frame.grid(row=2, column=2, columnspan=8, sticky=tk.NSEW)

        #
        # Slider creation
        #
        # self.current_framelist = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]
        self.sliderframe = ttk.Frame(self.configframe)
        self.slidervarSegment = tk.IntVar(value=self.index_of_peak+1)
        self.slider = ttk.Scale(self.sliderframe, from_=1, to=len(self.current_peak.peaktimes), variable=self.slidervarSegment, orient=tk.HORIZONTAL, command=self.update_figure)
        # self.slider = ttk.Scale(self.masterframe.viewPort, from_=1, to=len(self.current_peak.peaktimes), variable=self.slidervarSegment, orient=tk.HORIZONTAL, command=self.update_figure)
        # self.slider.grid(row=rown, column=0, columnspan=3, sticky=tk.NSEW)
        for ic in range(5):
            self.sliderframe.columnconfigure(ic, weight=1)
        for ir in range(1):
            self.sliderframe.rowconfigure(ir, weight=1)
        self.slider.grid(row=0, column=1, columnspan=3, sticky=tk.NSEW)
        # self.sliderframe.grid(row=6, column=2, columnspan=8, sticky=tk.NSEW)
        self.sliderframe.grid(row=3, column=2, columnspan=8, sticky=tk.NSEW)

        # .grid(row=rown, column=3, rowspan=rspan, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S)

        # for ir in range(7):
        for ir in range(4):
            self.configframe.grid_rowconfigure(ir, weight=1)
        for ic in range(11):
            self.configframe.columnconfigure(ic, weight=1)

        rspan = 4

        self.configframe.grid(row=rown, column=0, rowspan=rspan, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S)

        rown += rspan

        #
        # Third Big Title creation
        #

        #Add title frame
        self.thirdBigTitleFrame = ttk.Frame(self.masterframe.viewPort)
        # self.firstBigTitleLabel = ttk.Label(self.firstBigTitleFrame, text="", font=self.literals["controller"].subtitle_font)
        self.thirdBigTitleLabel = ttk.Label(self.thirdBigTitleFrame, text="Cell shortening", font=self.literals["controller"].title_font)
        rspan = 1
        self.thirdBigTitleLabel.grid(row=0,column=2,rowspan=1,columnspan=2, sticky=tk.NSEW)
        
        self.thirdBigTitleFrame.grid_rowconfigure(0, weight=1)
        for ic in range(5):
            self.thirdBigTitleFrame.columnconfigure(ic, weight=1)

        self.thirdBigTitleFrame.grid(row=rown, column=0, rowspan=rspan, columnspan=6, sticky=tk.NSEW)

        rown += rspan

        #
        # Cell length Plot creation
        #

        self.frameCellLengthPlot = ttk.Frame(self.masterframe.viewPort)

        # self.groupchecklist = ttk.Frame(frameCellLengthPlot)
        self.groupchecklist = ttkScrollFrame(self.frameCellLengthPlot, 'greyBackground.TFrame', self.literals["controller"].bgcolor, cvSize =(1,4), cvExpand=False, cvWidth=100) #add scroll bar
        #6 columns

        mname = ttk.Label(self.groupchecklist.viewPort, text="Graph data:", font=('Helvetica', 14))
        mname.grid(row=0, column=1, columnspan=1, sticky=tk.NSEW)
        #for each Wave, print in column
        self.checkedgroups = []
        # self.method_lbls = []
        for iep, ep in enumerate(self.peaks):
            # cnvar = 
            self.checkedgroups.append(tk.IntVar())
            # self.method_lbls.append(None)
            # cnvar.set(0)
            self.checkedgroups[-1].set(0)
            # cname = ttk.Label(self.groupchecklist, text="Wave: " + str(iep+1), font=('Helvetica', 14))
            # cname.grid(row=iep+1, column=0, columnspan=1, sticky=tk.NSEW)
            # ccheck = ttk.Checkbutton(self.groupchecklist, text="Calculate", variable = self.checkedgroups[-1], width=20)
            ccheck = ttk.Checkbutton(self.groupchecklist.viewPort, text="Wave: " + str(iep+1), variable = self.checkedgroups[-1], width=30)
            # ccheck.grid(row=iep+1, column=1, columnspan=1, sticky=tk.NSEW)
            # ccheck.grid(row=iep+1, column=0, columnspan=1, sticky=tk.NSEW)
            ccheck.grid(row=iep+1, column=1, columnspan=1, sticky=tk.NSEW)
            # self.method_lbls[-1] = ttk.Label(self.groupchecklist, text="Full segmentation", font=('Helvetica', 14))
            # self.method_lbls[-1].grid(row=iep+1, column=2, columnspan=1, sticky=tk.NSEW)

        for ic in range(2):
            self.groupchecklist.viewPort.columnconfigure(ic, weight=1)
        for ir in range(len(self.peaks)+1):
            self.groupchecklist.viewPort.rowconfigure(ir, weight=1)
        self.groupchecklist.grid(row=0,column=0, columnspan=1, rowspan=7)


        # self.cellLengthLabel = ttk.Label(self.frameCellLengthPlot, text="Cell Length Plot:", font=self.literals["controller"].subtitle_font)
        self.frameCellLength2 = ttk.Frame(self.frameCellLengthPlot)
        self.figCellLength = plt.figure(figsize=(6, 4), dpi=100 ,facecolor=self.literals["controller"].bgcolor, edgecolor="None")
        self.gsCellLength = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2, left=None, bottom=None, right=None, top=None)
        self.canvasCellLength = FigureCanvasTkAgg(self.figCellLength, master=self.frameCellLength2)
        self.mainplotartistCellLength = None
        self.mainplotartistCellHeight = None
        self.axbaselineCellLength = None
        self.axgridCellLength = None
        self.axCellLength = self.figCellLength.add_subplot(self.gsCellLength[0])
        self.canvasCellLength.draw()
        self.canvasCellLength.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # self.cellLengthLabel.grid(row=0, column=3, columnspan=1, sticky=tk.NSEW)
        self.frameCellLength2.grid(row=0, column=1, rowspan=6, columnspan=6, sticky=tk.NSEW)

        self.cell_percdiff = tk.IntVar(value=0)
        self.cell_zerocenter = tk.IntVar(value=0)

        self.checkCellLengthFrame = ttk.Frame(self.frameCellLengthPlot)

        self.cell_plottype = tk.IntVar(value=0)
        self.radioLengthMicro = ttk.Radiobutton(self.checkCellLengthFrame, text = "µm", variable = self.cell_plottype, value = 0, command=self.calculate_length_current, width=5)
        self.radioLengthPerc = ttk.Radiobutton(self.checkCellLengthFrame, text = "%", variable = self.cell_plottype, value = 1, command=self.calculate_length_current, width=5)
        self.radioAreaMicro = ttk.Radiobutton(self.checkCellLengthFrame, text = "µm²", variable = self.cell_plottype, value = 2, command=self.calculate_length_current, width=5)
        self.radioAreaPerc = ttk.Radiobutton(self.checkCellLengthFrame, text = "%", variable = self.cell_plottype, value = 3, command=self.calculate_length_current, width=5)

        self.cell_calctype = tk.IntVar(value=0)
        
        self.radioAbsolute = ttk.Radiobutton(self.checkCellLengthFrame, text = "Absolute", variable = self.cell_calctype, value = 0, command=self.calculate_length_current, width=15)
        # self.radioRelativeFirst = ttk.Radiobutton(self.checkCellLengthFrame, text = "Baseline (Wave start)", variable = self.cell_calctype, value = 1, command=self.calculate_length_current, width=20)
        
        # self.radioRelativePrevious = ttk.Radiobutton(self.checkCellLengthFrame, text = "Relative (Previous)", variable = self.cell_calctype, value = 2, command=self.calculate_length_current, width=20)
        self.radioRelativeFirst2 = ttk.Radiobutton(self.checkCellLengthFrame, text = "Normalized", variable = self.cell_calctype, value = 3, command=self.calculate_length_current, width=20)

        ttk.Label(self.checkCellLengthFrame, text="Length:").grid(row=0,column=1)
        self.radioLengthMicro.grid(row=0, column=2)
        self.radioLengthPerc.grid(row=0, column=3)
        # ttk.Label(self.checkCellLengthFrame, text="Area:").grid(row=0,column=5)
        # self.radioAreaMicro.grid(row=0, column=6)
        # self.radioAreaPerc.grid(row=0, column=7)
        ttk.Label(self.checkCellLengthFrame, text="Values:").grid(row=1,column=1)
        self.radioAbsolute.grid(row=1, column=3)

        # self.radioRelativeFirst.grid(row=1, column=4)
        # self.radioRelativePrevious.grid(row=1, column=5)

        self.radioRelativeFirst2.grid(row=1, column=4)
        # self.radioRelativeFirst2.grid(row=1, column=6)

        # self.checkPercLength = ttk.Checkbutton(self.checkCellLengthFrame, text="Show as Percentages", variable = self.cell_percdiff, width=20, command=self.calculate_length_current)
        # self.checkPercLength.grid(row=0, column=1, sticky=tk.NSEW)
        # self.checkBaselineLength =  ttk.Checkbutton(self.checkCellLengthFrame, text="First point as baseline", variable = self.cell_zerocenter, width=20, command=self.calculate_length_current)
        # self.checkBaselineLength.grid(row=0, column=2, sticky=tk.NSEW)
        # for ic in range(4):
        for ic in range(8):
            self.checkCellLengthFrame.columnconfigure(ic, weight=1)
        for ir in range(2):
            self.checkCellLengthFrame.rowconfigure(ir, weight=1)
        self.checkCellLengthFrame.grid(row=6, column=2, rowspan=2, columnspan=4, sticky=tk.NSEW)

        rspan = 7

        for ic in range(7):
            self.frameCellLengthPlot.columnconfigure(ic, weight=1)
        for ir in range(8):
            self.frameCellLengthPlot.rowconfigure(ir, weight=1)
        self.frameCellLengthPlot.grid(row=rown, column=0, rowspan=6, columnspan=6, sticky=tk.NSEW)


        rown += rspan

        for ir in range(rown+1):
            self.masterframe.viewPort.grid_rowconfigure(ir, weight=1)

        self.masterframe.columnconfigure(0, weight=1)
        self.masterframe.columnconfigure(1, weight=1)
        self.masterframe.columnconfigure(2, weight=1)
        self.masterframe.grid_rowconfigure(0, weight=1)

        self.masterframe.grid(row=0, column=0, rowspan=8, columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.analysebtn = ttk.Button(master, text="Analyse", command=self.calculate_length_current, width=15)
        # self.analysebtn.grid(row=8, column=1, columnspan=1)
        # self.calculatebutton = ttk.Button(master, text="Export", command=self.calculate_length_exportAll, width=15)
        self.calculatebutton = ttk.Button(master, text="Export", command=self.calculate_length_exportAll2, width=15)
        # self.calculatebutton.grid(row=8, column=2, columnspan=1)
        self.calculatebutton.grid(row=8, column=1, columnspan=1)
        # self.grid_rowconfigure
        self.drawSegmentedImage()

        self.packbtns = False

    def resize(self , event=None):
        print("resized")
        U = self.mframe.controller.current_anglist[self.mframe.current_frame][0]
        Y,X=U.shape
        print(X, Y)
        print(event)
        print(self.frameSegment2.winfo_width())
        print(self.frameSegment2.winfo_height())
        if self.frameSegment2.winfo_width() <= X:
            X=self.frameSegment2.winfo_width()
        if self.frameSegment2.winfo_height() <= Y:
            X=self.frameSegment2.winfo_width()
        
        self.figSegment.set_size_inches(X/300, Y/300, forward=True)
        self.figSegment.axes[0].set_position(self.gsSegment[0].get_position(self.figSegment))
        # self.canvasSegment.get_tk_widget().config(width=X)
        # self.canvasSegment.get_tk_widget().config(height=Y)
        self.figCellLength.canvas.draw()
        self.canvasSegment.draw()

    def exportLengthData(self):
        try:
            SaveTableDialog(self, title='Save Length Data', literals=[
                ("headers", [self.axCellLength.get_xlabel(), self.axCellLength.get_ylabel()]),
                ("data", [self.mainplotartistCellLength[0].get_xdata(), self.mainplotartistCellLength[0].get_ydata()]),
                ("data_t", "single")
                ], parent2=self.mframe)
        except TypeError:
            messagebox.showerror("Error", "No Data in plot")

    def exportSegmentedSpeedData(self):
        try:
            SaveTableDialog(self.mframe, title='Save Seg. Speed Data', literals=[
                ("headers", [self.axSpeedComp.get_xlabel(), self.axSpeedComp.get_ylabel()]),
                ("data", [self.mainplotartistSpeedComp[0].get_xdata(), self.mainplotartistSpeedComp[0].get_ydata()]),
                ("data_t", "single")
                ], parent2=self.mframe)
        except TypeError:
            messagebox.showerror("Error", "No Data in plot")

    def tableYScroll(self, *args):
        self.tableTree.yview(*args)

    def tableXScroll(self, *args):
        self.tableTree.xview(*args)

    def config_segmentation_spinners(self, event=None):
        self.blur_size_spin.delete(0,"end")
        self.blur_size_spin.insert(0,self.segmentation_dict[self.current_peak_index][self.index_of_peak][0])
        self.kernel_dilation_spin.delete(0,"end")
        self.kernel_dilation_spin.insert(0,self.segmentation_dict[self.current_peak_index][self.index_of_peak][1])
        self.kernel_erosion_spin.delete(0,"end")
        self.kernel_erosion_spin.insert(0,self.segmentation_dict[self.current_peak_index][self.index_of_peak][2])
        self.kernel_smoothing_contours_spin.delete(0,"end")
        self.kernel_smoothing_contours_spin.insert(0,self.segmentation_dict[self.current_peak_index][self.index_of_peak][3])
        self.border_thickness_spin.delete(0,"end")
        self.border_thickness_spin.insert(0,self.segmentation_dict[self.current_peak_index][self.index_of_peak][4])

    def set_segmentation(self, idseg, event=None):
        #On segmentation list
        getvalue = None
        if idseg == 0:
            getvalue = int(self.blur_size_spin.get().replace(",", "."))
            self.segmentation_dict[self.current_peak_index][self.index_of_peak][idseg] = getvalue
        elif idseg == 1:
            getvalue = int(self.kernel_dilation_spin.get().replace(",", "."))
            self.segmentation_dict[self.current_peak_index][self.index_of_peak][idseg] = getvalue
        elif idseg == 2:
            getvalue = int(self.kernel_erosion_spin.get().replace(",", "."))
            self.segmentation_dict[self.current_peak_index][self.index_of_peak][idseg] = getvalue
        elif idseg == 3:
            getvalue = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            self.segmentation_dict[self.current_peak_index][self.index_of_peak][idseg] = getvalue
        elif idseg == 4:
            getvalue = int(self.border_thickness_spin.get().replace(",", "."))
            self.segmentation_dict[self.current_peak_index][self.index_of_peak][idseg] = getvalue
        elif idseg == "all":
            getvalue1 = int(self.blur_size_spin.get().replace(",", "."))
            getvalue2 = int(self.kernel_dilation_spin.get().replace(",", "."))
            getvalue3 = int(self.kernel_erosion_spin.get().replace(",", "."))
            getvalue4 = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            getvalue5 = int(self.border_thickness_spin.get().replace(",", "."))
            getvalues = [getvalue1, getvalue2, getvalue3, getvalue4, getvalue5]
            for iop in range(len(self.segmentation_dict[self.current_peak_index])):
                self.segmentation_dict[self.current_peak_index][iop]= getvalues.copy()
        elif idseg == "allwaves":
            getvalue1 = int(self.blur_size_spin.get().replace(",", "."))
            getvalue2 = int(self.kernel_dilation_spin.get().replace(",", "."))
            getvalue3 = int(self.kernel_erosion_spin.get().replace(",", "."))
            getvalue4 = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            getvalue5 = int(self.border_thickness_spin.get().replace(",", "."))
            getvalues = [getvalue1, getvalue2, getvalue3, getvalue4, getvalue5]
            for pi in range(len(self.segmentation_dict.keys())):
                for iop in range(len(self.segmentation_dict[pi])):
                    self.segmentation_dict[pi][iop]= getvalues.copy()
        self.drawSegmentedImage()
        # self.calculate_length_current()

    def set_image_list(self, pi=None):
        self.current_framelist = []
        # self.current_segmentation_settings_list = []
        if pi is None:
            # print("pi is none")
            pi = self.current_peak_index
        try:
            if self.literals["controller"].current_analysis.gtype == "Folder":
                global img_opencv
                files_grabbed = [x for x in os.listdir(self.literals["controller"].current_analysis.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
                framelist = sorted(files_grabbed)
                files_grabbed_now = framelist[self.literals["controller"].selectedframes[self.peaks[pi].first]:(self.literals["controller"].selectedframes[self.peaks[pi].last+1])+1]
                files_grabbed_now = [self.literals["controller"].current_analysis.gpath + "/" + a for a in files_grabbed_now]
                for j in range(len(files_grabbed_now)-1):
                    frame1 = cv2.imread(r'%s' %files_grabbed_now[0+j])
                    self.current_framelist.append(frame1)
                    # self.current_segmentation_settings_list.append(self.segmentation_default.copy())
            elif self.literals["controller"].current_analysis.gtype == "Video":
                vc = cv2.VideoCapture(r'%s' %self.literals["controller"].current_analysis.gpath)
                count = self.literals["controller"].selectedframes[self.peaks[pi].first]
                vc.set(1, count-1)
                while(vc.isOpened() and count < (self.literals["controller"].selectedframes[self.peaks[pi].last]+1)):
                    _, frame1 = vc.read()
                    self.current_framelist.append(frame1)
                    count += 1
                vc.release()
            elif self.literals["controller"].current_analysis.gtype == "Tiff Directory" or self.literals["controller"].current_analysis.gtype == "CTiff":
                _, images = cv2.imreadmulti(r'%s' %self.literals["controller"].current_analysis.gpath, None, cv2.IMREAD_COLOR)
                images = images[self.literals["controller"].selectedframes[self.peaks[pi].first]:self.literals["controller"].selectedframes[(self.peaks[pi].last)+1]]
                for j in range(len(images)-1):
                    frame1 = images[0+j]
                    self.current_framelist.append(frame1)
        except Exception as e:
            messagebox.showerror("Error", "Could not retrieve frames\n" + str(e))

    def update_figure(self, event=None):
        self.literals["controller"].btn_lock = True
        current_fnum = self.slidervarSegment.get() - 1
        self.index_of_peak = current_fnum
        
        if self.rectWave != None:
            self.rectWave.remove()
        self.rectWave = None
        if self.rectContour != None:
            self.rectContour.remove()
        self.rectContour = None
        if self.rectCellLength != None:
            self.rectCellLength.remove()
        self.rectCellLength = None
        
        curlims = (self.axWave.get_xlim(), self.axWave.get_ylim())
        curlims2 = (self.axCellLength.get_xlim(), self.axCellLength.get_ylim())

        new_rect = Rectangle((0,0), 1, 1)
        new_rect.set_width(2 * self.half_timeWave)
        new_rect.set_height(curlims[1][1] + abs(curlims[1][0]) + 0.5)
        
        new_rect2 = Rectangle((0,0), 1, 1)
        new_rect2.set_width(2 * self.half_timeWave)
        new_rect2.set_height(curlims2[1][1] + abs(curlims2[1][0]) + 0.5)

        if self.mframe.plotsettings.plotline_opts["absolute_time"] == False:
            zerotime = self.current_peak.firsttime
            new_rect.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave - zerotime, curlims[1][0]))
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave - zerotime, curlims2[1][0]))
        else:
            new_rect.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave, curlims[1][0]))
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave, curlims2[1][0]))
        new_rect.set_facecolor(self.mframe.plotsettings.peak_plot_colors['rect_color'])
        new_rect2.set_facecolor(self.mframe.plotsettings.peak_plot_colors['rect_color'])
        self.axWave.set_xlim(curlims[0])
        self.axWave.set_ylim(curlims[1])
        self.rectWave = self.axWave.add_patch(new_rect)
        self.rectCellLength = self.axCellLength.add_patch(new_rect2)
        self.figWave.canvas.draw()
        self.figCellLength.canvas.draw()
        self.drawSegmentedImage()
        self.config_segmentation_spinners()
        self.literals["controller"].btn_lock = False
        return True

    def drawSegmentedImage(self, pi=None, ii=None, refreshFrames=True):
        if self.rectContour != None:
            self.rectContour.remove()
        self.rectContour = None
        #use index_of_peak to pull image, and draw contours using mframe functions
        # imgget = self.current_framelist[self.index_of_peak].copy()
        #plot image
        if pi is None:
            # print("pi is none")
            pi = self.current_peak_index
        if ii is None:
            # print("ii is none")
            ii = self.index_of_peak
        imgget = self.current_framelist[ii].copy()
        # bsize, kdil, kero, ksco, borders = self.current_segmentation_settings_list[self.index_of_peak]
        bsize, kdil, kero, ksco, borders = self.segmentation_dict[pi][ii]

        mask, contourdimensions, largest_area_rect, largest_contour = self.mframe.get_contour(imgget, bsize, kdil, kero, ksco, borders)
        # (largest_area_rect_x, largest_area_rect_y), (largest_area_rect_width, largest_area_rect_height), largest_area_rect_angle = largest_area_rect

        height_value = "{:.3f}".format(contourdimensions[1] * self.literals["controller"].current_analysis.pixelsize)
        width_value = "{:.3f}".format(contourdimensions[0] * self.literals["controller"].current_analysis.pixelsize)


        # box = cv2.boxPoints(largest_area_rect) # cv2.boxPoints(rect) for OpenCV 3.x
        # box = np.int0(box)
        # imgrect = cv2.drawContours(imgget ,[box],0,(0,0,255),2)
        imgrect = cv2.rectangle(imgget ,(largest_area_rect[0],largest_area_rect[1]),(largest_area_rect[0]+largest_area_rect[2],largest_area_rect[1]+largest_area_rect[3]),(0,0,255),2)

        imgrect2 = cv2.drawContours(imgrect ,largest_contour,0,(0,0,255),2)
        cont_area = "{:.3f}".format(cv2.contourArea(largest_contour[-1]) * self.literals["controller"].current_analysis.pixelsize)
        if refreshFrames == True:
            self.height_label_calc['text'] = "Height: " + height_value + "µm"
            self.width_label_calc['text'] = "Width: " + width_value + "µm"        # print("There are: " + str(len(box)) + " rectangles ")
            self.area_label_calc['text'] = "Area: " + width_value + "µm²"        # print("There are: " + str(len(box)) + " rectangles ")
        
        #plot rectangle in image
        #here viz is reset and plotted according to selection
        if refreshFrames == True:
            # print("refresh seg img")
            self.axSegment.clear()
            # self.figSegment.canvas.draw()
            # self.canvasSegment.draw()
            self.axSegment.spines['top'].set_visible(False)
            self.axSegment.spines['right'].set_visible(False)
            self.axSegment.spines['bottom'].set_visible(False)
            self.axSegment.spines['left'].set_visible(False)
            self.axSegment.set_xticks([], [])
            self.axSegment.set_yticks([], [])
            self.axSegment.set_xticklabels([])
            self.axSegment.set_yticklabels([])
            # self.axSegment.set_facecolor('black')
            # self.axSegment.axis('on')
            self.rectContour = self.axSegment.imshow(imgrect2)
            self.figSegment.canvas.draw()
            self.canvasSegment.draw()
            return
        else:
            return float(height_value), float(width_value), float(cont_area)

    # def set_figure(self, figx, figy):
    #     self.figmax.set_size_inches(figx/300, figy/300, forward=True)
    #     self.figmax.axes[0].set_position(self.gs_nobordermax[0].get_position(self.figmax))
    #     self.canvasmax.get_tk_widget().config(width=figx)
    #     self.canvasmax.get_tk_widget().config(height=figy)

    #     self.figmax.canvas.draw()
    #     self.canvasmax.draw()


    def update_screen_method(self):
        #add/hide slider depending on radio btn select
        #value 0 means full segmentation
        #value 1 means reference according to table
        # for elbl in self.method_lbls:
        #     if self.cell_length_method.get() == 0:
        #         elbl['text'] = "Full segmentation"
        #     else:
        #         elbl['text'] = "Reference + Plot Area"
        for ivar in self.checkedgroups:
            print("checked:" + str(ivar.get()) )

    def tabletreeselection(self, event=None):
        self.literals["controller"].btn_lock = True
        # self.figWave
        # self.mainplotartistWave
        # self.axWave 
        # self.axbaselineWave 
        # self.axgridWave
        # self.canvasWave

        self.axWave.clear()
        self.axWave.set_xlabel("Time ("+self.literals["controller"].current_timescale+")")
        self.axWave.set_ylabel("Average Speed ("+self.literals["controller"].current_speedscale+")")

        self.mainplotartistWave = None
        try:
            curItem = self.tableTree.focus()
            # curItem = self.tableTree.selection()
            diff_list1 = [a for a in list(self.tableTree.selection()) if a not in self.treePrevSelectedNames]
            diff_list2 = [b for b in self.treePrevSelectedNames if b not in list(self.tableTree.selection())]
            if len(diff_list1) > 0: #new items to add
                curItem = diff_list1[-1]
                # curItem = self.tableTree.selection()
                # for eit in list(self.tableTree.selection()):
                    # if eit not in self.treePrevSelectedNames:
                        # curItem = eit
                        # self.treePrevSelectedNames.append(eit)
                        # break
            elif len(diff_list2) > 0: #items to remove
                curItem = list(self.tableTree.selection())[-1]
            # elif len(self.tableTree.selection()) == 1:
            #     self.treePrevSelectedNames = [curItem]
            # else:
            #     self.treePrevSelectedNames = []
            self.treePrevSelectedNames = list(self.tableTree.selection())
            # self.tableTree.
            # print("curItem")
            # print(curItem)
            # print("self.tableTree.selection()")
            # print(self.tableTree.selection())
            # print("len(self.tableTree.selection())")
            # print(len(self.tableTree.selection()))
            # print("self.tableTree.selection_get()")
            # print(self.tableTree.selection_get())
            # print("self.tableTree.focus_get()")
            # print(self.tableTree.focus_get())
            curItemDecomp = self.tableTree.item(curItem)
            curRow = int(curItemDecomp["text"])
            self.firstBigTitleLabel['text'] = "Contraction-relaxation Wave " + str(int(curItemDecomp["text"]) + 1)
            self.current_peak_index = curRow
            self.current_peak = self.peaks[self.current_peak_index]
            self.set_image_list()

            self.slider["to"] = len(self.current_framelist)
            self.slider.grid_forget()
            # self.slider.grid(row=13, column=0, columnspan=3, sticky=tk.NSEW)
            self.slider.grid(row=0, column=1, columnspan=3, sticky=tk.NSEW)
            self.index_of_peak = 0
            self.slidervarSegment.set(self.index_of_peak+1)
            
            if self.mframe.plotsettings.plotline_opts["absolute_time"] == True:
                self.mainplotartistWave = self.axWave.plot(self.peaks[curRow].peaktimes, self.peaks[curRow].peakdata, color=self.mframe.plotsettings.peak_plot_colors["main"])
                if self.mframe.plotsettings.plotline_opts["show_dots"] == True:
                    self.axWave.plot(self.peaks[curRow].firsttime, self.peaks[curRow].firstvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["first"], picker=5)
                    self.axWave.plot(self.peaks[curRow].secondtime, self.peaks[curRow].secondvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["max"], picker=5)
                    self.axWave.plot(self.peaks[curRow].thirdtime, self.peaks[curRow].thirdvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["min"], picker=5)
                    self.axWave.plot(self.peaks[curRow].fourthtime, self.peaks[curRow].fourthvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["max"], picker=5)
                    self.axWave.plot(self.peaks[curRow].fifthtime, self.peaks[curRow].fifthvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["last"], picker=5)
            else:
                zerotime = self.peaks[curRow].firsttime
                self.mainplotartistWave = self.axWave.plot([ttime - zerotime for ttime in self.peaks[curRow].peaktimes], self.peaks[curRow].peakdata, color=self.mframe.plotsettings.peak_plot_colors["main"])
                if self.mframe.plotsettings.plotline_opts["show_dots"] == True:
                    self.axWave.plot(self.peaks[curRow].firsttime - zerotime, self.peaks[curRow].firstvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["first"], picker=5)
                    self.axWave.plot(self.peaks[curRow].secondtime - zerotime, self.peaks[curRow].secondvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["max"], picker=5)
                    self.axWave.plot(self.peaks[curRow].thirdtime - zerotime, self.peaks[curRow].thirdvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["min"], picker=5)
                    self.axWave.plot(self.peaks[curRow].fourthtime - zerotime, self.peaks[curRow].fourthvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["max"], picker=5)
                    self.axWave.plot(self.peaks[curRow].fifthtime - zerotime, self.peaks[curRow].fifthvalue, "o", linewidth=2, fillstyle='none', color=self.mframe.plotsettings.peak_plot_colors["last"], picker=5)
        except ValueError:
            pass

        self.half_timeWave = self.current_peak.peaktimes[1] - self.current_peak.peaktimes[0]

        curlims = (self.axWave.get_xlim(), self.axWave.get_ylim())
        curlims2 = (self.axCellLength.get_xlim(), self.axCellLength.get_ylim())

        if self.rectWave != None:
            self.rectWave.remove()
        self.rectWave = None
        if self.rectContour != None:
            self.rectContour.remove()
        self.rectContour = None
        if self.rectCellLength != None:
            self.rectCellLength.remove()
        self.rectCellLength = None

        new_rect = Rectangle((0,0), 1, 1)
        new_rect.set_width(2 * self.half_timeWave)
        new_rect.set_height(curlims[1][1] + abs(curlims[1][0]) + 0.5)

        new_rect2 = Rectangle((0,0), 1, 1)
        new_rect2.set_width(2 * self.half_timeWave)
        new_rect2.set_height(curlims2[1][1] + abs(curlims2[1][0]) + 0.5)

        if self.mframe.plotsettings.plotline_opts["absolute_time"] == True:
            new_rect.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave, curlims[1][0]))
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave, curlims2[1][0]))
        else:
            zerotime = self.current_peak.firsttime
            new_rect.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave - zerotime, curlims[1][0]))
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave - zerotime, curlims2[1][0]))
        
        new_rect.set_facecolor(self.mframe.plotsettings.peak_plot_colors['rect_color'])
        new_rect2.set_facecolor(self.mframe.plotsettings.peak_plot_colors['rect_color'])

        print("new_rect2")
        print(new_rect2)

        self.rectWave = self.axWave.add_patch(new_rect)
        self.rectCellLength = self.axCellLength.add_patch(new_rect2)
        print("self.rectCellLength")
        print(self.rectCellLength)
        # self.figWave.tight_layout()

        if self.axbaselineWave != None:
            self.axbaselineWave.remove()
            self.axbaselineWave = None
        if self.mframe.plotsettings.plotline_opts["zero"] == True:
            self.axbaselineWave = self.axWave.axhline(y=0.0, color=self.mframe.plotsettings.plotline_opts["zero_color"], linestyle='-')

        if self.axgridWave != None:
            self.axgridWave.remove()
            self.axgridWave = None
        if self.mframe.plotsettings.plotline_opts["grid"] == True:
            self.axgridWave = self.axWave.grid(linestyle="-", color=self.mframe.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.axWave.grid(False)
        
        self.axWave.set_xlim(curlims[0])
        self.axWave.set_ylim(curlims[1])

        if self.pressmovecidWave != None:
            self.figWave.canvas.mpl_disconnect(self.pressmovecidWave)
            self.pressmovecidWave = None
        self.pressmovecidWave = self.figWave.canvas.mpl_connect("button_press_event", self.on_press_event_slider)
        
        self.figWave.canvas.draw()
        self.figCellLength.canvas.draw()
        self.update_figure()
        self.calculate_length_current()
        self.literals["controller"].btn_lock = False

    def on_press_event_slider(self, event):
        array = None
        if self.mframe.plotsettings.plotline_opts["absolute_time"] == True:
            array = np.asarray(self.current_peak.peaktimes)
        else:
            zerotime = self.current_peak.firsttime
            array = np.asarray([ttime - zerotime for ttime in self.current_peak.peaktimes])
        try:
            idx = (np.abs(array - event.xdata)).argmin()
            self.slidervarSegment.set(idx+1)
            a = self.update_figure()
            if a == True:
                return a
        except TypeError:
            pass

    def calculate_length_current(self):
        #todo: calculate current and add to plot
        #todo: add plot of current pspeed segmented pspeed and new segmented pspeed
        # list_lengths = []
        width_lengths = []
        height_lengths = []
        area_lengths = []
        for e_index in range(len(self.peaks[self.current_peak_index].peaktimes)):
            h_value, w_value,c_area = self.drawSegmentedImage(pi=None, ii=e_index, refreshFrames=False)
            # list_lengths.append([h_value, w_value])
            width_lengths.append(w_value)
            height_lengths.append(h_value)
            area_lengths.append(c_area)

        xlbl = ""
        if self.cell_plottype.get() == 0:
            xlbl = "Cell length (µm)"
            fpval = width_lengths[0]
            if self.cell_calctype.get() == 0:
                width_lengths = [a for a in width_lengths]
            elif self.cell_calctype.get() == 1:
                width_lengths = [a - fpval for a in width_lengths]
            elif self.cell_calctype.get() == 2:
                o_width_lengths = [0.0]
                width_lengths = [width_lengths[i] - width_lengths[i-1] for i in range(1, len(width_lengths))]
                o_width_lengths.extend(width_lengths)
                width_lengths = o_width_lengths.copy()
            elif self.cell_calctype.get() == 3:
                n_width_lengths = []
                for a in width_lengths:
                    abs_val = np.abs(a - fpval)
                    if fpval > a:
                        abs_val *= -1.0
                    n_width_lengths.append(abs_val)
                width_lengths = n_width_lengths.copy()
        if self.cell_plottype.get() == 1:
            xlbl = "Cell length (%)"
            fpval = width_lengths[0]
            if self.cell_calctype.get() == 0:
                width_lengths = [(a / fpval) * 100.0 for a in width_lengths]
            elif self.cell_calctype.get() == 1:
                width_lengths = [((a / fpval) * 100.0) - 100.0 for a in width_lengths]
            elif self.cell_calctype.get() == 2:
                o_width_lengths = [0.0]
                n_width_lengths = [(width_lengths[i] / fpval) - (width_lengths[i-1] / fpval) for i in range(1, len(width_lengths))]
                o_width_lengths.extend(n_width_lengths)
                width_lengths = o_width_lengths.copy()
            elif self.cell_calctype.get() == 3:
                n_width_lengths = []
                for a in width_lengths:
                    abs_val = np.abs(((a / fpval) * 100.0) - 100.0)
                    if ((a / fpval) * 100.0) < 100.0 :
                        abs_val *= -1.0
                    n_width_lengths.append(abs_val)
                width_lengths = n_width_lengths.copy()
        if self.cell_plottype.get() == 2:
            xlbl = "Cell area (µm²)"
            fpval = area_lengths[0]
            if self.cell_calctype.get() == 0:
                width_lengths = [a for a in area_lengths]
            elif self.cell_calctype.get() == 1:
                width_lengths = [a - fpval for a in area_lengths]
            elif self.cell_calctype.get() == 2:
                width_lengths = [0.0]
                n_width_lengths = [area_lengths[i] - area_lengths[i-1] for i in range(1, len(area_lengths))]
                width_lengths.extend(n_width_lengths)
            elif self.cell_calctype.get() == 3:
                n_width_lengths = []
                for a in area_lengths:
                    abs_val = np.abs(a - fpval)
                    if fpval > a:
                        abs_val *= -1.0
                    n_width_lengths.append(abs_val)
                width_lengths = n_width_lengths.copy()
        if self.cell_plottype.get() == 3:
            xlbl = "Cell area (%)"
            fpval = area_lengths[0]
            if self.cell_calctype.get() == 0:
                width_lengths = [(a / fpval) * 100.0 for a in area_lengths]
            elif self.cell_calctype.get() == 1:
                width_lengths = [((a / fpval) * 100.0) - 100.0 for a in area_lengths]
            elif self.cell_calctype.get() == 2:
                o_width_lengths = [0.0]
                n_width_lengths = [(area_lengths[i] / fpval) - (area_lengths[i-1] / fpval) for i in range(1, len(area_lengths))]
                o_width_lengths.extend(n_width_lengths)
                width_lengths = o_width_lengths.copy()
            elif self.cell_calctype.get() == 3:
                n_width_lengths = []
                for a in area_lengths:
                    abs_val = np.abs(((a / fpval) * 100.0) - 100.0)
                    if ((a / fpval) * 100.0) < 100.0 :
                        abs_val *= -1.0
                    n_width_lengths.append(abs_val)
                width_lengths = n_width_lengths.copy()

        # cell_zerocenter
        # fpval = width_lengths[0]
        # if self.cell_zerocenter.get() == 1:
        #     width_lengths = [a - fpval for a in width_lengths]
        # # cell_percdiff
        # if self.cell_percdiff.get() == 1:
        #     width_lengths = [(a / fpval) * 100.0 for a in width_lengths]
        
        #add to length plots
        self.axCellLength.clear()
        self.mainplotartistCellLength = None
        self.mainplotartistCellHeight = None
        self.axbaselineCellLength = None
        self.axgridCellLength = None
        # self.axCellLength.set_title("Select an interval for analysis:")
        self.axCellLength.set_xlabel("Time ("+self.literals["controller"].current_timescale+")")
        self.axCellLength.set_ylabel(xlbl)
        
        if self.mframe.plotsettings.plotline_opts["absolute_time"] == True:
            self.mainplotartistCellLength = self.axCellLength.plot(self.peaks[self.current_peak_index].peaktimes, width_lengths, color=self.mframe.plotsettings.peak_plot_colors["main"])
            # min_wd = np.amin(width_lengths)
            # self.axCellLength.plot(self.peaks[self.current_peak_index].peaktimes[min_wd], width_lengths[min_wd], ".", color=self.mframe.plotsettings.peak_plot_colors["min"])

            # self.mainplotartistCellLength = self.axCellLength.plot(width_lengths, color=self.mframe.plotsettings.peak_plot_colors["main"])
            # self.mainplotartistCellHeight = self.axCellLength.plot(self.peaks[self.current_peak_index].peaktimes, height_lengths, color=self.mframe.plotsettings.peak_plot_colors["fft"])
        else:
            zerotime = self.peaks[self.current_peak_index].firsttime
            self.mainplotartistCellLength = self.axCellLength.plot([ttime - zerotime for ttime in self.peaks[self.current_peak_index].peaktimes], width_lengths, color=self.mframe.plotsettings.peak_plot_colors["main"])
            # min_wd = np.amin(width_lengths)
            # self.axCellLength.plot([ttime - zerotime for ttime in self.peaks[self.current_peak_index].peaktimes][min_wd], width_lengths[min_wd], ".", color=self.mframe.plotsettings.peak_plot_colors["min"])

            # self.mainplotartistCellLength = self.axCellLength.plot(width_lengths, color=self.mframe.plotsettings.peak_plot_colors["main"])
            # self.mainplotartistCellHeight = self.axCellLength.plot([ttime - zerotime for ttime in self.peaks[self.current_peak_index].peaktimes], height_lengths, color=self.mframe.plotsettings.peak_plot_colors["fft"])
        # print("width_lengths")
        # print(width_lengths)
        self.axCellLength.set_ylim(np.min(width_lengths)-0.2, np.max(width_lengths)+0.2)

        curlims = (self.axCellLength.get_xlim(), self.axCellLength.get_ylim())

        if self.axbaselineCellLength != None:
            self.axbaselineCellLength.remove()
            self.axbaselineCellLength = None
        if self.mframe.plotsettings.plotline_opts["zero"] == True:
            self.axbaselineCellLength = self.axCellLength.axhline(y=0.0, color=self.mframe.plotsettings.plotline_opts["zero_color"], linestyle='-')

        if self.axgridCellLength != None:
            self.axgridCellLength.remove()
            self.axgridCellLength = None
        if self.mframe.plotsettings.plotline_opts["grid"] == True:
            self.axgridCellLength = self.axCellLength.grid(linestyle="-", color=self.mframe.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.axCellLength.grid(False)
        
        if self.rectCellLength != None:
            self.rectCellLength.remove()
        self.rectCellLength = None
        
        new_rect2 = Rectangle((0,0), 1, 1)
        new_rect2.set_width(2 * self.half_timeWave)
        new_rect2.set_height(curlims[1][1] + abs(curlims[1][0]) + 0.5)
        if self.mframe.plotsettings.plotline_opts["absolute_time"] == False:
            zerotime = self.current_peak.firsttime
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave - zerotime, curlims[1][0]))
        else:
            new_rect2.set_xy((self.current_peak.peaktimes[self.index_of_peak] - self.half_timeWave, curlims[1][0]))
        new_rect2.set_facecolor(self.mframe.plotsettings.peak_plot_colors['rect_color'])

        self.rectCellLength = self.axCellLength.add_patch(new_rect2)
        
        self.axCellLength.set_xlim(curlims[0])
        self.axCellLength.set_ylim(curlims[1])
        self.figCellLength.canvas.draw()

    def calcFlowMatrices(self):
        speed_segmentation2 = []
        # try:
        if self.literals["controller"].current_analysis.gtype == "Folder":
            global img_opencv
            files_grabbed = [x for x in os.listdir(self.literals["controller"].current_analysis.gpath) if os.path.isdir(x) == False and str(x).lower().endswith(img_opencv)]
            framelist = sorted(files_grabbed)
            files_grabbed_now = framelist[self.literals["controller"].selectedframes[self.current_peak.first]:(self.literals["controller"].selectedframes[self.current_peak.last+1])+1]
            files_grabbed_now = [self.literals["controller"].current_analysis.gpath + "/" + a for a in files_grabbed_now]

            for j in range(len(files_grabbed_now)-1):
                frame1 = cv2.imread(r'%s' %files_grabbed_now[0+j])
                frame2 = cv2.imread(r'%s' %files_grabbed_now[1+j])

                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.literals["controller"].current_analysis.pyr_scale, self.literals["controller"].current_analysis.levels, self.literals["controller"].current_analysis.winsize, self.literals["controller"].current_analysis.iterations, self.literals["controller"].current_analysis.poly_n, self.literals["controller"].current_analysis.poly_sigma, 0)

                U=flow[...,0]
                V=flow[...,1]

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)

                bsize, kdil, kero, ksco, borders = self.segmentation_dict[self.current_peak_index][j]

                mask, contourdimensions, largest_area_rect, largest_contour = self.mframe.get_contour(frame1, bsize, kdil, kero, ksco, borders)

                mag_segmented = np.ma.masked_where(mask, mag)
                
                meanval2 = np.abs(mag_segmented).mean() *  self.literals["controller"].current_peak.FPS * self.literals["controller"].current_peak.pixel_val
                meanval2 = float("{:.3f}".format(meanval2))

                speed_segmentation2.append(meanval2)
        elif self.literals["controller"].current_analysis.gtype == "Video":
            vc = cv2.VideoCapture(r'%s' %self.literals["controller"].current_analysis.gpath)
            count = self.literals["controller"].selectedframes[self.current_peak.first]
            vc.set(1, count-1)
            _, frame1 = vc.read()
            while(vc.isOpened() and count < (self.literals["controller"].selectedframes[self.current_peak.last]+1)):
                _, frame2 = vc.read()
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.literals["controller"].current_analysis.pyr_scale, self.literals["controller"].current_analysis.levels, self.literals["controller"].current_analysis.winsize, self.literals["controller"].current_analysis.iterations, self.literals["controller"].current_analysis.poly_n, self.literals["controller"].current_analysis.poly_sigma, 0)

                U=flow[...,0]
                V=flow[...,1]

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                #filter mags not in contour for given frame1
                bsize, kdil, kero, ksco, borders = self.segmentation_dict[self.current_peak_index][j]


                mask, contourdimensions, largest_area_rect, largest_contour = self.mframe.get_contour(frame1, bsize, kdil, kero, ksco, borders)

                mag = np.ma.masked_where(mask, mag)
                speed_segmentation2.append(np.mean(mag) * self.literals["controller"].current_peak.FPS * self.literals["controller"].current_peak.pixel_val)
                #calculate vectors and cluster by cossine similarity

                # self.controller.current_framelist.append(frame1)
                # self.controller.current_maglist.append(mag * self.controller.current_peak.FPS * self.controller.current_peak.pixel_val)
                # self.controller.current_anglist.append((U,V))

                frame1 = frame2.copy()
                count += 1
            vc.release()
        elif self.literals["controller"].current_analysis.gtype == "Tiff Directory" or self.literals["controller"].current_analysis.gtype == "CTiff":
            _, images = cv2.imreadmulti(r'%s' %self.literals["controller"].current_analysis.gpath, None, cv2.IMREAD_COLOR)
            images = images[self.literals["controller"].selectedframes[self.current_peak.first]:self.literals["controller"].selectedframes[(self.current_peak.last)+1]]
            for j in range(len(images)-1):
                frame1 = images[0+j]
                frame2 = images[1+j]
        
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                prvs2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, self.literals["controller"].current_analysis.pyr_scale, self.literals["controller"].current_analysis.levels, self.literals["controller"].current_analysis.winsize, self.literals["controller"].current_analysis.iterations, self.literals["controller"].current_analysis.poly_n, self.literals["controller"].current_analysis.poly_sigma, 0)

                U=flow[...,0]
                V=flow[...,1]

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                #filter mags not in contour for given frame1
                bsize, kdil, kero, ksco, borders = self.segmentation_dict[self.current_peak_index][j]

                mask, contourdimensions, largest_area_rect, largest_contour = self.mframe.get_contour(frame1, bsize, kdil, kero, ksco, borders)

                mag = np.ma.masked_where(mask, mag)
                speed_segmentation2.append(np.mean(mag) * self.literals["controller"].current_peak.FPS * self.literals["controller"].current_peak.pixel_val)
        # except Exception as e:
        #     messagebox.showerror("Error", "Could not retrieve frames\n" + str(e))
        return speed_segmentation2

    def calculate_speedComp_current(self):
        #calculate variation of length
        speed_segmentation2 = self.calcFlowMatrices()

        #add to length plots
        self.axSpeedComp.clear()
        self.mainplotartistSpeedComp = None
        self.mainplotartistSpeedComp2 = None
        self.mainplotartistSpeedComp3 = None
        self.mainplotartistSpeedComp4 = None
        self.axbaselineSpeedComp = None
        self.axgridSpeedComp = None
        self.axSpeedComp.set_xlabel("Time ("+self.literals["controller"].current_timescale+")")
        self.axSpeedComp.set_ylabel("Speed ("+self.literals["controller"].current_speedscale+")")
        
        if self.mframe.plotsettings.plotline_opts["absolute_time"] == True:
            self.mainplotartistSpeedComp = self.axSpeedComp.plot(self.peaks[self.current_peak_index].peaktimes, self.peaks[self.current_peak_index].peakdata, color=self.mframe.plotsettings.peak_plot_colors["main"])
            self.mainplotartistSpeedComp2 = self.axSpeedComp.plot(self.peaks[self.current_peak_index].peaktimes, speed_segmentation2[1:], color=self.mframe.plotsettings.peak_plot_colors["fft"])
        else:
            zerotime = self.peaks[self.current_peak_index].firsttime
            self.mainplotartistSpeedComp = self.axSpeedComp.plot([ttime - zerotime for ttime in self.peaks[self.current_peak_index].peaktimes], self.peaks[self.current_peak_index].peakdata, color=self.mframe.plotsettings.peak_plot_colors["main"])
            self.mainplotartistSpeedComp2 = self.axSpeedComp.plot([ttime - zerotime for ttime in self.peaks[self.current_peak_index].peaktimes[1:]], speed_segmentation2[1:], color=self.mframe.plotsettings.peak_plot_colors["fft"])
                
        curlims = (self.axSpeedComp.get_xlim(), self.axSpeedComp.get_ylim())

        if self.axbaselineSpeedComp != None:
            self.axbaselineSpeedComp.remove()
            self.axbaselineSpeedComp = None
        if self.mframe.plotsettings.plotline_opts["zero"] == True:
            self.axbaselineSpeedComp = self.axSpeedComp.axhline(y=0.0, color=self.mframe.plotsettings.plotline_opts["zero_color"], linestyle='-')

        if self.axgridSpeedComp != None:
            self.axgridSpeedComp.remove()
            self.axgridSpeedComp = None
        if self.mframe.plotsettings.plotline_opts["grid"] == True:
            self.axgridSpeedComp = self.axSpeedComp.grid(linestyle="-", color=self.mframe.plotsettings.plotline_opts["grid_color"], alpha=0.5)
        else:
            self.axSpeedComp.grid(False)
        
        self.axSpeedComp.set_xlim(curlims[0])
        self.axSpeedComp.set_ylim(curlims[1])
        self.figSpeedComp.canvas.draw()
    
    def calculate_length_exportAll2(self):
        self.mframe.controller.controller.showwd(parent2=self.mframe)

        topmerge_row= {
            0:[["Absolute", 1, 2] , ["Normalized", 3,4]],
            1:[["Absolute", 1, 2] , ["Normalized", 3,4]]
        }
        cols = ["Cell shortening (µm)", "Cell shortening (%)","Cell shortening (µm) ","Cell shortening (%) "]
        lengthavgcols = ["Avg. " + a for a in cols]
        rows = [[] for col in cols]
        avg_array = [0.0 for col in cols]
        #get selected groups
        valid_pis = []
        for ii2, ivar in enumerate(self.checkedgroups):
            if ivar.get() == 1:
                valid_pis.append(ii2)
        #calculate rows for each group
        all_names_data = []
        all_cols_data = []
        all_rows_data = []
        for ipi, valid_pi in enumerate(valid_pis):
            peak_obj = self.peaks[valid_pi]
            list_lengths = []
            width_lengths = []
            for e_index in range(len(self.peaks[valid_pi].peaktimes)):
                self.set_image_list(pi=valid_pi)
                h_value, w_value, c_area = self.drawSegmentedImage(pi=valid_pi, ii=e_index, refreshFrames=False)
                list_lengths.append([h_value, w_value, c_area])
                width_lengths.append(w_value)

            fpval = width_lengths[0]
            minimum_width = width_lengths[peak_obj.peaktimes.index(peak_obj.thirdtime)]

            absolute_widths = [float("{:.3f}".format(a)) for a in width_lengths]
            absolute_widths_perc = [float("{:.3f}".format(a / fpval * 100.0)) for a in absolute_widths]
            normalized_widths = [float("{:.3f}".format(a - fpval)) for a in absolute_widths]
            normalized_widths_perc = [float("{:.3f}".format((a / fpval * 100.0) - 100.0)) for a in absolute_widths]
            rows_data = [absolute_widths,absolute_widths_perc,normalized_widths,normalized_widths_perc]

            all_names_data.append("Wave " + str(valid_pi+1))
            all_cols_data.append(cols.copy())
            all_rows_data.append(rows_data)
            topmerge_row[ipi+2] = [["Absolute", 1, 2] , ["Normalized", 3,4]]

            absolute_minimum = minimum_width
            absolute_minimum_perc = (minimum_width / fpval * 100.0)
            normalized_minimum = minimum_width - fpval
            normalized_minimum_perc =  (minimum_width / fpval * 100.0) - 100.0
            rows[0].append(float("{:.3f}".format(absolute_minimum)))
            rows[1].append(float("{:.3f}".format(absolute_minimum_perc)))
            rows[2].append(float("{:.3f}".format(normalized_minimum)))
            rows[3].append(float("{:.3f}".format(normalized_minimum_perc)))
            avg_array[0] += float("{:.3f}".format(absolute_minimum / float(len(valid_pis))))
            avg_array[1] += float("{:.3f}".format(absolute_minimum_perc / float(len(valid_pis))))
            avg_array[2] += float("{:.3f}".format(normalized_minimum / float(len(valid_pis))))
            avg_array[3] += float("{:.3f}".format(normalized_minimum_perc / float(len(valid_pis))))
        #send data to average array
        avg_array = [[ei] for ei in avg_array]
        final_headers = [lengthavgcols ,cols]
        final_data = [avg_array, rows]
        final_names = ["Avg. Cell Measures", "Cell Measures"]

        final_headers.extend(all_cols_data)
        final_data.extend(all_rows_data)
        final_names.extend(all_names_data)

        self.set_image_list()
        self.mframe.controller.controller.cancelwd()
        SaveTableDialog(self, title='Save Table', literals=[
            ("headers", final_headers),
            ("data", final_data),
            ("sheetnames", final_names),
            ("data_t", "multiple"),
            ("mergetop", topmerge_row)
            ], parent2=self.mframe)

    def calculate_length_exportAll(self):
        self.mframe.controller.controller.showwd(parent2=self.mframe)
        #one sheet containing each group's contraction and relaxation amplitude
        #one sheet with contraction and relaxation amplitude averages

        # cols = ["Cell Contraction Amplitude", "Cell Relaxation Amplitude"]
        # cols = ["Cell Contraction Amplitude"]
        cols = ["Cell Contraction Shortening"]
        
        suffix_cols = " (%)"
        if self.cell_plottype.get() == 0:
            suffix_cols = " (µm)"
        if self.cell_plottype.get() == 2:
            suffix_cols = " (µm²)"
        if self.cell_plottype.get() == 3:
            suffix_cols = " (% - Area)"
        cols = [a + suffix_cols for a in cols]
        lengthavgcols = ["Avg." + a for a in cols]

        rows = [[] for col in cols]
        avg_array = [0.0 for col in cols]

        times_to_rows = []
        times_to_cols = []
        wave_names = []
        valid_pis = []

        xlbl = ""
        if self.cell_plottype.get() == 0:
            xlbl = "Cell length (µm)"
        if self.cell_plottype.get() == 1:
            xlbl = "Cell length (%)"
        if self.cell_plottype.get() == 2:
            xlbl = "Cell area (µm²)"
        if self.cell_plottype.get() == 3:
            xlbl = "Cell area (%)"
        
        for ii2, ivar in enumerate(self.checkedgroups):
            if ivar.get() == 1:
                valid_pis.append(ii2)
                wave_names.append("Wave " + str(ii2+1))
                # times.append([])
                times_to_rows.append([[], []])
                times_to_cols.append(["Time ("+self.literals["controller"].current_timescale+")", xlbl])
        #check method for calculating lengths
        single_ref = 1
        # if self.cell_length_method.get() == 1:
        #     #ask for single or multiple references
        #     MsgBox = CustomYesNo(self, title='Set all Waves reference as selected Wave first image?', parent2=self.mframe)
        #     if MsgBox.result == True:
        #         single_ref = 1
        #     else:
        #         single_ref = 0
        for ipi, valid_pi in enumerate(valid_pis):
            contraction_length_var = None
            relaxation_length_var = None
            peak_obj = self.peaks[valid_pi]
            if True:
            # if self.cell_length_method.get() == 0:
                list_lengths = []
                width_lengths = []
                area_lengths = []
                for e_index in range(len(self.peaks[valid_pi].peaktimes)):
                # for e_index in range(self.peaks[valid_pi].first, self.peaks[valid_pi].last+1):
                    # print("e_index")
                    # print(e_index)
                    self.set_image_list(pi=valid_pi)
                    h_value, w_value, c_area = self.drawSegmentedImage(pi=valid_pi, ii=e_index, refreshFrames=False)
                    list_lengths.append([h_value, w_value, c_area])
                    width_lengths.append(w_value)
                    area_lengths.append(c_area)
                len_ind = 1
                if self.cell_plottype.get() == 2 or self.cell_plottype.get() == 3:
                    len_ind = 2
                contraction_length_var = np.abs(list_lengths[peak_obj.peaktimes.index(peak_obj.firsttime)][len_ind] - list_lengths[peak_obj.peaktimes.index(peak_obj.thirdtime)][len_ind])
                relaxation_length_var = np.abs(list_lengths[peak_obj.peaktimes.index(peak_obj.thirdtime)][len_ind] - list_lengths[peak_obj.peaktimes.index(peak_obj.fifthtime)][len_ind])
                if self.cell_percdiff.get() == 1:
                    contraction_length_var = list_lengths[peak_obj.peaktimes.index(peak_obj.firsttime)][len_ind] / contraction_length_var
                    contraction_length_var *= 100
                    relaxation_length_var = list_lengths[peak_obj.peaktimes.index(peak_obj.firsttime)][len_ind] / relaxation_length_var
                    relaxation_length_var *= 100
                time_var = peak_obj.peaktimes
                if self.cell_plottype.get() == 0:
                    fpval = width_lengths[0]
                    if self.cell_calctype.get() == 0:
                        width_lengths = [a for a in width_lengths]
                    elif self.cell_calctype.get() == 1:
                        width_lengths = [a - fpval for a in width_lengths]
                    elif self.cell_calctype.get() == 2:
                        o_width_lengths = [0.0]
                        width_lengths = [width_lengths[i] - width_lengths[i-1] for i in range(1, len(width_lengths))]
                        o_width_lengths.extend(width_lengths)
                        width_lengths = o_width_lengths.copy()
                    elif self.cell_calctype.get() == 3:
                        n_width_lengths = []
                        for a in width_lengths:
                            abs_val = np.abs(a - fpval)
                            if fpval > a:
                                abs_val *= -1.0
                            n_width_lengths.append(abs_val)
                        width_lengths = n_width_lengths.copy()
                if self.cell_plottype.get() == 1:
                    fpval = width_lengths[0]
                    if self.cell_calctype.get() == 0:
                        width_lengths = [(a / fpval) * 100.0 for a in width_lengths]
                    elif self.cell_calctype.get() == 1:
                        width_lengths = [((a / fpval) * 100.0) - 100.0 for a in width_lengths]
                    elif self.cell_calctype.get() == 2:
                        o_width_lengths = [0.0]
                        n_width_lengths = [(width_lengths[i] / fpval) - (width_lengths[i-1] / fpval) for i in range(1, len(width_lengths))]
                        o_width_lengths.extend(n_width_lengths)
                        width_lengths = o_width_lengths.copy()
                    elif self.cell_calctype.get() == 3:
                        n_width_lengths = []
                        for a in width_lengths:
                            abs_val = np.abs(((a / fpval) * 100.0) - 100.0)
                            if ((a / fpval) * 100.0) > 100.0 :
                                abs_val *= -1.0
                            n_width_lengths.append(abs_val)
                        width_lengths = n_width_lengths.copy()
                if self.cell_plottype.get() == 2:
                    fpval = area_lengths[0]
                    if self.cell_calctype.get() == 0:
                        width_lengths = [a for a in area_lengths]
                    elif self.cell_calctype.get() == 1:
                        width_lengths = [a - fpval for a in area_lengths]
                    elif self.cell_calctype.get() == 2:
                        width_lengths = [0.0]
                        n_width_lengths = [area_lengths[i] - area_lengths[i-1] for i in range(1, len(area_lengths))]
                        width_lengths.extend(n_width_lengths)
                    elif self.cell_calctype.get() == 3:
                        n_width_lengths = []
                        for a in area_lengths:
                            abs_val = np.abs(a - fpval)
                            if fpval > a:
                                abs_val *= -1.0
                            n_width_lengths.append(abs_val)
                        width_lengths = n_width_lengths.copy()
                if self.cell_plottype.get() == 3:
                    fpval = area_lengths[0]
                    if self.cell_calctype.get() == 0:
                        width_lengths = [(a / fpval) * 100.0 for a in area_lengths]
                    elif self.cell_calctype.get() == 1:
                        width_lengths = [((a / fpval) * 100.0) - 100.0 for a in area_lengths]
                    elif self.cell_calctype.get() == 2:
                        o_width_lengths = [0.0]
                        n_width_lengths = [(area_lengths[i] / fpval) - (area_lengths[i-1] / fpval) for i in range(1, len(area_lengths))]
                        o_width_lengths.extend(n_width_lengths)
                        width_lengths = o_width_lengths.copy()
                    elif self.cell_calctype.get() == 3:
                        n_width_lengths = []
                        for a in area_lengths:
                            abs_val = np.abs(((a / fpval) * 100.0) - 100.0)
                            if ((a / fpval) * 100.0) > 100.0 :
                                abs_val *= -1.0
                            n_width_lengths.append(abs_val)
                        width_lengths = n_width_lengths.copy()
                times_to_rows[ipi][0] = time_var.copy()
                times_to_rows[ipi][1] = width_lengths.copy()
            # elif single_ref == 1:
            #     list_lengths = []
            #     # h_value, w_value, c_area = drawSegmentedImage(self, pi=reference_pi, ii=self.peaks[reference_pi].first, refreshFrames=False)
            #     h_value, w_value, c_area = self.drawSegmentedImage(pi=None, ii=0, refreshFrames=False)
            #     to_iterate = peak_obj.fulldata[peak_obj.first:peak_obj.min+1]
            #     integral_contraction = float(np.trapz([a for a in peak_obj.fulldata[peak_obj.first:peak_obj.min+1]]))
            #     integral_relaxation = float(np.trapz([a for a in peak_obj.fulldata[peak_obj.min:peak_obj.last+1]]))
            #     contraction_length_var = np.abs(float(c_area) - integral_contraction)
            #     relaxation_length_var = np.abs(float(c_area) - integral_contraction + integral_relaxation)
            #     if self.cell_percdiff.get() == 1:
            #         contraction_length_var = float(c_area) / contraction_length_var
            #         contraction_length_var *= 100
            #         relaxation_length_var = float(c_area) / relaxation_length_var
            #         relaxation_length_var *= 100
            #         contraction_length_var = w_value * contraction_length_var
            # else:
            #     list_lengths = []
            #     self.set_image_list(pi=valid_pi)
            #     h_value, w_value, c_area = self.drawSegmentedImage(pi=valid_pi, ii=0, refreshFrames=False)
            #     integral_contraction = float(np.trapz([a for a in peak_obj.fulldata[peak_obj.first:peak_obj.min+1]]))
            #     integral_relaxation = float(np.trapz([a for a in peak_obj.fulldata[peak_obj.min:peak_obj.last+1]]))
            #     contraction_length_var = float(c_area) - integral_contraction
            #     relaxation_length_var = float(c_area) - integral_contraction + integral_relaxation
            #     if self.cell_percdiff.get() == 1:
            #         contraction_length_var = float(c_area) / contraction_length_var
            #         contraction_length_var *= 100
            #         relaxation_length_var = float(c_area) / relaxation_length_var
            #         relaxation_length_var *= 100
            rows[0].append(float("{:.3f}".format(contraction_length_var)))
            # rows[1].append(float("{:.3f}".format(relaxation_length_var)))
            avg_array[0] += float("{:.3f}".format(contraction_length_var/ len(valid_pis)))
            # avg_array[1] += float("{:.3f}".format(relaxation_length_var/ len(valid_pis))) 
        avg_array = [[ei] for ei in avg_array]
        self.set_image_list()
        final_headers = [lengthavgcols ,cols]
        final_headers.extend(times_to_cols)
        final_data = [avg_array, rows]
        final_data.extend(times_to_rows)
        final_names = ["Avg. Cell Measures","Cell Measures"]
        final_names.extend(wave_names)
        self.mframe.controller.controller.cancelwd()
        SaveTableDialog(self, title='Save Table', literals=[
            ("headers", final_headers),
            ("data", final_data),
            ("sheetnames", final_names),
            # ("headers", [lengthavgcols ,cols]),
            # ("data", [avg_array, rows]),
            # ("sheetnames", ["Avg. Cell Length","Cell Length"]),
            ("data_t", "multiple")
            ], parent2=self.mframe)
        # try:
        #     d = SaveTableDialog(self, title='Save Table', literals=[
        #         ("headers", [self.ax2.get_xlabel(), self.ax2.get_ylabel()]),
        #         ("data", [self.mainplotartist[0].get_xdata(), self.mainplotartist[0].get_ydata()]),
        #         ("data_t", "single")
        #         ])
        # except TypeError:
        #     messagebox.showerror("Error", "No Data in plot")

    def validate(self):
        # print("class CellLengthDialog def validate start")
        #segmentation checking for disturbances
        self.valid = True
        return 1

    def apply(self):
        # print("class CellLengthDialog def apply start")
        self.result = True

class WaitDialogProgress(tkDialog.DialogBlockNonGrab):
# class WaitDialogProgress(tkDialog.DialogNonBlock):

    def body(self, master):
        ttk.Label(master, text="Please wait until processing is done...").grid(row=0,column=0, columnspan=3)

    def validate(self):
        # print("class AboutDialog def validate start")
        self.valid = False
        return 0

    def apply(self):
        # print("class AboutDialog def apply start")
        pass

class QuiverJetMaximize(tkDialog.DialogNonBlockMax):

    def body(self, master):
        # print("class QuiverJetMaximize def body creation")
        self.master_window = master

        # self.figmax = plt.figure(figsize=(4, 3), dpi=100 ,facecolor=master.cget('bg'), edgecolor="None")
        # self.mbg = master.cget('bg')
        self.figmax = plt.figure(figsize=(4, 3), dpi=300 ,facecolor="black", edgecolor="None")
        
        self.gsmax = gridspec.GridSpec(1, 1, height_ratios=[5], hspace=0.2)
        
        self.gs_nobordermax = gridspec.GridSpec(1, 1, height_ratios=[5], left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.frame_canvasmax = tk.Frame(master)
        
        self.canvasmax = FigureCanvasTkAgg(self.figmax, master=self.frame_canvasmax)
        self.canvasmax.get_tk_widget().configure(background='black',  highlightcolor='black', highlightbackground='black')
        
        self.axmax = self.figmax.add_subplot(self.gs_nobordermax[0])

        self.canvasmax.draw()
        
        self.canvasmax.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # self.canvasmax.bind("<Configure>", self.resize_max)
        
        self.frame_canvasmax.grid(row=0, column=0, sticky=tk.NSEW)

        # self.frame_canvasmax.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # self.packbtns = False
        # print("class QuiverJetMaximize def body done")

    def set_figure(self, figx, figy):
        self.figmax.set_size_inches(figx/300, figy/300, forward=True)
        self.figmax.axes[0].set_position(self.gs_nobordermax[0].get_position(self.figmax))
        self.canvasmax.get_tk_widget().config(width=figx)
        self.canvasmax.get_tk_widget().config(height=figy)

        self.figmax.canvas.draw()
        self.canvasmax.draw()

    # def set_figure(self, buf, figx, figy):
    #     self.fig2 = pickle.load(buf)
    #     self.fig2.set_size_inches(figx/100, figy/100, forward=True)

    #     self.canvasmax.get_tk_widget().destroy()
    #     self.canvasmax = FigureCanvasTkAgg(self.fig2, master=self.frame_canvasmax)
    #     self.canvasmax.get_tk_widget().configure(background='black',  highlightcolor='black', highlightbackground='black')
        
    #     # prevmargins = plt.margins()
    #     # fig2.axes[0].change_geometry(1,1,1)
    #     self.canvasmax.draw()
    #     self.canvasmax.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    #     # self.canvasmax.bind("<Configure>", self.resize_max)
    #     # self.fig2.axes[0].get_legend().remove()
    #     self.fig2.axes[0].set_position(self.gs_nobordermax[0].get_position(self.fig2))
    #     # self.fig2.axes[0].set_facecolor('black')
    #     # self.fig2.gca().set_axis_off()
    #     # ax = self.fig2.axes[0]
    #     # ax.set_axis_off()

    #     # ax.add_artist(ax.patch)
    #     # ax.patch.set_zorder(-1)

    #     # extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

    #     # plt.margins(0,0)

    #     # self.fig2.gca().xaxis.set_major_locator(plt.NullLocator())
    #     # self.fig2.gca().yaxis.set_major_locator(plt.NullLocator())

    #     # plt.axis('off')
    #     # self.fig2.tight_layout(pad=0)

    #     self.fig2.canvas.draw()
    #     self.canvasmax.draw()

    #     # plt.margins(prevmargins[0], prevmargins[1])
    #     # plt.axis('on')

    def resize_max(self):
        pass

    def validate(self):
        # print("class QuiverJetMaximize def validate start")
        self.valid = True
        return 1

    def apply(self):
        # print("class QuiverJetMaximize def apply start")
        self.literals["updatable_frame"].clear_maximize()
        pass

class QuiverJetSettings(tkDialog.DialogNonBlock):

    def body(self, master):
        # print("class QuiverJetSettings def body creation")
        ttk.Label(master, text='Vector Spacing:', font=('Helvetica', 14) ).grid(row=0, column=0, rowspan=1, columnspan=2)
        #list_features = ["current_windowX", "current_windowY", "blur_size", "kernel_dilation", "kernel_erosion", "kernel_smoothing_contours", "border_thickness"]
        rown = 1

        ttk.Label(master,text=self.literals["current_windowX"][0]).grid(row=rown, column=0)
        self.current_windowX_spin = tk.Spinbox(master, from_=self.literals["config"]["current_windowX"][0], to=self.literals["config"]["current_windowX"][1], increment=1, width=10, command=lambda: self.up_frame(self.current_windowX_spin, "current_windowX"))
        self.current_windowX_spin.thistype = "current_windowX" 
        self.current_windowX_spin.grid(row=rown, column=1)
        self.current_windowX_spin.delete(0,"end")
        self.current_windowX_spin.insert(0,self.literals["current_windowX"][1])
        self.current_windowX_spin.bind('<Return>', lambda *args: self.up_frame(self.current_windowX_spin, "current_windowX"))

        rown += 1
        ttk.Label(master,text=self.literals["current_windowY"][0]).grid(row=rown, column=0)
        self.current_windowY_spin = tk.Spinbox(master, from_=self.literals["config"]["current_windowY"][0], to=self.literals["config"]["current_windowY"][1], increment=1, width=10, command=lambda: self.up_frame(self.current_windowY_spin, "current_windowY"))
        self.current_windowY_spin.thistype = "current_windowY" 
        self.current_windowY_spin.grid(row=rown, column=1)
        self.current_windowY_spin.delete(0,"end")
        self.current_windowY_spin.insert(0,self.literals["current_windowY"][1])
        self.current_windowY_spin.bind('<Return>', lambda *args: self.up_frame(self.current_windowY_spin, "current_windowY"))

        
        rown += 1
        
        sep = ttk.Separator(master, orient='horizontal')
        sep.grid(row=rown, column=0, columnspan=2, sticky="ew", ipadx=20, ipady=1)
        rown += 1

        ttk.Label(master, text='Image Filtering:',  font=('Helvetica', 14)  ).grid(row=rown, column=0, rowspan=1, columnspan=2)
        rown += 1

        ttk.Label(master,text=self.literals["blur_size"][0]).grid(row=rown, column=0)
        self.blur_size_spin = tk.Spinbox(master, from_=self.literals["config"]["blur_size"][0], to=self.literals["config"]["blur_size"][1], increment=2, width=10, command=lambda: self.up_frame(self.blur_size_spin, "blur_size"))
        self.blur_size_spin.thistype = "blur_size" 
        self.blur_size_spin.grid(row=rown, column=1)
        self.blur_size_spin.delete(0,"end")
        self.blur_size_spin.insert(0,self.literals["blur_size"][1])
        self.blur_size_spin.bind('<Return>', lambda *args: self.up_frame(self.blur_size_spin, "blur_size"))

        rown += 1
        ttk.Label(master,text=self.literals["kernel_dilation"][0]).grid(row=rown, column=0)
        self.kernel_dilation_spin = tk.Spinbox(master, from_=self.literals["config"]["kernel_dilation"][0], to=self.literals["config"]["kernel_dilation"][1], increment=2, width=10, command=lambda: self.up_frame(self.kernel_dilation_spin, "kernel_dilation"))
        self.kernel_dilation_spin.thistype = "kernel_dilation" 
        self.kernel_dilation_spin.grid(row=rown, column=1)
        self.kernel_dilation_spin.delete(0,"end")
        self.kernel_dilation_spin.insert(0,self.literals["kernel_dilation"][1])
        self.kernel_dilation_spin.bind('<Return>', lambda *args: self.up_frame(self.kernel_dilation_spin, "kernel_dilation"))
       
        rown += 1
        ttk.Label(master,text=self.literals["kernel_erosion"][0]).grid(row=rown, column=0)
        self.kernel_erosion_spin = tk.Spinbox(master, from_=self.literals["config"]["kernel_erosion"][0], to=self.literals["config"]["kernel_erosion"][1], increment=2, width=10, command=lambda: self.up_frame(self.kernel_erosion_spin, "kernel_erosion"))
        self.kernel_erosion_spin.thistype = "kernel_erosion" 
        self.kernel_erosion_spin.grid(row=rown, column=1)
        self.kernel_erosion_spin.delete(0,"end")
        self.kernel_erosion_spin.insert(0,self.literals["kernel_erosion"][1])
        self.kernel_erosion_spin.bind('<Return>', lambda *args: self.up_frame(self.kernel_erosion_spin, "kernel_erosion"))

        rown += 1
        ttk.Label(master,text=self.literals["kernel_smoothing_contours"][0]).grid(row=rown, column=0)
        self.kernel_smoothing_contours_spin = tk.Spinbox(master, from_=self.literals["config"]["kernel_smoothing_contours"][0], to=self.literals["config"]["kernel_smoothing_contours"][1], increment=1, width=10, command=lambda: self.up_frame(self.kernel_smoothing_contours_spin, "kernel_smoothing_contours"))
        self.kernel_smoothing_contours_spin.thistype = "kernel_smoothing_contours" 
        self.kernel_smoothing_contours_spin.grid(row=rown, column=1)
        self.kernel_smoothing_contours_spin.delete(0,"end")
        self.kernel_smoothing_contours_spin.insert(0,self.literals["kernel_smoothing_contours"][1])
        self.kernel_smoothing_contours_spin.bind('<Return>', lambda *args: self.up_frame(self.kernel_smoothing_contours_spin, "kernel_smoothing_contours"))

        rown += 1
        ttk.Label(master,text=self.literals["border_thickness"][0]).grid(row=rown, column=0)
        self.border_thickness_spin = tk.Spinbox(master, from_=self.literals["config"]["border_thickness"][0], to=self.literals["config"]["border_thickness"][1], increment=1, width=10, command=lambda: self.up_frame(self.border_thickness_spin, "border_thickness"))
        self.border_thickness_spin.thistype = "border_thickness"
        self.border_thickness_spin.grid(row=rown, column=1)
        self.border_thickness_spin.delete(0,"end")
        self.border_thickness_spin.insert(0,self.literals["border_thickness"][1])
        self.border_thickness_spin.bind('<Return>', lambda *args: self.up_frame(self.border_thickness_spin, "border_thickness"))

        rown += 1
        ttk.Label(master,text=self.literals["jetalpha"][0]).grid(row=rown, column=0)
        self.jetalpha_spin = tk.Spinbox(master, from_=self.literals["config"]["jetalpha"][0], to=self.literals["config"]["jetalpha"][1], increment=0.1, width=10, command=lambda: self.up_frame(self.jetalpha_spin, "jetalpha"))
        self.jetalpha_spin.thistype = "jetalpha"
        self.jetalpha_spin.grid(row=rown, column=1)
        self.jetalpha_spin.delete(0,"end")
        self.jetalpha_spin.insert(0,self.literals["jetalpha"][1])
        self.jetalpha_spin.bind('<Return>', lambda *args: self.up_frame(self.jetalpha_spin, "jetalpha"))
        rown += 1

        ttk.Label(master,text=self.literals["quiveralpha"][0]).grid(row=rown, column=0)
        self.quiveralpha_spin = tk.Spinbox(master, from_=self.literals["config"]["quiveralpha"][0], to=self.literals["config"]["quiveralpha"][1], increment=0.1, width=10, command=lambda: self.up_frame(self.quiveralpha_spin, "quiveralpha"))
        self.quiveralpha_spin.thistype = "quiveralpha"
        self.quiveralpha_spin.grid(row=rown, column=1)
        self.quiveralpha_spin.delete(0,"end")
        self.quiveralpha_spin.insert(0,self.literals["quiveralpha"][1])
        self.quiveralpha_spin.bind('<Return>', lambda *args: self.up_frame(self.quiveralpha_spin, "quiveralpha"))
        rown += 1
        
        sep2 = ttk.Separator(master, orient='horizontal')
        sep2.grid(row=rown, column=0, columnspan=2, sticky="ew", ipadx=20, ipady=1)
        rown += 1

        ttk.Label(master, text='Scale Normalization:', font=('Helvetica', 14) ).grid(row=rown, column=0, rowspan=1, columnspan=2)
        rown += 1

        ttk.Label(master,text=self.literals["minscale"][0] + " (μm/s)").grid(row=rown, column=0)
        self.minscale_spin = tk.Spinbox(master, from_=self.literals["config"]["minscale"][0], to=self.literals["config"]["minscale"][1], increment=1, width=10, command=lambda: self.up_frame(self.minscale_spin, "minscale"))
        self.minscale_spin.thistype = "minscale"
        self.minscale_spin.grid(row=rown, column=1)
        self.minscale_spin.delete(0,"end")
        self.minscale_spin.insert(0,self.literals["minscale"][1])
        self.minscale_spin.bind('<Return>', lambda *args: self.up_frame(self.minscale_spin, "minscale"))

        rown += 1
        ttk.Label(master,text=self.literals["maxscale"][0] + " (μm/s)").grid(row=rown, column=0)
        self.maxscale_spin = tk.Spinbox(master, from_=self.literals["config"]["maxscale"][0], to=self.literals["config"]["maxscale"][1], increment=1, width=10, command=lambda: self.up_frame(self.maxscale_spin, "maxscale"))
        self.maxscale_spin.thistype = "maxscale"
        self.maxscale_spin.grid(row=rown, column=1)
        self.maxscale_spin.delete(0,"end")
        self.maxscale_spin.insert(0,self.literals["maxscale"][1])
        self.maxscale_spin.bind('<Return>', lambda *args: self.up_frame(self.maxscale_spin, "maxscale"))
        
        rown += 1
        mval = 0
        if self.literals["plotmax"] == True:
            mval = 1
        self.checkbutton_max_val = tk.IntVar(value=mval)
        self.checkbutton_max = ttk.Checkbutton(master, text = "Add Scale Max. to legend:", variable = self.checkbutton_max_val, \
                         onvalue = 1, offvalue = 0, command=lambda: self.up_frame(self.checkbutton_min, "plotmax"))
        self.checkbutton_max.thistype = "plotmax"
        self.checkbutton_max.grid(row=rown, column=0, columnspan=2)
        
        rown += 1
        ttk.Label(master,text=self.literals["defminscale"][0] + " (μm/s)").grid(row=rown, column=0)
        self.defminscale_spin = tk.Spinbox(master, from_=self.literals["config"]["defminscale"][0], to=self.literals["config"]["defminscale"][1], increment=1, width=10, command=lambda: self.up_frame(self.defminscale_spin, "defminscale"))
        self.defminscale_spin.thistype = "defminscale"
        self.defminscale_spin.grid(row=rown, column=1)
        self.defminscale_spin.delete(0,"end")
        self.defminscale_spin.insert(0,self.literals["defminscale"][1])
        self.defminscale_spin.bind('<Return>', lambda *args: self.up_frame(self.defminscale_spin, "defminscale"))

        rown += 1
        mival = 0
        if self.literals["plotmin"] == True:
            mival = 1
        self.checkbutton_min_val = tk.IntVar(value=mival)
        self.checkbutton_min = ttk.Checkbutton(master, text = "Add Scale Min. to legend:", variable = self.checkbutton_min_val, \
                         onvalue = 1, offvalue = 0, command=lambda: self.up_frame(self.checkbutton_min, "plotmin"))
        self.checkbutton_min.thistype = "plotmin"
        self.checkbutton_min.grid(row=rown, column=0, columnspan=2)

        # print("class QuiverJetSettings def body creation done")
        
        # tk.Label(master, text=literals[k][0]).grid(row=rown, column=0)
        # tk.Spinbox(master, from_=self.literals["config"][k][0], to=self.literals["config"][k][0], increment=1, width=10).grid(row=rown, column=1)

    def up_frame(self, current, event=None):
        # print("class QuiverJetSettings def up_frame")
        if event != None:
            # print("class QuiverJetSettings def up_frame event")
            # print(event)
            valid = self.validate()
            if valid == True:
                if event not in ["jetalpha", "quiveralpha", "plotmax", "plotmin", "minscale", "maxscale"]:
                    self.literals["updatable_frame"].update_config(event,int(current.get().replace(",", ".")))
                elif event not in ["jetalpha", "quiveralpha", "minscale", "maxscale"]:
                    if event == "plotmin":
                        self.literals["updatable_frame"].update_config(event,bool(self.checkbutton_min_val.get()))
                    elif event == "plotmax":
                        self.literals["updatable_frame"].update_config(event,bool(self.checkbutton_max_val.get()))
                else:
                    self.literals["updatable_frame"].update_config(event,float(current.get().replace(",", ".")))
        # self.literals["updatable_frame"].update_frame()
        # self.literals["updatable_frame"].update()
        pass

    def validate(self):
        # print("class QuiverJetSettings def validate start")
        try:
            current_windowX_spin_val = int(self.current_windowX_spin.get().replace(",", "."))
            if current_windowX_spin_val < self.literals["config"]["current_windowX"][0] or current_windowX_spin_val > self.literals["config"]["current_windowX"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0 
            current_windowY_spin_val = int(self.current_windowY_spin.get().replace(",", "."))
            if current_windowY_spin_val < self.literals["config"]["current_windowY"][0] or current_windowY_spin_val > self.literals["config"]["current_windowY"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            blur_size_spin_val = int(self.blur_size_spin.get().replace(",", "."))
            if blur_size_spin_val < self.literals["config"]["blur_size"][0] or blur_size_spin_val > self.literals["config"]["blur_size"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            if blur_size_spin_val % 2 == 0:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            kernel_dilation_spin_val = int(self.kernel_dilation_spin.get().replace(",", "."))
            if kernel_dilation_spin_val < self.literals["config"]["kernel_dilation"][0] or kernel_dilation_spin_val > self.literals["config"]["kernel_dilation"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0  
            if kernel_dilation_spin_val % 2 == 0:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0 
            kernel_erosion_spin_val = int(self.kernel_erosion_spin.get().replace(",", "."))
            if kernel_erosion_spin_val < self.literals["config"]["kernel_erosion"][0] or kernel_erosion_spin_val > self.literals["config"]["kernel_erosion"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0   
            if kernel_erosion_spin_val % 2 == 0:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0 
            kernel_smoothing_contours_spin_val = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            if kernel_smoothing_contours_spin_val < self.literals["config"]["kernel_smoothing_contours"][0] or kernel_smoothing_contours_spin_val > self.literals["config"]["kernel_smoothing_contours"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
                
            border_thickness_spin_val = int(self.border_thickness_spin.get().replace(",", "."))
            if border_thickness_spin_val < self.literals["config"]["border_thickness"][0] or border_thickness_spin_val > self.literals["config"]["border_thickness"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            minscale_val = float(self.minscale_spin.get().replace(",", "."))
            if minscale_val < self.literals["config"]["minscale"][0] or minscale_val > self.literals["config"]["minscale"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            defminscale_val = float(self.defminscale_spin.get().replace(",", "."))
            if defminscale_val < self.literals["config"]["defminscale"][0] or defminscale_val > self.literals["config"]["defminscale"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            maxscale_val = float(self.maxscale_spin.get().replace(",", "."))
            if maxscale_val < self.literals["config"]["maxscale"][0] or maxscale_val > self.literals["config"]["maxscale"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            if maxscale_val <= minscale_val or maxscale_val <= defminscale_val:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            jetalpha_val = float(self.jetalpha_spin.get().replace(",", "."))
            if jetalpha_val < self.literals["config"]["jetalpha"][0] or jetalpha_val > self.literals["config"]["jetalpha"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            quiveralpha_val = float(self.quiveralpha_spin.get().replace(",", "."))
            if quiveralpha_val < self.literals["config"]["quiveralpha"][0] or quiveralpha_val > self.literals["config"]["quiveralpha"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            #nothing to validate
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class QuiverJetSettings def apply start")
        #save configs
        self.result = None
        # print("self.valid")
        # print(self.valid)
        if self.valid == True:
            self.result = {}
            # print(self.result)
            self.result["current_windowX"] = int(self.current_windowX_spin.get().replace(",", "."))
            self.result["current_windowY"] = int(self.current_windowY_spin.get().replace(",", "."))
            self.result["blur_size"] = int(self.blur_size_spin.get().replace(",", "."))
            self.result["kernel_dilation"] = int(self.kernel_dilation_spin.get().replace(",", "."))
            self.result["kernel_erosion"] = int(self.kernel_erosion_spin.get().replace(",", "."))
            self.result["kernel_smoothing_contours"] = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            self.result["border_thickness"] = int(self.border_thickness_spin.get().replace(",", "."))
            self.result["minscale"] = float(self.minscale_spin.get().replace(",", "."))
            self.result["defminscale"] = float(self.defminscale_spin.get().replace(",", "."))
            self.result["maxscale"] = float(self.maxscale_spin.get().replace(",", "."))
            self.result["jetalpha"] = float(self.jetalpha_spin.get().replace(",", "."))
            self.result["quiveralpha"] = float(self.quiveralpha_spin.get().replace(",", "."))
            self.result["plotmax"] = bool(self.checkbutton_max_val.get())
            self.result["plotmin"] = bool(self.checkbutton_min_val.get())
            # print(self.result)
            self.literals["updatable_frame"].update_all_settings(self.result)
        return True

# class AdjustDeltaFFTDialog(tkDialog.Dialog):
class AdjustDeltaFFTDialog(tkDialog.DialogNonBlock):
    def body(self, master):
        rown = 0
        self.spindeltafftlbl = ttk.Label(master, text= 'FFT peak detection Delta:')
        self.spindeltafftlbl.grid(row=rown, column=0)
        rown += 1
        self.spindeltafft = tk.Spinbox(master, from_=-10000000000, to=10000000000, increment=0.5, width=10, command=self.updateplot)
        self.spindeltafft.grid(row=rown, column=0)
        self.spindeltafft.delete(0,"end")
        self.spindeltafft.insert(0,self.literals["delta_fft"])
        # self.spindeltafft.bind('<Return>', lambda *args: self.validate())
        self.spindeltafft.bind('<Return>', lambda *args: self.updateplot())
        rown += 1

        self.show_allcheckvar = tk.IntVar(value=1)
        if self.literals["updatable_frame"].plotFFTAll == False:
            self.show_allcheckvar.set(0)
        self.showall_check_ttk = ttk.Checkbutton(master, text = "Show all detected peaks", variable = self.show_allcheckvar, \
                         onvalue = 1, offvalue = 0, command=self.updateplot)
        self.showall_check_ttk.grid(row=rown, column=0)

    def validate(self):
        #nothing to validate
        fvalue = float(self.spindeltafft.get().replace(",", "."))
        if fvalue <  -10000000000 or fvalue > 10000000000:
            self.valid = False
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            return 0 
        self.valid = True
        return 1

    def updateplot(self, args=None):
        self.validate()
        if self.valid:
            self.result = float(self.spindeltafft.get().replace(",", "."))
            doReset = False
            if (self.literals["delta_fft"] != self.result):
                # self.literals["updatable_frame"].plotFFTSelection = 0
                self.literals["delta_fft"] = self.result
                doReset = True
            if (self.show_allcheckvar.get() == 0):
                self.literals["updatable_frame"].plotFFTAll = False
            else:
                self.literals["updatable_frame"].plotFFTAll = True
            self.literals["updatable_frame"].updateadjustfftdelta(self.result, reset=doReset)        
    
    def apply(self):
        self.result = float(self.spindeltafft.get().replace(",", "."))
        doReset = False
        if (self.literals["delta_fft"] != self.result):
            # self.literals["updatable_frame"].plotFFTSelection = 0
            doReset = True
        if (self.show_allcheckvar.get() == 0):
            self.literals["updatable_frame"].plotFFTAll = False
        else:
            self.literals["updatable_frame"].plotFFTAll = True
        self.literals["updatable_frame"].updateadjustfftdelta(self.result, reset=doReset, close=True)        

# class AdjustExponentialDialog(tkDialog.Dialog):
# class AdjustExponentialDialog(tkDialog.DialogNonBlock):
class AdjustWaveEndDialog(tkDialog.DialogNonBlock):
    def body(self, master):
        self.result = None
        self.smoothdict = {
            "always": "Always",
            "Always": "always",
            "never": "Never",
            "Never": "never",
            "noisecriteria": "Noisy areas",
            "Noisy areas": "noisecriteria"
        }
        expconfigs = self.literals["exponentialsettings"]
        if len(expconfigs) > 1:
            self.endnoisecriteria = expconfigs[0]
            self.smoothbeforeregression = expconfigs[1]
            self.noiseratio = expconfigs[2]
            self.local_minimum_check = expconfigs[3]
        else:
            self.endnoisecriteria = 0.9
            # self.smoothbeforeregression = "noisecriteria"
            self.smoothbeforeregression = "never"
            self.noiseratio = 1.0
            # self.local_minimum_check = True
            self.local_minimum_check = False

        rown = 0
        self.end_current_type = self.literals["end_current_type"]
        self.end_thres_val = self.literals["end_thres_val"]
        self.endtypechangeval = tk.IntVar()

        if self.end_current_type == "threshold":
            self.endtypechangeval.set(0)
        else:
            self.endtypechangeval.set(1)

        #create and add radio selection for detection type
        # self.radio1thres = ttk.Radiobutton(master, text = "Threshold of Relaxation Max", variable = self.endtypechangeval, value = 0, command=self.detection_update)
        # self.radio1thres.grid(row=rown, column=0)#, sticky=tk.W+tk.E+tk.N+tk.S)
        # rown += 1

        # self.spinmaxlbl2 = ttk.Label(master, text= 'Fraction of Relaxation Max:')
        # self.spinmaxlbl2.grid(row=rown, column=0)
        # rown += 1

        self.spinmaxbox2 = tk.Spinbox(master, from_=0, to=1.0, increment=0.05, width=10, command=self.updateplot)
        self.spinmaxbox2.grid(row=rown, column=0)
        self.spinmaxbox2.delete(0,"end")
        self.spinmaxbox2.insert(0,self.endnoisecriteria)
        self.spinmaxbox2.bind('<Return>', lambda *args: self.updateplot())
        #forget this man
        self.spinmaxbox2.grid_forget()
        # rown += 1


        # self.radio2sergio = ttk.Radiobutton(master, text = "Threshold of Exp. fit", variable = self.endtypechangeval, value = 1, command=self.detection_update)
        # self.radio2sergio.grid(row=rown, column=0)#, sticky=tk.W+tk.E+tk.N+tk.S)
        # rown += 1

        self.spinmaxlbl = ttk.Label(master, text= 'Fraction of Wave Max Area:')
        self.spinmaxlbl.grid(row=rown, column=0)
        rown += 1
        self.spinmaxbox = tk.Spinbox(master, from_=0, to=1.0, increment=0.05, width=10, command=self.updateplot)
        self.spinmaxbox.grid(row=rown, column=0)
        self.spinmaxbox.delete(0,"end")
        self.spinmaxbox.insert(0,self.endnoisecriteria)
        # self.spinmaxbox.bind('<Return>', lambda *args: self.validate())
        self.spinmaxbox.bind('<Return>', lambda *args: self.updateplot())
        rown += 1

        ttk.Label(master, text='Data smoothing before regression:').grid(row=rown, column=0)
        rown += 1
        self.formatvar = tk.StringVar(master)
        self.formatchoices = {'Always', 'Never', 'Noisy areas'}
        self.formatvar.set(self.smoothdict[self.smoothbeforeregression]) # set the default option
        self.smoothmenu = ttk.OptionMenu(master, self.formatvar, self.smoothdict[self.smoothbeforeregression], *self.formatchoices, command=self.updateplot)
        self.smoothmenu.grid(row=rown, column=0)
        rown += 1
        # self.noiseratiolbl = ttk.Label(master, text= 'Maximum local max/min frequency in Wave Max Filter area allowed:')
        # self.noiseratiobox = tk.Spinbox(master, from_=-10000000000, to=10000000000, increment=0.5, width=10)
        # self.noiseratiobox.bind('<Return>', lambda *args: self.validate())

        self.local_minimum_check_var = tk.IntVar(value=1)
        if self.local_minimum_check == False:
            self.local_minimum_check_var.set(0)
        self.local_minimum_check_ttk = ttk.Checkbutton(master, text = "Stop only at local minimum", variable = self.local_minimum_check_var, \
                         onvalue = 1, offvalue = 0, command=self.updateplot)
        self.local_minimum_check_ttk.grid(row=rown, column=0)
        self.detection_update()
    
    def detection_update(self):
        if self.endtypechangeval.get() == 0:
        # if self.end_current_type == "threshold":
            # self.rlabel4_AnswerBox['state'] = 'normal'
            #disable all grid exponential. enable value threshold
            # self.rlabel4_AnswerBox['state'] = 'disabled'
            self.spinmaxbox['state'] = 'disabled'
            self.smoothmenu['state'] = 'disabled'
            self.local_minimum_check_ttk['state'] = 'disabled'
            # self.spinmaxbox2['state'] = 'normal'
        else:
            self.spinmaxbox['state'] = 'normal'
            self.smoothmenu['state'] = 'normal'
            self.local_minimum_check_ttk['state'] = 'normal'
            # self.spinmaxbox2['state'] = 'disabled'

    def validate(self):
        #nothing to validate
        fvalue = float(self.spinmaxbox.get().replace(",", "."))
        if fvalue <  0 or fvalue > 1:
            self.valid = False
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            return 0 
        self.valid = True
        return 1
        
    def updateplot(self, args=None):
        self.validate()
        if self.valid == True:
            self.result = [float(self.spinmaxbox.get().replace(",", ".")) , self.smoothdict[self.formatvar.get()] , self.noiseratio , bool(self.local_minimum_check_var.get())]
            self.literals["updatable_frame"].updateadjustexponential(self.result, endt=self.endtypechangeval.get(), endtv=float(self.spinmaxbox2.get().replace(",", ".")))
        
    def apply(self):
        self.result = [float(self.spinmaxbox.get().replace(",", ".")) , self.smoothdict[self.formatvar.get()] , self.noiseratio , bool(self.local_minimum_check_var.get())]
        self.literals["updatable_frame"].updateadjustexponential(self.result, True, endt=self.endtypechangeval.get(), endtv=float(self.spinmaxbox2.get().replace(",", ".")))

# class AdjustNoiseDetectDialog(tkDialog.Dialog):
class AdjustNoiseDetectDialog(tkDialog.DialogNonBlock):

    def body(self, master):
        # print("class AdjustNoiseDetectDialog def body creation")
        # ttk.Label(master, text='Noise Advanced Parameters:').grid(row=0, column=0)
        # self.checkfvar = tk.IntVar(value=self.literals["noiseareasfiltering"])
        # self.checkf = ttk.Checkbutton(master, text = "Noise Areas Min. Size filtering", variable = self.checkfvar, \
        #                  onvalue = 1, offvalue = 0)
        self.check_d_var = tk.IntVar(value=self.literals["noisedecrease"])
        self.checkd = ttk.Checkbutton(master, text = "Decrease Avg. Noise from Plot", variable = self.check_d_var, \
                         onvalue = 1, offvalue = 0, command=self.hidespin)
        
        self.check_u_var = tk.IntVar(value=self.literals["userdecrease"])
        self.checku = ttk.Checkbutton(master, text = "Decrease Custom Value from Plot", variable = self.check_u_var, \
                         onvalue = 1, offvalue = 0, command=self.showspin)
        
        self.spinlbl = ttk.Label(master, text= "Custom Value: ")
        self.spinu = tk.Spinbox(master,from_=-10000000000, to=10000000000, increment=0.1, width=10, command=self.updateplot)
        # self.spinu.bind('<Return>', lambda *args: self.validate())
        self.spinu.bind('<Return>', lambda *args: self.updateplot())

        # print('self.literals["noisedecreasevalue"]')
        # print(self.literals["noisedecreasevalue"])
        self.spinu.delete(0,"end")
        self.spinu.insert(0,self.literals["noisedecreasevalue"])
        rown = 0
        # self.checkf.grid(row=rown, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        # rown += 1
        self.checkd.grid(row=rown, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        rown += 1
        self.checku.grid(row=rown, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        rown += 1

        self.spinlbl.grid(row=rown, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        self.spinu.grid(row=rown, column=1, sticky=tk.W+tk.E+tk.N+tk.S)
        if self.check_u_var.get() == 0:
            self.spinlbl.grid_forget()
            self.spinu.grid_forget()
        # print("class AdjustNoiseDetectDialog def body created")

    def updateplot(self):
        self.validate()
        if self.valid == True:
            self.result = {}
            # if self.checkfvar.get() == 1:
            #     self.result["adjustnoisevar"] = True
            # else:
                # self.result["adjustnoisevar"]= False
            if self.check_d_var.get() == 1:
                self.result["noisedecrease"] = True
            else:
                self.result["noisedecrease"] = False
            if self.check_u_var.get() == 1:
                self.result["userdecrease"] = True
                self.result["noisevalue"] = float(self.spinu.get().replace(",", "."))
            else:
                self.result["userdecrease"] = False
                self.result["noisevalue"] = None
            self.literals["updatable_frame"].adjustnoiseupdate(self.result)
        pass

    def showspin(self):
        # print("class AdjustNoiseDetectDialog def showspin start")
        if self.check_u_var.get() == 0:
            self.spinu.grid_forget()
            self.spinlbl.grid_forget()
        else:
            self.check_d_var.set(0)
            self.spinlbl.grid(row=4, column=0)
            self.spinu.grid(row=4, column=1)
        self.updateplot()
        # print("class AdjustNoiseDetectDialog def showspin end")

    def hidespin(self):
        # print("class AdjustNoiseDetectDialog def hidespin start")
        # if self.check_u_var.get() == 0:
        # else:
        #     self.check_d_var.set(0)
        #     self.spinlbl.grid(row=4, column=0)
        #     self.spinu.grid(row=4, column=1)
        if self.check_d_var.get() == 1:
            self.check_u_var.set(0)
            self.spinu.grid_forget()
            self.spinlbl.grid_forget()
        self.updateplot()
        # print("class AdjustNoiseDetectDialog def hidespin end")

    def validate(self):
        # print("class AdjustNoiseDetectDialog def validate start")
        #nothing to validate
        if self.check_u_var.get() == 1:
            fvalue = float(self.spinu.get().replace(",", "."))
            if fvalue <  -10000000000 or fvalue > 10000000000:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0 
        self.valid = True
        # print("class AdjustNoiseDetectDialog def validate done")
        return 1

    def apply(self):
        # print("class AdjustNoiseDetectDialog def apply start")
        #save configs
        # print("self.valid")
        # print(self.valid)
        if self.valid == True:
            self.result = {}
            # if self.checkfvar.get() == 1:
            #     self.result["adjustnoisevar"] = True
            # else:
                # self.result["adjustnoisevar"]= False
            if self.check_d_var.get() == 1:
                self.result["noisedecrease"] = True
            else:
                self.result["noisedecrease"] = False
            if self.check_u_var.get() == 1:
                self.result["userdecrease"] = True
                self.result["noisevalue"] = float(self.spinu.get().replace(",", "."))
            else:
                self.result["userdecrease"] = False
                self.result["noisevalue"] = None
            # print("class AdjustNoiseDetectDialog def apply done")
            self.literals["updatable_frame"].adjustnoiseupdate(self.result, True)

class SaveFigureVideoDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class SaveFigureVideoDialog def body creation")
        nrow = 0
        ttk.Label(master, text='Output Format:').grid(row=nrow, column=0)


        self.formatvar = tk.StringVar(master)
        self.formatchoices = []
        self.formatchoices.extend(self.literals["formats"])
        self.formatchoices.append("avi")
        self.formatvar.set('png')
        self.optmenu = ttk.OptionMenu(master, self.formatvar, "png", *self.formatchoices, command=self.format_detection)
        self.optmenu.grid(row = nrow, column = 1)
        nrow += 1

        self.heightvar = tk.StringVar()
        self.heightvar.set(str(self.literals["height"]))
        ttk.Label(master, text='Height (Pixels):').grid(row=nrow, column=0)
        self.heightspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.heightvar, increment=1, width=10)
        self.heightspin.grid(row=nrow, column=1)
        self.heightspin.bind('<Return>', lambda *args: self.validate())
        nrow+=1

        self.widthvar = tk.StringVar()
        self.widthvar.set(str(self.literals["width"]))
        ttk.Label(master, text='Width (Pixels):').grid(row=nrow, column=0)
        self.widthspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.widthvar, increment=1, width=10)
        self.widthspin.grid(row=nrow, column=1)
        self.widthspin.bind('<Return>', lambda *args: self.validate())
        nrow+=1

        self.dpi = tk.StringVar()
        self.dpi.set(str(self.literals["dpi"]))
        # ttk.Label(master, text='Dots Per Inch (DPI):').grid(row=nrow, column=0)
        # self.dpispin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.dpi, increment=1, width=10)
        # self.dpispin.grid(row=nrow, column=1)
        # self.dpispin.bind('<Return>', lambda *args: self.validate())
        # nrow+=1

        # self.dpi = tk.StringVar()
        # self.dpi.set("300")
        # ttk.Label(master, text='Dots Per Inch (DPI):').grid(row=nrow, column=0)
        # self.dpispin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.dpi, increment=1, width=10)
        # self.dpispin.grid(row=nrow, column=1)
        # self.dpispin.bind('<Return>', lambda *args: self.validate())
        # nrow+=1

        self.quality = tk.StringVar()
        self.quality.set("70.0")
        self.jpglabel = ttk.Label(master, text='JPG Quality:')
        self.jpglabel.grid(row=nrow, column=0)
        self.jpglabel.grid_forget()
        self.qualityspin = tk.Spinbox(master, from_=1, to=100, textvariable=self.quality, increment=1, width=10)
        self.qualityspin.grid(row=nrow, column=1)
        self.qualityspin.bind('<Return>', lambda *args: self.validate())
        self.qualityspin.grid_forget()
        nrow+=1

        self.fpsrun = tk.StringVar()
        self.fpsrun.set("1")
        self.fpslabel = ttk.Label(master, text='Frames Per Second:')
        self.fpslabel.grid(row=nrow, column=0)
        self.fpslabel.grid_forget()
        self.fpsnspin = tk.Spinbox(master, from_=1, to=9999, textvariable=self.fpsrun, increment=1, width=10)
        self.fpsnspin.grid(row=nrow, column=1)
        self.fpsnspin.bind('<Return>', lambda *args: self.validate())
        self.fpsnspin.grid_forget()

    def format_detection(self, args=None):
        # print("args")
        # print(args)
        self.formatvar.set(args)
        if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
            self.jpglabel.grid(row=3, column=0)
            self.qualityspin.grid(row=3, column=1)
        else:        
            self.jpglabel.grid_forget()
            self.qualityspin.grid_forget()
        if self.formatvar.get() == "avi":
            self.fpslabel.grid(row=4, column=0)
            self.fpsnspin.grid(row=4, column=1)
        else:
            self.fpslabel.grid_forget()
            self.fpsnspin.grid_forget()

    def validate(self):
        # print("class SaveFigureVideoDialog def validate start")
        try:
            heightf= int(self.heightvar.get())
            if heightf < 0:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            widthf= int(self.widthvar.get())
            if widthf < 0:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            first= int(self.dpi.get())
            if first < 0:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
                second = float(self.quality.get())
                if second < 1 or second > 100:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            if self.formatvar.get() == "avi":
                third =  int(self.fpsrun.get())
                if third < 1 or third > 9999:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class SaveFigureVideoDialog def apply start")
        #save configs
        self.result = None
        if self.valid == True:
            self.result = {}
            fname = filedialog.asksaveasfile(defaultextension="." + self.formatvar.get())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                self.result = None
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            self.result["outtype"] = "image"
            if self.formatvar.get() == "avi":
                self.result["outtype"] = "video"
                self.result["fps"] = int(self.fpsrun.get())
            self.result["name"] = str(filename).split(".")[0] + "." + self.formatvar.get()
            self.result["format"] = self.formatvar.get()
            self.result["dpi"] = int(self.dpi.get())
            self.result["width"] = int(self.widthvar.get())
            self.result["height"] = int(self.heightvar.get())
            self.result["quality"] = float(self.quality.get())

class SaveFigureDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class SaveFigureDialog def body creation")

        nrow = 0

        if "height" in self.literals.keys():
            ttk.Label(master, text='Obs: Output dimensions refer to the full plot image size.').grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            ttk.Label(master, text='As such, the Legend dimensions will be resized during exporting').grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            self.previewheight = ttk.Label(master, text='Legend preview height: ' + self.literals["pheight"] + 'px').grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            self.previewwidth = ttk.Label(master, text='Legend preview width: ' + self.literals["pwidth"] + 'px').grid(row=nrow, column=0, columnspan=2)
            nrow += 1

        ttk.Label(master, text='Figure Format:').grid(row=nrow, column=0)

        self.formatvar = tk.StringVar(master)
        self.formatchoices = self.literals["formats"]
        self.formatvar.set('png')
        self.optmenu = ttk.OptionMenu(master, self.formatvar, "png", *self.formatchoices, command=self.jpg_detection)
        self.optmenu.grid(row = nrow, column = 1)
        nrow += 1
        
        self.dpi = tk.StringVar()
        if "dpi" in self.literals.keys():
             self.dpi.set(str(self.literals["dpi"]))
        else:
            self.dpi.set("300")
        if "height" in self.literals.keys():
            self.heightvar = tk.StringVar()
            self.heightvar.set(str(self.literals["height"]))
            ttk.Label(master, text='Full Image Height (Pixels):').grid(row=nrow, column=0)
            self.heightspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.heightvar, increment=1, width=10)
            self.heightspin.grid(row=nrow, column=1)
            self.heightspin.bind('<Return>', lambda *args: self.validate())
            nrow+=1
        if "width" in self.literals.keys():
            self.widthvar = tk.StringVar()
            self.widthvar.set(str(self.literals["width"]))
            ttk.Label(master, text='Full Image (Pixels):').grid(row=nrow, column=0)
            self.widthspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.widthvar, increment=1, width=10)
            self.widthspin.grid(row=nrow, column=1)
            self.widthspin.bind('<Return>', lambda *args: self.validate())
            nrow+=1
        else:
            ttk.Label(master, text='Dots Per Inch (DPI):').grid(row=nrow, column=0)
            self.dpispin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.dpi, increment=1, width=10)
            self.dpispin.grid(row=nrow, column=1)
            self.dpispin.bind('<Return>', lambda *args: self.validate())
            nrow+=1

        self.bboxvar = tk.StringVar(master)
        self.bboxvar.set('None')
        if self.literals["bbox"] == 1:
            self.bboxformatchoices = {"None", "tight"}
            ttk.Label(master, text='Bbox Type:').grid(row=nrow, column=0)
            self.bboxoptmenu = ttk.OptionMenu(master, self.bboxvar, "None", *self.bboxformatchoices)
            self.bboxoptmenu.grid(row = nrow, column = 1)
            nrow+=1

        self.quality = tk.StringVar()
        self.quality.set("70.0")
        self.jpglabel = ttk.Label(master, text='JPG Quality:')
        self.jpglabel.grid(row=nrow, column=0)
        self.jpglabel.grid_forget()
        self.qualityspin = tk.Spinbox(master, from_=1, to=100, textvariable=self.quality, increment=1, width=10)
        self.qualityspin.grid(row=nrow, column=1)
        self.qualityspin.bind('<Return>', lambda *args: self.validate())
        self.qualityspin.grid_forget()

    def jpg_detection(self, args=None):
        # print("args")
        # print(args)
        self.formatvar.set(args)
        if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
            self.jpglabel.grid(row=3, column=0)
            self.qualityspin.grid(row=3, column=1)
        else:        
            self.jpglabel.grid_forget()
            self.qualityspin.grid_forget()

    def validate(self):
        # print("class SaveFigureDialog def validate start")
        try:
            first= int(self.dpi.get())
            if first < 0:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            if "height" in self.literals.keys():
                heightf= int(self.heightvar.get())
                if heightf < 0:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            if "width" in self.literals.keys():
                widthf= int(self.widthvar.get())
                if widthf < 0:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            if "height" in self.literals.keys() and "width" in self.literals.keys():
                newwidth, newheight = self.literals["updatable_frame"].get_cax_size(int(self.widthvar.get()), int(self.heightvar.get()))
            if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
                second = float(self.quality.get())
                if second < 1 or second > 100:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class SaveFigureDialog def apply start")
        #save configs
        self.result = None
        if self.valid == True:
            self.result = {}
            fname = filedialog.asksaveasfile(defaultextension="." + self.formatvar.get())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                self.result = None
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            self.result["name"] = str(filename).split(".")[0] + "." + self.formatvar.get()
            self.result["format"] = self.formatvar.get()
            self.result["dpi"] = int(self.dpi.get())
            self.result["quality"] = float(self.quality.get())
            if "width" in self.literals.keys():
                self.result["width"] = int(self.widthvar.get())
            if "height" in self.literals.keys():
                self.result["height"] = int(self.heightvar.get())

            bboxv = self.bboxvar.get()
            if bboxv == "None":
                bboxv = None
            self.result["bbox"] = bboxv

class SaveLegendDialog(tkDialog.DialogNonBlock):

    def body(self, master):
        # print("class SaveFigureDialog def body creation")

        nrow = 0

        if "height" in self.literals.keys():
            ttk.Label(master, text='Obs: Output dimensions refer to the full plot image size.').grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            ttk.Label(master, text='As such, the Legend dimensions will be resized during exporting').grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            self.previewheight = ttk.Label(master, text='Legend preview height: ' + self.literals["pheight"] + 'px')
            self.previewheight.grid(row=nrow, column=0, columnspan=2)
            nrow += 1
            self.previewwidth = ttk.Label(master, text='Legend preview width: ' + self.literals["pwidth"] + 'px')
            self.previewwidth.grid(row=nrow, column=0, columnspan=2)
            nrow += 1

        ttk.Label(master, text='Figure Format:').grid(row=nrow, column=0)

        self.formatvar = tk.StringVar(master)
        self.formatchoices = self.literals["formats"]
        self.formatvar.set('png')
        self.optmenu = ttk.OptionMenu(master, self.formatvar, "png", *self.formatchoices, command=self.jpg_detection)
        self.optmenu.grid(row = nrow, column = 1)
        nrow += 1
        
        self.dpi = tk.StringVar()
        if "dpi" in self.literals.keys():
             self.dpi.set(str(self.literals["dpi"]))
        else:
            self.dpi.set("300")
        if "height" in self.literals.keys():
            self.heightvar = tk.StringVar()
            self.heightvar.set(str(self.literals["height"]))
            ttk.Label(master, text='Full Image Height (Pixels):').grid(row=nrow, column=0)
            self.heightspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.heightvar, increment=1, width=10, command=self.validate)
            self.heightspin.grid(row=nrow, column=1)
            self.heightspin.bind('<Return>', lambda *args: self.validate())
            nrow+=1
        if "width" in self.literals.keys():
            self.widthvar = tk.StringVar()
            self.widthvar.set(str(self.literals["width"]))
            ttk.Label(master, text='Full Image (Pixels):').grid(row=nrow, column=0)
            self.widthspin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.widthvar, increment=1, width=10, command=self.validate)
            self.widthspin.grid(row=nrow, column=1)
            self.widthspin.bind('<Return>', lambda *args: self.validate())
            nrow+=1
        else:
            ttk.Label(master, text='Dots Per Inch (DPI):').grid(row=nrow, column=0)
            self.dpispin = tk.Spinbox(master, from_=1, to=2147483646, textvariable=self.dpi, increment=1, width=10)
            self.dpispin.grid(row=nrow, column=1)
            self.dpispin.bind('<Return>', lambda *args: self.validate())
            nrow+=1

        self.bboxvar = tk.StringVar(master)
        self.bboxvar.set('None')
        if self.literals["bbox"] == 1:
            self.bboxformatchoices = {"None", "tight"}
            ttk.Label(master, text='Bbox Type:').grid(row=nrow, column=0)
            self.bboxoptmenu = ttk.OptionMenu(master, self.bboxvar, "None", *self.bboxformatchoices)
            self.bboxoptmenu.grid(row = nrow, column = 1)
            nrow+=1

        self.quality = tk.StringVar()
        self.quality.set("70.0")
        self.jpglabel = ttk.Label(master, text='JPG Quality:')
        self.jpglabel.grid(row=nrow, column=0)
        self.jpglabel.grid_forget()
        self.qualityspin = tk.Spinbox(master, from_=1, to=100, textvariable=self.quality, increment=1, width=10)
        self.qualityspin.grid(row=nrow, column=1)
        self.qualityspin.bind('<Return>', lambda *args: self.validate())
        self.qualityspin.grid_forget()

    def jpg_detection(self, args=None):
        # print("args")
        # print(args)
        self.formatvar.set(args)
        if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
            self.jpglabel.grid(row=3, column=0)
            self.qualityspin.grid(row=3, column=1)
        else:        
            self.jpglabel.grid_forget()
            self.qualityspin.grid_forget()

    def validate(self):
        # print("class SaveFigureDialog def validate start")
        try:
            first= int(self.dpi.get())
            if first < 0:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            if "height" in self.literals.keys():
                heightf= int(self.heightvar.get())
                if heightf < 0:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            if "width" in self.literals.keys():
                widthf= int(self.widthvar.get())
                if widthf < 0:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            if "height" in self.literals.keys() and "width" in self.literals.keys():
                newwidth, newheight = self.literals["updatable_frame"].get_cax_size(int(self.widthvar.get()), int(self.heightvar.get()))
                self.previewheight['text'] = 'Legend preview height: ' + str(int(newheight)) + 'px'
                self.previewwidth['text'] = 'Legend preview width: ' + str(int(newwidth)) + 'px'
            if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
                second = float(self.quality.get())
                if second < 1 or second > 100:
                    messagebox.showwarning(
                        "Bad input",
                        "Illegal values, please try again"
                    )
                    self.valid = False
                    return 0
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class SaveFigureDialog def apply start")
        #save configs
        self.result = None
        if self.valid == True:
            self.result = {}
            fname = filedialog.asksaveasfile(defaultextension="." + self.formatvar.get())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                self.result = None
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            self.result["name"] = str(filename).split(".")[0] + "." + self.formatvar.get()
            self.result["format"] = self.formatvar.get()
            self.result["dpi"] = int(self.dpi.get())
            self.result["quality"] = float(self.quality.get())
            if "width" in self.literals.keys():
                self.result["width"] = int(self.widthvar.get())
            if "height" in self.literals.keys():
                self.result["height"] = int(self.heightvar.get())

            bboxv = self.bboxvar.get()
            if bboxv == "None":
                bboxv = None
            self.result["bbox"] = bboxv
            self.literals["updatable_frame"].runexport(self.result)

class SaveTableDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class SaveTableDialog def body creation")
        # group = self.literals["current_group"]
        ttk.Label(master, text='Table Format:').grid(row=0, column=0)
        # Create a Tkinter variable
        self.formatvar = tk.StringVar(master)

        # Dictionary with options
        defform = 'XLSX'
        if _platform == "linux" or _platform == "linux2":
            # self.formatchoices = { 'CSV','XLS'}
            self.formatchoices = { 'XLS', 'CSV'}
            defform = 'XLS'
        else:
            # self.formatchoices = { 'CSV','XLSX'}
            self.formatchoices = { 'XLSX', 'CSV'}
        self.formatvar.set(defform) # set the default option

        # ttk.OptionMenu(master, self.formatvar, "CSV", *self.formatchoices).grid(row = 0, column = 1)
        ttk.OptionMenu(master, self.formatvar, defform, *self.formatchoices).grid(row = 0, column = 1)

    def validate(self):
        # print("class SaveTableDialog def validate start")
        self.valid = True
        return 1

    def apply(self):
        # print("class SaveTableDialog def apply start")
        #save table to file
         # Create a Pandas Excel writer using XlsxWriter as the engine.
        # writer = pd.ExcelWriter(FolderResult + "/" +str(NameResult) + "_Contratilidade.xlsx", engine='xlsxwriter')
       
        # # Convert the dataframe to an XlsxWriter Excel object.
        # data.to_excel(writer, sheet_name='Contractility')
       
        # # Get the xlsxwriter workbook and worksheet objects.
        # workbook  = writer.book
        # worksheet = writer.sheets['Contractility']
        # worksheet.set_column('B:L', 15)
       
        # # Add a header format.
        # header_format = workbook.add_format({
        #     'bold': True,
        #     'text_wrap': True,
        #     'valign': 'top',
        #     'fg_color': '#D7E4BC',
        #     'align': 'center','border': 1})
       
        # # Write the column headers with the defined format.
        # for col_num, value in enumerate(data.columns.values):
        #     worksheet.write(0, col_num + 1, value, header_format)
       
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()

        # try:
            
        format_sel = self.formatvar.get()
        data_type = self.literals["data_t"]
        data = np.array(self.literals["data"])
        # print("data")
        # print(data)
        headers = self.literals["headers"]
        if "single" in data_type:
            new_dict = {}
            for header, data_col in zip(headers, data):
                new_dict[header] = data_col.copy()
            new_df = pd.DataFrame(new_dict)
            fname = filedialog.asksaveasfile(defaultextension="." + format_sel.lower())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            if format_sel == "CSV":
                new_df.to_csv(str(filename),index=False)
            elif format_sel == "XLS" or format_sel == "XLSX":
                sheet_n = "Data"
                if "sheetnames" in self.literals.keys():
                    sheet_n = self.literals["sheetnames"]
                writer = pd.ExcelWriter(str(filename).split(".")[0] + "." + format_sel.lower(), engine='xlsxwriter', mode='w')
                # new_df.to_excel(str(fname.name), sheet_name=sheet_n,index=False, encoding='utf8')
                new_df.to_excel(writer, sheet_name=sheet_n,index=False)#, encoding='utf8')
                # # Add a header format.
                workbook = writer.book
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'align': 'center','border': 1})
                worksheet = writer.sheets[sheet_n]

                for idx, col in enumerate(new_df):  # loop through all columns
                    series = new_df[col]
                    max_len = max((
                        series.astype(str).map(len).max(),  # len of largest item
                        len(str(series.name))  # len of column name/header
                        )) + 1  # adding a little extra space
                    worksheet.set_column(idx, idx, max_len)  # set column width
                # worksheet.set_column('B:L', 15)
                for col_num, value in enumerate(new_df.columns.values):
                    # worksheet.write(0, col_num + 1, value, header_format)
                    worksheet.write(0, col_num, value, header_format)
                writer.save()
            messagebox.showinfo(
                "File saved",
                "File was successfully saved"
            )
        elif "multiple" in data_type:
            sheets = self.literals["sheetnames"]
            # print("sheets")
            # print(sheets)
            fname = filedialog.asksaveasfile(defaultextension="." + format_sel.lower())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            # print("filename")
            # print(filename)
            writer = None
            workbook = None
            header_format = None
            topmerge_format_odd = None
            topmerge_format_even = None
            if format_sel == "XLS" or format_sel == "XLSX":
                writer = pd.ExcelWriter(str(filename).split(".")[0] + "." + format_sel.lower(), engine='xlsxwriter', mode='w')
                workbook = writer.book
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'align': 'center','border': 1})
                topmerge_format_odd = workbook.add_format({
                    'bold': False,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#FDEB9C',
                    'align': 'center','border': 1})
                topmerge_format_even = workbook.add_format({
                    'bold': False,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#FFCB99',
                    'align': 'center','border': 1})
            sheet_idx = 0
            print("headers")
            print(headers)
            print("data")
            print(data)
            print("sheets")
            print(sheets)
            for e_header, e_data, e_sheet in zip(headers, data, sheets):
                new_dict = {}
                new_df = None
                for header, data_col in zip(e_header, e_data):
                    # print("header")
                    # print(header)
                    # print("data_col")
                    # print(data_col)
                    # print("type(data_col)")
                    # print(type(data_col))
                    new_dict[header] = data_col.copy()
                new_df = pd.DataFrame(new_dict)
                if format_sel == "CSV":
                    new_df.to_csv(str(filename).split(".")[0] + "_" + e_sheet + "." + str(filename).split(".")[1], index=False)
                elif format_sel == "XLS" or format_sel == "XLSX":
                    startrow_idx = 0
                    #check for topmerges in current sheet
                    if "mergetop" in self.literals.keys():
                        if sheet_idx in self.literals["mergetop"].keys():
                            # startrow_idx is 1 if topmerges
                            startrow_idx = 1
                    new_df.to_excel(writer, sheet_name=e_sheet,index=False, startrow=startrow_idx)
                    worksheet = writer.sheets[e_sheet]
                    for idx, col in enumerate(new_df):  # loop through all columns
                        series = new_df[col]
                        max_len = max((
                            series.astype(str).map(len).max(),  # len of largest item
                            len(str(series.name))  # len of column name/header
                            )) + 1  # adding a little extra space
                        worksheet.set_column(idx, idx, max_len)  # set column width
                    # worksheet.set_column('B:L', 15)
                    #check for topmerges in current sheet
                    #write topmerge data
                    #merge top merge data
                    if "mergetop" in self.literals.keys():
                        if sheet_idx in self.literals["mergetop"].keys():
                            for col_num in range(len(self.literals["mergetop"][sheet_idx])):
                                col_id_here = xlsxwriter.utility.xl_col_to_name(self.literals["mergetop"][sheet_idx][col_num][1]-1)
                                col_id_there = xlsxwriter.utility.xl_col_to_name(self.literals["mergetop"][sheet_idx][col_num][2]-1)
                                if col_num % 2 == 0:
                                    # worksheet.write(0, col_num, literals["mergetop"][sheet_idx][col_num][0], topmerge_format_odd)
                                    # worksheet.merge_range( "A" + str(self.literals["mergetop"][sheet_idx][col_num][1]) + ":A" + str(self.literals["mergetop"][sheet_idx][col_num][2]), self.literals["mergetop"][sheet_idx][col_num][0], topmerge_format_odd)
                                    worksheet.merge_range( col_id_here + "1:"+col_id_there+"1", self.literals["mergetop"][sheet_idx][col_num][0], topmerge_format_odd)
                                else:
                                    # worksheet.write(0, col_num, literals["mergetop"][sheet_idx][col_num][0], topmerge_format_even)
                                    # worksheet.merge_range( "A" + str(self.literals["mergetop"][sheet_idx][col_num][1]) + ":A" + str(self.literals["mergetop"][sheet_idx][col_num][2]), self.literals["mergetop"][sheet_idx][col_num][0], topmerge_format_even)
                                    worksheet.merge_range( col_id_here + "1:"+col_id_there+"1", self.literals["mergetop"][sheet_idx][col_num][0], topmerge_format_even)
                    for col_num, value in enumerate(new_df.columns.values):
                        # worksheet.write(0, col_num + 1, value, header_format)
                        worksheet.write(startrow_idx, col_num, value.split("_")[0], header_format)
                sheet_idx +=1
            if format_sel == "XLS" or format_sel == "XLSX":
                writer.save()
            messagebox.showinfo(
                "File saved",
                "File was successfully saved"
            )
            
        # except Exception as e:
        #     messagebox.showerror("Error", "Could not save Data file\n" + str(e))

class SummarizeTablesDialog(tkDialog.Dialog):
    def body(self, master):
        # print("class SummarizeTablesDialog def body creation")

        tframe = ttk.Frame(self)
        scrollbar = ttk.Scrollbar(tframe, orient=tk.VERTICAL)
        scrollbar2 = ttk.Scrollbar(tframe, orient=tk.HORIZONTAL)
        self.listbox = tk.Listbox(tframe, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar2.set, selectmode=tk.SINGLE, width=50)
        scrollbar.config(command=self.listbox.yview)
        scrollbar2.config(command=self.listbox.xview)
        
        # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # scrollbar2.pack(side=tk.BOTTOM, fill=tk.X)

        scrollbar.grid(row=0,column=1,sticky=tk.NS)
        scrollbar2.grid(row=1,column=0,sticky=tk.EW)

        # self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.listbox.grid(row=0, column=0, sticky=tk.NSEW)

        for i in range(0,2):
            tframe.rowconfigure(i, weight=1)
        for i in range(0,2):
            tframe.columnconfigure(i, weight=1)

        # tframe.grid(row=1, column=0, rowspan=5, columnspan=3)
        tframe.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.addtableimg = tk.PhotoImage(file="icons/add-sharp.png")
        self.deltableimg = tk.PhotoImage(file="icons/close-sharp.png")

        bframe = ttk.Frame(self)

        btn1frame = ttk.Frame(bframe)
        btn1lbl = ttk.Label(btn1frame, text="Add Table")
        btn1lbl.grid(row=0,column=1)
        button_addgroup = ttk.Button(btn1frame, image=self.addtableimg,
                   command=self.add_table)
        button_addgroup.image=self.addtableimg
        button_addgroup.grid(row=0, column=0, columnspan=1)
        btn1frame.grid(row=0, column=0, columnspan=1)

        btn2frame = ttk.Frame(bframe)
        btn2lbl = ttk.Label(btn2frame, text="Delete Table")
        btn2lbl.grid(row=0, column=1)
        button_deletegroup = ttk.Button(btn2frame, image=self.deltableimg,
           command=self.delete_table)
        button_deletegroup.image=self.deltableimg
        button_deletegroup.grid(row=0, column=0, columnspan=1)
        btn2frame.grid(row=0, column=2, columnspan=1)

        for i in range(0,1):
            bframe.rowconfigure(i, weight=1)
        for i in range(0,3):
            bframe.columnconfigure(i, weight=1)
        # bframe.grid(row=7, column=0, rowspan=1, columnspan=3)
        bframe.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # for i in range(0,8):
        #     self.rowconfigure(i, weight=1)
        # for i in range(0,3):
        #     self.columnconfigure(i, weight=1)
        # print("class SummarizeTablesDialog def body created")
    
    def add_table(self, event=None):
        # print("class SummarizeTablesDialog def add_table start")
        # print("class SummarizeTablesDialog def add_table open file dialog selection")
        # if _platform == "linux" or _platform == "linux2":
            # filenames = filedialog.askopenfilenames(parent=self,title='Choose Data Tables to Summarize:',filetypes = (("XLS Files","*.xls"), ("XLSX Files","*.xlsx")))
        # else:
        filenames = filedialog.askopenfilenames(parent=self,title='Choose Data Tables to Summarize:',filetypes = (("XLSX Files","*.xlsx"), ("XLS Files","*.xls")))
        # print("class SummarizeTablesDialog def add_table open file dialog done")
        # print("filenames")
        # print(filenames)
        if filenames:
            # filenames = [r'%s' for fna in filenames]
            newfilenames = []
            for fna in filenames:
                newfilenames.append(r'%s' %fna)
            # print("class SummarizeTablesDialog def add_table 1 or more files selected")
            for filename in newfilenames:
                # print("class SummarizeTablesDialog def add_table inserting file in listbox")
                self.listbox.insert(tk.END, filename)
        # print("class SummarizeTablesDialog def add_table done")

    
    def delete_table(self, event=None):
        # print("class SummarizeTablesDialog def delete_table start")
        self.listbox.delete(self.listbox.curselection()[0])
        # print("class SummarizeTablesDialog def delete_table done")

    def validate(self):
        # print("class SummarizeTablesDialog def validate start")
        items = self.listbox.get(0, tk.END)
        # print("items")
        # print(items)
        if len(items) > 0:
            for item in items:
                # xl = pd.ExcelFile(item)
                # xl = pd.ExcelFile(item, engine='openpyxl')
                xl = None
                if ".xlsx" in item:
                    xl = pd.ExcelFile(item, engine='openpyxl')
                else:
                    xl = pd.ExcelFile(item)
                names = xl.sheet_names  # see all sheet names
                timeexist = [name for name in names if "Avg. Time" in name and "Decay" not in name]
                speedexist = [name for name in names if "Avg. Speed" in name]
                areaexist = [name for name in names if "Avg. Area" in name]
                decayexist = [name for name in names if "Avg. Time" in name and "Decay" in name]
                # if "Avg. Time" not in names or "Avg. Speed" not in names or "Avg. Area" not in names:
                if len(timeexist) == 0 or len(speedexist) == 0 or len(areaexist) == 0 or len(decayexist) == 0:
                    messagebox.showerror("Error", "Wrong table format")
                    self.valid = False
                    return 0
            self.valid = True
            return 1
        else:
            messagebox.showerror("Error", "No tables selected")
            self.valid = False
            return 0

        # print("class SummarizeTablesDialog def validate done")

    def apply(self):
        # print("class SummarizeTablesDialog def apply start")
        timeavgdf = {
            "Name": []
        }
        timedecayavgdf = {
            "Name": []
        }
        speedvgdf = {
            "Name": []
        }
        areaavgdf = {
            "Name": []
        }
        current_timeunit = None
        for item in self.listbox.get(0, tk.END):
            conversion_timefactor = 1
            xl = None
            if ".xlsx" in item:
                xl = pd.ExcelFile(item, engine='openpyxl')
            else:
                xl = pd.ExcelFile(item)
            xl_sheetnames = xl.sheet_names 
            avgtime_sheetname = [shname for shname in xl_sheetnames if "Avg. Time" in shname and "Decay" not in shname][0]
            # timedf = xl.parse("Avg. Time")  # read a specific sheet to DataFrame
            avgtime_sheetname_unit = avgtime_sheetname.split("(")[1].split(")")[0]
            if current_timeunit is None:
                current_timeunit = avgtime_sheetname_unit
            elif avgtime_sheetname_unit != current_timeunit and avgtime_sheetname_unit == "s":
                #second to millisecond
                # conversion_timefactor = 000.1
                conversion_timefactor = 1000
                # conversion_timefactor = 000.1
            elif avgtime_sheetname_unit != current_timeunit and avgtime_sheetname_unit == "ms":
                #millisecond to second
                # conversion_timefactor = 1000
                conversion_timefactor = 0.001
            timedf = xl.parse(avgtime_sheetname)  # read a specific sheet to DataFrame
            k1 = timedf.keys()[0]
            v1 = timedf[k1]
            for v in v1:
                timeavgdf["Name"].append(os.path.basename(item))
            for key in timedf.keys():
                if key not in timeavgdf.keys():
                    timeavgdf[key] = []
                values = timedf[key]
                for value in values:
                    timeavgdf[key].append(float(value) * conversion_timefactor)
            
            avgtimedecay_sheetname = [shname for shname in xl_sheetnames if "Avg. Time" in shname and "Decay" in shname][0]

            avgtimedecay_sheetname_unit = avgtimedecay_sheetname.split("(")[2].split(")")[0]
            # timedecayavgdf
            timedecaydf = xl.parse(avgtimedecay_sheetname, skiprows=1)  # read a specific sheet to DataFrame
            k1 = timedecaydf.keys()[0]
            v1 = timedecaydf[k1]
            for v in v1:
                timedecayavgdf["Name"].append(os.path.basename(item))
            for key in timedecaydf.keys():
                if key not in timedecayavgdf.keys():
                    timedecayavgdf[key] = []
                values = timedecaydf[key]
                for value in values:
                    timedecayavgdf[key].append(value)

            avgspeed_sheetname = [shname for shname in xl_sheetnames if "Avg. Speed" in shname][0]
            # timedf = xl.parse("Avg. Time")  # read a specific sheet to DataFrame
            avgspeed_sheetname_unit = avgtime_sheetname.split("(")[1].split(")")[0]
            # speeddf = xl.parse("Avg. Speed")  # read a specific sheet to DataFrame
            speeddf = xl.parse(avgspeed_sheetname)  # read a specific sheet to DataFrame
            k1 = speeddf.keys()[0]
            v1 = speeddf[k1]
            for v in v1:
                speedvgdf["Name"].append(os.path.basename(item))
            for key in speeddf.keys():
                if key not in speedvgdf.keys():
                    speedvgdf[key] = []
                values = speeddf[key]
                for value in values:
                    speedvgdf[key].append(value)
            avgarea_sheetname = [shname for shname in xl_sheetnames if "Avg. Area" in shname][0]
            # timedf = xl.parse("Avg. Time")  # read a specific sheet to DataFrame
            avgarea_sheetname_unit = avgtime_sheetname.split("(")[1].split(")")[0]
            # areadf = xl.parse("Avg. Area")  # read a specific sheet to DataFrame
            areadf = xl.parse(avgarea_sheetname)  # read a specific sheet to DataFrame
            k1 = areadf.keys()[0]
            v1 = areadf[k1]
            for v in v1:
                areaavgdf["Name"].append(os.path.basename(item))
            for key in areadf.keys():
                if key not in areaavgdf.keys():
                    areaavgdf[key] = []
                values = areadf[key]
                for value in values:
                    areaavgdf[key].append(value)
        try:
            format_do = ".xlsx"
            if _platform == "linux" or _platform == "linux2":
                format_do = ".xls"
            fname = filedialog.asksaveasfile(defaultextension=format_do)
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            fname.close()
            fnamename = str(fname.name)
            filename =  r'%s' %fnamename
            os.remove(str(filename))
            # print("filename")
            # print(filename)

            writer = pd.ExcelWriter(str(filename).split(".")[0] + format_do, engine='xlsxwriter', mode='w')
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'align': 'center','border': 1})
            topmerge_format_odd = workbook.add_format({
                'bold': False,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#FDEB9C',
                'align': 'center','border': 1})
            topmerge_format_even = workbook.add_format({
                'bold': False,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#FFCB99',
                'align': 'center','border': 1})
            timeavgdf = pd.DataFrame(timeavgdf)
            # timeavgdf.to_excel(writer, sheet_name="Time. Concat",index=False)
            timeavgdf.to_excel(writer, sheet_name="Time. Concat (" + self.master.controller.current_timescale + ")",index=False)
            # worksheet = writer.sheets["Time. Concat"]
            worksheet = writer.sheets["Time. Concat (" + self.master.controller.current_timescale + ")"]
            for idx, col in enumerate(timeavgdf):  # loop through all columns
                series = timeavgdf[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
            for col_num, value in enumerate(timeavgdf.columns.values):
                worksheet.write(0, col_num, value, header_format)

            speedvgdf = pd.DataFrame(speedvgdf)
            # speedvgdf.to_excel(writer, sheet_name="Speed. Concat",index=False)
            speedvgdf.to_excel(writer, sheet_name="Speed. Concat (" + self.master.controller.current_speedscale .replace('/', ' per ')+ ")",index=False)
            # worksheet = writer.sheets["Speed. Concat"]
            worksheet = writer.sheets["Speed. Concat (" + self.master.controller.current_speedscale.replace('/', ' per ') + ")"]
            for idx, col in enumerate(speedvgdf):  # loop through all columns
                series = speedvgdf[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
            for col_num, value in enumerate(speedvgdf.columns.values):
                worksheet.write(0, col_num, value, header_format)

            areaavgdf = pd.DataFrame(areaavgdf)
            # areaavgdf.to_excel(writer, sheet_name="Area. Concat",index=False)
            areaavgdf.to_excel(writer, sheet_name="Area. Concat (" + self.master.controller.current_areascale + ")",index=False)
            # worksheet = writer.sheets["Area. Concat"]
            worksheet = writer.sheets["Area. Concat (" + self.master.controller.current_areascale + ")"]
            for idx, col in enumerate(areaavgdf):  # loop through all columns
                series = areaavgdf[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
            for col_num, value in enumerate(areaavgdf.columns.values):
                worksheet.write(0, col_num, value, header_format)

            timedecayavgdf = pd.DataFrame(timedecayavgdf)
            # areaavgdf.to_excel(writer, sheet_name="Area. Concat",index=False)
            timedecayavgdf.to_excel(writer, sheet_name="Time Decay. Concat (" + self.master.controller.current_timescale + " %)",index=False, startrow=1)
            # worksheet = writer.sheets["Area. Concat"]
            worksheet = writer.sheets["Time Decay. Concat (" + self.master.controller.current_timescale + " %)"]
            for idx, col in enumerate(timedecayavgdf):  # loop through all columns
                series = timedecayavgdf[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width

            # merge_top = [["T10", 1, 3] , ["T20", 4,6] , ["T30", 7, 9] , ["T40", 10, 12] , ["T50", 13, 15] , ["T60", 16, 18] , ["T70", 19, 21] , ["T80", 22, 24] , ["T90", 25, 27]]
            merge_top = [["T10", 2, 4] , ["T20", 5,7] , ["T30", 8, 10] , ["T40", 11, 13] , ["T50", 14, 16] , ["T60", 17, 19] , ["T70", 20, 22] , ["T80", 23, 25] , ["T90", 26, 28]]
            for col_num in range(len(merge_top)):
                col_id_here = xlsxwriter.utility.xl_col_to_name(merge_top[col_num][1]-1)
                col_id_there = xlsxwriter.utility.xl_col_to_name(merge_top[col_num][2]-1)
                if col_num % 2 == 0:
                    worksheet.merge_range( col_id_here + "1:"+col_id_there+"1", merge_top[col_num][0], topmerge_format_odd)
                else:
                    worksheet.merge_range( col_id_here + "1:"+col_id_there+"1", merge_top[col_num][0], topmerge_format_even)

            for col_num, value in enumerate(timedecayavgdf.columns.values):
                if value.endswith('.1') or value.endswith('.2') or value.endswith('.3') or value.endswith('.4') or value.endswith('.5') or value.endswith('.6') or value.endswith('.7') or value.endswith('.8'):
                    value = value[:-2]
                # worksheet.write(0, col_num, value, header_format)
                worksheet.write(1, col_num, value, header_format)

            writer.save()
            messagebox.showinfo(
                "File saved",
                "File was successfully saved"
            )
        except Exception as e:
            # print(str(e))
            messagebox.showerror("Error", "Could not save Data file\n" + str(e))

        # print("class SummarizeTablesDialog def apply end")

class SavGolDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class SavGolDialog def body creation")

        ttk.Label(master, text='Window Length:').grid(row=0, column=0)

        self.e1 = tk.Spinbox(master, from_=self.literals["windowstart"], to=self.literals["maxvalues"], increment=1, width=10)
        self.e1.grid(row=0, column=1)
        self.e1.bind('<Return>', lambda *args: self.validate())


        
        ttk.Label(master, text='Polynomial Order:').grid(row=1, column=0)

        self.e2 = tk.Spinbox(master, from_=self.literals["polystart"], to=self.literals["maxvalues"], increment=1, width=10)
        self.e2.grid(row=1, column=1)
        self.e2.bind('<Return>', lambda *args: self.validate())

        self.valid = False

        return self.e1 # initial focus

    def validate(self):
        # print("class SavGolDialog def validate start")
        try:
            first= int(self.e1.get().replace(",", "."))
            second = int(self.e2.get().replace(",", "."))
            #third = int(self.e3var.get())
            if first > self.literals["maxvalues"] or second > self.literals["maxvalues"]:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            if first % 2 == 0:
                messagebox.showwarning(
                    "Bad input",
                    "Window length must be odd"
                )
                self.valid = False
                return 0
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class SavGolDialog def apply start")
        first = int(self.e1.get().replace(",", "."))
        second = int(self.e2.get().replace(",", "."))
        self.result = first, second

        # print first, second # or something

class NpConvDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class NpConvDialog def body creation")

        ttk.Label(master, text='Window Length:').grid(row=0, column=0)

        self.e1 = tk.Spinbox(master, from_=self.literals["windowstart"], to=self.literals["maxvalues"], increment=1, width=10)
        self.e1.grid(row=0, column=1)
        self.e1.bind('<Return>', lambda *args: self.validate())


        # Dictionary with options
        ttk.Label(master, text='Window Scaling Type:').grid(row=1, column=0)
        self.formatvar = tk.StringVar(master)
        self.formatchoices = {'Flat', 'Hanning', 'Hamming', 'Bartlett', 'Blackman'}
        self.formatvar.set(self.literals["window_type"]) # set the default option

        ttk.OptionMenu(master, self.formatvar, self.literals["window_type"], *self.formatchoices).grid(row = 1, column = 1)

        self.valid = False

        return self.e1 # initial focus

    def validate(self):
        # print("class NpConvDialog def validate start")
        try:
            first= int(self.e1.get().replace(",", "."))
            if first > self.literals["maxvalues"]:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            self.valid = True
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            return 0

    def apply(self):
        # print("class NpConvDialog def apply start")
        first = int(self.e1.get().replace(",", "."))
        self.result = first, self.formatvar.get()

class FourierConvDialog(tkDialog.Dialog):

    def body(self, master):
        # print("class FourierConvDialog def body creation")
        ttk.Label(master, text="% of Frequencies kept:").grid(row=0, column=0)

        self.e1 = tk.Spinbox(master, from_=self.literals["freqstart"], to=self.literals["maxvalues"], increment=0.1, width=10)
        self.e1.grid(row=0, column=1)
        self.e1.bind('<Return>', lambda *args: self.validate())


        self.valid = False

        # print("class FourierConvDialog def body created")
        return self.e1 # initial focus

    def validate(self):
        # print("class FourierConvDialog def validate start")
        try:
            first= float(self.e1.get().replace(",", "."))
            if first > self.literals["maxvalues"]:
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                self.valid = False
                return 0
            self.valid = True
            # print("class FourierConvDialog def validate true")
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            # print("class FourierConvDialog def validate false")
            return 0

    def apply(self):
        # print("class FourierConvDialog def apply start")
        first = float(self.e1.get().replace(",", "."))
        self.result = first
        # print("class FourierConvDialog def apply done")