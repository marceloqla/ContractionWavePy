import tkinter as tk
from tkinter import ttk

# ************************
# Scrollable Frame Class
# ************************
class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent) # create a frame (self)

        defaultbg = parent.cget('bg')
        # self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")          #place canvas on self
        self.canvas = tk.Canvas(self, borderwidth=0, background=defaultbg)          #place canvas on self
        # self.viewPort = tk.Frame(self.canvas, background="#ffffff")                    #place a frame on the canvas, this frame will hold the child widgets 
        self.viewPort = tk.Frame(self.canvas, background=defaultbg)                    #place a frame on the canvas, this frame will hold the child widgets 
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
        self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas
        
        # self.canvas.configure(yscrollcommand=self.scrollGetPosition)                          #attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")                                       #pack scrollbar to right of self
        self.canvas.pack(side="left", fill="both", expand=True)                     #pack canvas to left of self and expand to fil
        self.canvas_window = self.canvas.create_window((4,4), window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                  tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.onFrameConfigure)                       #bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>", self.onCanvasConfigure)                       #bind an event whenever the size of the viewPort frame changes.

        self.onFrameConfigure(None)                                                 #perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize

    # def scrollGetPosition(self, y0, y1):
    #     self.vsb.set(y0, y1)
    #     print("YSCROLLED!")
    #     print("y0, y1")
    #     print(y0, y1)
    #     print("")
    #     if float(y0) < .1 and self.infinitescroll == True:
    #         print("MOVED TO NEAR START EDGE")
    #     if float(y1) > .9 and self.infinitescroll == True:
    #         print("MOVED TO NEAR END EDGE")

    def onFrameConfigure(self, event):                                              
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))                 #whenever the size of the frame changes, alter the scroll region respectively.

    def onCanvasConfigure(self, event):
        '''Reset the canvas window to encompass inner frame when required'''
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width = canvas_width)            #whenever the size of the canvas changes alter the window region respectively.


class ttkScrollFrame(ttk.Frame):
    def __init__(self, parent, tkstyle, bgcolor=None, cvSize = (4,4), cvExpand = True, cvWidth=False):
        super().__init__(parent) # create a frame (self)

        defaultbg = bgcolor
        if not cvWidth:
            self.canvas = tk.Canvas(self, borderwidth=0, background=defaultbg)          #place canvas on self
            self.viewPort = ttk.Frame(self.canvas)#, style=tkstyle)                    #place a frame on the canvas, this frame will hold the child widgets 
        else:
            self.canvas = tk.Canvas(self, width=cvWidth, borderwidth=0, background=defaultbg)          #place canvas on self
            self.viewPort = ttk.Frame(self.canvas, width=cvWidth)#, style=tkstyle)                    #place a frame on the canvas, this frame will hold the child widgets 
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
        self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas
        
        # self.canvas.configure(yscrollcommand=self.scrollGetPosition)                          #attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")                                       #pack scrollbar to right of self
        self.canvas.pack(side="left", fill="both", expand=cvExpand)                     #pack canvas to left of self and expand to fil
        self.canvas_window = self.canvas.create_window(cvSize, window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                  tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.onFrameConfigure)                       #bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>", self.onCanvasConfigure)                       #bind an event whenever the size of the viewPort frame changes.

        self.onFrameConfigure(None)                                                 #perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize

    def onFrameConfigure(self, event):                                              
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))                 #whenever the size of the frame changes, alter the scroll region respectively.

    def onCanvasConfigure(self, event):
        '''Reset the canvas window to encompass inner frame when required'''
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width = canvas_width)            #whenever the size of the canvas changes alter the window region respectively.
