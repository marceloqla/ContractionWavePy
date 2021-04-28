import math
import numpy as np
from scipy.signal import argrelextrema
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tkinter import messagebox
from peakdetectpure import peakdet
from matplotlib.artist import Artist

class PeaksObj(object):
    def __init__(self):
        print("class PeaksObj def init start")
        self.thisgroup = None
        self.thisframes = None
        self.peaks = []
        self.mag_sindex = None
        self.mag_findex = None
        print("class PeaksObj def init end")
    def convert_timescales_all(self, timescale):
        print("class PeaksObj def convert_timescales_all start")
        for peak in self.peaks:
            peak.switch_timescale(timescale)
        print("class PeaksObj def convert_timescales_all end")

class PeakObj(object):
    def __init__(self):
        print("class PeakObj def init start")
        # self.thisgroup = None
        # self.thisframes = None
        print("class PeakObj def init initialize all state variables")
        self.first = None
        self.fmax = None
        self.min = None
        self.smax = None
        self.last = None
        self.time_scale = None

        self.peaktimes = []
        self.peakdata = []
        self.fulldata = []

        self.FPS = None
        self.pixel_val = None

        self.parameters = {
        "Contraction-Relaxation Time (CRT)": 0.0,
        "Contraction Time (CT)": 0.0,
        "Relaxation Time (RT)": 0.0,
        "Contraction time-to-peak (CTP)": 0.0,
        "Contraction time from peak to minimum speed (CTPMS)": 0.0,
        "Relaxation time-to-peak (RTP)": 0.0,
        "Relaxation time from peak to Baseline (RTPB)": 0.0,
        "Time between Contraction-Relaxation maximum speed (TBC-RMS)": 0.0,

        'Maximum Contraction Speed (MCS)': 0.0,
        'Maximum Relaxation Speed (MRS)': 0.0,
        'MCS/MRS Difference Speed (MCS/MRS-DS)': 0.0,

        'Contraction-Relaxation Area (CRA)': 0.0,
        'Shortening Area (SA)': 0.0
        }

        self.firsttime = None
        self.secondtime = None
        self.thirdtime = None
        self.fourthtime = None
        self.fifthtime = None

        self.firstvalue = None
        self.secondvalue = None
        self.thirdvalue = None
        self.fourthvalue = None
        self.fifthvalue = None
        print("class PeakObj def init initialize all state variables done")
        print("class PeakObj def init done")

    def switch_timescale(self, timescale):
        print("class PeakObj def switch_timescale start")
        print("class PeakObj def switch_timescale conversion from: " + self.time_scale + " to " + timescale)
        if timescale == "s" and self.time_scale != "s":
            self.firsttime = self.firsttime / 1000
            self.secondtime = self.secondtime / 1000
            self.thirdtime = self.thirdtime / 1000
            self.fourthtime = self.fourthtime / 1000
            self.fifthtime = self.fifthtime / 1000
            for i,a in enumerate(self.peaktimes):
                self.peaktimes[i] = a / 1000
            self.parameters["Contraction-Relaxation Time (CRT)"] = (self.fifthtime - self.firsttime)
            self.parameters["Contraction Time (CT)"] = (self.thirdtime - self.firsttime)
            self.parameters["Relaxation Time (RT)"] = (self.fifthtime - self.thirdtime)
            self.parameters["Contraction time-to-peak (CTP)"] = (self.secondtime - self.firsttime)
            self.parameters["Contraction time from peak to minimum speed (CTPMS)"] = (self.thirdtime - self.secondtime)
            self.parameters["Relaxation time-to-peak (RTP)"] = (self.fourthtime - self.thirdtime)
            self.parameters["Relaxation time from peak to Baseline (RTPB)"] = (self.fifthtime - self.fourthtime)
            self.parameters["Time between Contraction-Relaxation maximum speed (TBC-RMS)"] = (self.fourthtime - self.secondtime)
            self.time_scale = timescale
        elif timescale == "ms" and self.time_scale != "ms":
            self.firsttime = self.firsttime * 1000
            self.secondtime = self.secondtime * 1000
            self.thirdtime = self.thirdtime * 1000
            self.fourthtime = self.fourthtime * 1000
            self.fifthtime = self.fifthtime * 1000
            for i,a in enumerate(self.peaktimes):
                self.peaktimes[i] = a * 1000
            self.parameters["Contraction-Relaxation Time (CRT)"] = (self.fifthtime - self.firsttime)
            self.parameters["Contraction Time (CT)"] = (self.thirdtime - self.firsttime)
            self.parameters["Relaxation Time (RT)"] = (self.fifthtime - self.thirdtime)
            self.parameters["Contraction time-to-peak (CTP)"] = (self.secondtime - self.firsttime)
            self.parameters["Contraction time from peak to minimum speed (CTPMS)"] = (self.thirdtime - self.secondtime)
            self.parameters["Relaxation time-to-peak (RTP)"] = (self.fourthtime - self.thirdtime)
            self.parameters["Relaxation time from peak to Baseline (RTPB)"] = (self.fifthtime - self.fourthtime)
            self.parameters["Time between Contraction-Relaxation maximum speed (TBC-RMS)"] = (self.fourthtime - self.secondtime)
            self.time_scale = timescale
        print("class PeakObj def switch_timescale done")

    def calc_parameters(self, recalc=True):
        print("class PeakObj def calc_parameters start")
        self.peakdata = []
        self.peaktimes = []
        print("class PeakObj def calc_parameters retrieving time and value data")
        for i in range(self.first, self.last+1):
            self.peaktimes.append(i / self.FPS)
            # self.peakdata.append(self.fulldata[i] * self.FPS * self.pixel_val)
            self.peakdata.append(self.fulldata[i])
        print("class PeakObj def calc_parameters retrieving time and value data done")

        print("class PeakObj def calc_parameters calculating time from dots")
        self.firsttime = self.first / self.FPS
        self.secondtime = self.fmax / self.FPS
        self.thirdtime = self.min / self.FPS
        self.fourthtime = self.smax / self.FPS
        self.fifthtime = self.last / self.FPS
        if self.time_scale == "ms":
            self.firsttime = self.firsttime * 1000
            self.secondtime = self.secondtime * 1000
            self.thirdtime = self.thirdtime * 1000
            self.fourthtime = self.fourthtime * 1000
            self.fifthtime = self.fifthtime * 1000
            for i,a in enumerate(self.peaktimes):
                self.peaktimes[i] = a * 1000

        print("class PeakObj def calc_parameters calculating time from dots done")
        
        print("class PeakObj def calc_parameters printing max dots and times")
        print("self.fmax")
        print(self.fmax)
        print("self.smax")
        print(self.smax)
        print("self.secondtime")
        print(self.secondtime)
        print("self.fourthtime")
        print(self.fourthtime)
        print("class PeakObj def calc_parameters printing max dots and times done")

        # self.firstvalue = self.fulldata[self.first] * self.FPS * self.pixel_val
        # self.secondvalue = self.fulldata[self.fmax] * self.FPS * self.pixel_val
        # self.thirdvalue = self.fulldata[self.min] * self.FPS * self.pixel_val
        # self.fourthvalue = self.fulldata[self.smax] * self.FPS * self.pixel_val
        # self.fifthvalue = self.fulldata[self.last] * self.FPS * self.pixel_val

        print("class PeakObj def calc_parameters retrieving values for each dot")
        self.firstvalue = self.fulldata[self.first]
        self.secondvalue = self.fulldata[self.fmax]
        self.thirdvalue = self.fulldata[self.min]
        self.fourthvalue = self.fulldata[self.smax]
        self.fifthvalue = self.fulldata[self.last]
        print("class PeakObj def calc_parameters retrieving values for each dot done")
        self.advanced_parameters()

    def advanced_parameters(self):
        print("class PeakObj def calc_parameters calculating interest variables from times and values")
        self.parameters["Contraction-Relaxation Time (CRT)"] = (self.fifthtime - self.firsttime)
        self.parameters["Contraction Time (CT)"] = (self.thirdtime - self.firsttime)
        self.parameters["Relaxation Time (RT)"] = (self.fifthtime - self.thirdtime)
        self.parameters["Contraction time-to-peak (CTP)"] = (self.secondtime - self.firsttime)
        self.parameters["Contraction time from peak to minimum speed (CTPMS)"] = (self.thirdtime - self.secondtime)
        self.parameters["Relaxation time-to-peak (RTP)"] = (self.fourthtime - self.thirdtime)
        self.parameters["Relaxation time from peak to Baseline (RTPB)"] = (self.fifthtime - self.fourthtime)
        self.parameters["Time between Contraction-Relaxation maximum speed (TBC-RMS)"] = (self.fourthtime - self.secondtime)

        print("class PeakObj def calc_parameters printing max difference")
        print("(self.fourthtime - self.secondtime)")
        print((self.fourthtime - self.secondtime))

        print("class PeakObj def calc_parameters printing max difference done")
        self.parameters['Maximum Contraction Speed (MCS)'] = self.secondvalue
        self.parameters['Maximum Relaxation Speed (MRS)'] = self.fourthvalue
        self.parameters['MCS/MRS Difference Speed (MCS/MRS-DS)'] = abs(self.secondvalue - self.fourthvalue)

        # self.parameters['Contraction-Relaxation Area (CRA)'] = np.trapz([a * self.FPS * self.pixel_val for a in self.fulldata[self.first:self.last+1]])
        self.parameters['Contraction-Relaxation Area (CRA)'] = np.trapz([a for a in self.fulldata[self.first:self.last+1]])
        # self.parameters['Shortening Area (SA)'] = np.trapz([a * self.FPS * self.pixel_val for a in self.fulldata[self.min:self.last+1]])
        self.parameters['Shortening Area (SA)'] = np.trapz([a for a in self.fulldata[self.min:self.last+1]])
        print("class PeakObj def calc_parameters calculating interest variables from times and values done")

        print("class PeakObj def calc_parameters formating parameter float")
        for k in self.parameters.keys():
            self.parameters[k] = float("{:.3f}".format(self.parameters[k]))
        print("class PeakObj def calc_parameters formating parameter float done")
        print("class PeakObj def calc_parameters done")

class MoveDragHandler(object):
    """ A simple class to handle Drag n Drop.

    This is a simple example, which works for Text objects only
    """
    def __init__(self, master, currentgroup = None, FPS=None, pixel_val = None, figure=None, ax=None, data=[], selectCMenu=None, areaCMenu=None, areaNMenu=None, colorify = None, plotconf=None, noiseindexes = [], dsizes=(None,None), ax2baseline=None, ax2grid=None, deltafft=None) :
        """ Create a new drag handler and connect it to the figure's event system."""
        self.master = master
        #Figure, Ax, Plot Data, Click Mode, Right Click Menus
        self.thisgroup = currentgroup
        self.figure = figure
        self.ax = ax
        self.ax2 = None
        self.subplotartist = None
        self.data = data.copy()
        self.noiseindexes = noiseindexes.copy()
        # self.mode = "edit"
        self.selectCMenu = selectCMenu
        self.areaCMenu = areaCMenu
        self.areaNMenu = areaNMenu
        self.colorify = colorify
        self.plotconf = plotconf
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.FPS = FPS
        self.pixel_val = pixel_val

        #Plot State variables
        self.current_selax = None
        self.areaopen = False
        self.circle = None
        self.lock = False
        self.lockrect = False
        self.ax2_type = "Disabled"
        self.selectareaopen = False
        self.selectdotarea = None
        self.selectloc = None

        #Area menu rectangle xpos
        self.arearect_x0 = None
        self.arearect_x1 = None

        #Draw Rectangles variables
        self.rect = None
        self.rectx0 = None
        self.recty0 = None
        self.rectx1 = None
        self.recty1 = None
        self.drawnrects = []
        self.background = None
        self.old_pos = None

        #Ax2 state and drawing variables
        self.noiserect = None
        self.noiserectx0 = None
        self.noiserecty0 = None
        self.noiserectx1 = None
        self.noiserecty1 = None
        self.locknoiserect = False
        self.noisearearect_x0 = None
        self.noisearearect_x1 = None
        self.noiseareaopen = False
        self.noisedrawnrects = []
        self.complement_circle = None
        self.complement_ax = None
        self.complement_background = None
        self.user_selected_noise = []
        self.user_removed_noise = []

        #Click, Mouse Move and Release binders
        self.pressmovecid = self.figure.canvas.mpl_connect("button_press_event", self.on_press_event)
        self.motionmovecid = self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion_event)
        self.releasemovecid = self.figure.canvas.mpl_connect("button_release_event", self.on_release_event)

        self.original_dot = dsizes[0]
        self.double_dot = dsizes[1]
        # Connect events and callbacks
        self.mode = "edit"
        self.ax2baseline = ax2baseline
        self.ax2grid = ax2grid

        self.zoomed = False
        self.lockzoom = False
        self.zoomrect = None
        self.zoomrectx0 = None
        self.zoomrecty0 = None
        self.zoomrectx1 = None
        self.zoomrecty1 = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        self.axcurlims = (None, None)
        self.ax2curlims = (None, None)
        self.delta_fft = deltafft
        self.FFTxData = {}
        self.tempresult = False

    def get_rectangles_data(self):
        if not self.drawnrects:
            return None
        else:
            peaks = []
            for r in self.drawnrects:
                rstart = r.get_x()
                rend = r.get_x() + r.get_width()
                # print("rstart, rend")
                # print(rstart, rend)
                smallest_int = int(math.ceil(rstart))
                if smallest_int < 0:
                    smallest_int = 0
                highest_int = int(math.floor(rend))
                if highest_int > len(self.data):
                    highest_int = len(self.data) - 1
                # print("smallest_int, highest_int")
                # print(smallest_int, highest_int)
                #r_curdata = self.data[smallest_int: highest_int+1]
                # print("r_curdata")
                # print(r_curdata)
                #get all dots between data and check if there is first, second, third, fourth and fifth in area
                first_dots = []
                first_dots_x = []
                max_dots = []
                min_dots = []
                last_dots = []
                for child in self.ax.get_children():
                    if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data()[0][0] >= smallest_int and child.get_data()[0][0] <= highest_int:
                        xdata, ydata = child.get_data()
                        if child.pointtype == "first":
                            first_dots_x.append(int(xdata[0]))
                            first_dots.append([xdata[0], ydata[0]])
                        elif child.pointtype == "max":
                            max_dots.append([xdata[0], ydata[0]])
                        elif child.pointtype == "min":
                            min_dots.append([xdata[0], ydata[0]])
                        elif child.pointtype == "last":
                            last_dots.append([xdata[0], ydata[0]])
                #DEBUGGING
                # print("first_dots")
                # print(first_dots)
                # print("max_dots")
                # print(max_dots)
                # print("min_dots")
                # print(min_dots)
                # print("last_dots")
                # print(last_dots)
                if len(first_dots) > 0 and len(max_dots) > 1 and len(min_dots) > 0 and len(last_dots) > 0:
                    for cd in range(smallest_int,highest_int+1):
                        if cd in first_dots_x:
                            fd = first_dots[first_dots_x.index(cd)]
                            ld = sorted([la for la in last_dots if la[0] > fd[0]] )
                            if len(ld) == 0:
                                break
                            ld = ld[0] #1th last point above fd
                            maxes = [mx for mx in max_dots if mx[0] > fd[0] and mx[0] < ld[0]] #all maxes in between
                            mins = [mn for mn in min_dots if mn[0] > fd[0] and mn[0] < ld[0]] #all mins in between
                            mins = sorted(mins, key=lambda x: x[1])
                            mmin = []
                            mmax = max(maxes, key=lambda x: x[1])
                            mmax2 = []
                            #get smallest minimum before and after highest maximum with another maximum after or before minimum
                            for mi in mins:
                                after_max = [m for m in maxes if m[0] > mi[0]]
                                before_max = [m for m in maxes if m[0] < mi[0]]
                                if mi[0] > mmax[0] and len(after_max) > 0:
                                    amax = max(after_max, key=lambda x: x[1])
                                    mmin = mi
                                    mmax2 = amax
                                    break
                                elif mi[0] < mmax[0] and len(before_max) > 0:
                                    bmax = max(before_max, key=lambda x: x[1])
                                    mmin = mi
                                    mmax2 = bmax
                                    break
                            #if all points are get, save area as peak
                            # print("fd, ld, mmin, mmax, mmax2")
                            # print(fd, ld, mmin, mmax, mmax2)
                            # print("fd and ld and mmin and mmax and mmax2")
                            # print(fd and ld and mmin and mmax and mmax2)
                            if fd and ld and mmin and mmax and mmax2:
                                pobj = PeakObj()
                                pobj.first = int(fd[0])
                                pobj.fmax = np.min([int(mmax[0]), int(mmax2[0])])
                                pobj.min = int(mmin[0])
                                pobj.smax = np.max([int(mmax[0]), int(mmax2[0])])
                                pobj.last = int(ld[0])
                                pobj.FPS = self.FPS
                                pobj.time_scale = self.master.current_timescale
                                pobj.pixel_val = self.pixel_val
                                pobj.fulldata = self.data.copy()
                                pobj.calc_parameters()
                                # print("appended pobj to peaks")
                                peaks.append(pobj)
            return peaks

    def set_ax2(self, ax2):
        self.ax2 = ax2
    
    def unset_area_as_noise(self):
        x0 = self.noisearearect_x0
        x1 = self.noisearearect_x1
        smallest_int = int(math.ceil(x0))
        if smallest_int < 0:
            smallest_int = 0
        highest_int = int(math.floor(x1))
        if highest_int > len(self.data):
            highest_int = len(self.data) - 1
        rangenoiserect = [a for a in range(smallest_int, highest_int+1)]
        fulldatarange = [a for a in range(0, len(self.data))]

        #DEBUGGING:
        # print("")
        # print("UNSET START")
        # print("self.user_selected_noise BEFORE")
        # print(self.user_selected_noise)
        self.user_selected_noise = sorted(list(set(self.user_selected_noise.copy()) - set(rangenoiserect)))
        # print("self.user_selected_noise AFTER")
        # print(self.user_selected_noise)
        prenewset = sorted(list(set(self.user_removed_noise.copy()) | set(rangenoiserect)))
        # print("prenewset")
        # print(prenewset)
        # print("self.user_removed_noise BEFORE")
        # print(self.user_removed_noise)
        self.user_removed_noise = sorted(list(set(fulldatarange) & set(prenewset)))
        # print("self.user_removed_noise AFTER")
        # print(self.user_removed_noise)
        # print("UNSET DONE")
        # print("")

    def set_area_as_noise(self):
        x0 = self.noisearearect_x0
        x1 = self.noisearearect_x1
        smallest_int = int(math.ceil(x0))
        if smallest_int < 0:
            smallest_int = 0
        highest_int = int(math.floor(x1))
        if highest_int > len(self.data):
            highest_int = len(self.data) - 1
        rangenoiserect = [a for a in range(smallest_int, highest_int+1)]
        fulldatarange = [a for a in range(0, len(self.data))]
        # print("")
        # print("SET START")
        # print("self.user_removed_noise BEFORE")
        # print(self.user_removed_noise)
        self.user_removed_noise = sorted(list(set(self.user_removed_noise.copy()) - set(rangenoiserect)))
        # print("self.user_removed_noise AFTER")
        # print(self.user_removed_noise)
        prenewset = sorted(list(set(self.user_selected_noise.copy()) | set(rangenoiserect)))
        # print("prenewset")
        # print(prenewset)
        # print("self.user_selected_noise BEFORE")
        # print(self.user_selected_noise)
        self.user_selected_noise = sorted(list(set(fulldatarange) & set(prenewset)))
        # print("self.user_selected_noise AFTER")
        # print(self.user_selected_noise)
        # print("SET DONE")
        # print("")

    def example_function(self, event=None):
        print("Hi!")   
    
    def reset_modes(self):
        print("reset modes")
        print('on_press disconnected (cid='+str(self.pressmovecid)+')')
        self.figure.canvas.mpl_disconnect(self.pressmovecid)
        self.figure.canvas.mpl_disconnect(self.motionmovecid)
        self.figure.canvas.mpl_disconnect(self.releasemovecid)
        self.pressmovecid = None
        self.motionmovecid = None
        self.releasemovecid = None
        self.mode = None
        self.figure.canvas.draw()

    def reconnect_modes(self):
        self.pressmovecid = self.figure.canvas.mpl_connect("button_press_event", self.on_press_event)
        self.motionmovecid = self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion_event)
        self.releasemovecid = self.figure.canvas.mpl_connect("button_release_event", self.on_release_event)
        self.figure.canvas.draw()

    def on_press_event(self, event):
        print("on_press_event")
        print("event")
        print(event.inaxes)
        print(self.ax)
        print(event.inaxes == self.ax)
        print(self.ax2)
        print(event.inaxes == self.ax2)
        if self.mode == "edit" and event.inaxes == self.ax:
            self.on_press_buffer(event)
            print("")
            print("")
            return
        elif self.mode == "point" and event.inaxes == self.ax:
            self.on_press_moveevent(event, event.inaxes)
            print("")
            print("")
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #Function: determine area as noise
            self.on_press_ax2noise(event)
            return

        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #Function: move points on both axes
            print(self.mode + " " + self.ax2_type)
            self.on_press_moveevent(event, event.inaxes)
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #Function: edit plot title for selected frequency
            print(self.mode + " " + self.ax2_type)
            self.on_press_ax2fft(event)
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #No function
            print(self.mode + " " + self.ax2_type)
            return
        
        elif self.mode == "zoom" and event.inaxes == self.ax:
            self.on_press_zoom(event)
            return
        elif self.mode == "zoom" and event.inaxes == self.ax2:
            self.on_press_zoom(event)
            return

        return

    def on_motion_event(self, event):
        if self.mode == "edit" and event.inaxes == self.ax:
            self.on_motion_buffer(event)
            return
        elif self.mode == "point" and event.inaxes == self.ax:
            #TODO: Add tooltip
            #https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
            # tooltipselected = None
            # for child in self.ax.get_children():
            #     if isinstance(child, Line2D) and child.contains(event)[0] and child.get_marker() == "o":
            #         tooltipselected = child
            #         break
            # if tooltipselected:
            #     cont, ind = tooltipselected.contains(event)
            #     pos = sc.get_offsets()[ind["ind"][0]]
            #     self.annot.xy = pos
            #     text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
            #                            " ".join([names[n] for n in ind["ind"]]))
            #     self.annot.set_text(text)
            #     self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            #     self.annot.get_bbox_patch().set_alpha(0.4)
            #     self.annot.set_visible(True)
            self.on_motion_moveevent(event)
            return


        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #Function: determine area as noise
            self.on_motion_ax2noise(event)
            return


        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #Function: move points on both axes
            print(self.mode + " " + self.ax2_type)
            self.on_motion_moveevent(event)
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #Function: edit plot title for selected frequency, none on motion
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #No function
            print(self.mode + " " + self.ax2_type)
            return
            return
        
        elif self.mode == "zoom" and event.inaxes == self.ax:
            self.on_motion_zoom(event)
            return
        elif self.mode == "zoom" and event.inaxes == self.ax2:
            self.on_motion_zoom(event)
            return
        
        return

    def on_release_event(self, event):
        print("on_release_event")
        if self.mode == "edit" and event.inaxes == self.ax:
            self.on_release_buffer(event)
            return
        elif self.mode == "point" and event.inaxes == self.ax:
            self.on_release_moveevent(event)
            return


        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #No function should be done
            return
        elif self.mode == "edit" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #Function: determine area as noise
            self.on_release_ax2noise(event)
            return

        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "Zoom":
            #Function: move points on both axes
            print(self.mode + " " + self.ax2_type)
            self.on_release_moveevent(event)
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "FFT":
            #Function: edit plot title for selected frequency
            # self.on_release_ax2fft(event)
            # print(self.mode + " " + self.ax2_type)
            return
        elif self.mode == "point" and event.inaxes == self.ax2 and self.ax2_type == "PeakNoise":
            #No function
            print(self.mode + " " + self.ax2_type)
            return
            return
        
        elif self.mode == "zoom" and event.inaxes == self.ax:
            self.on_release_zoom(event)
            return
        elif self.mode == "zoom" and event.inaxes == self.ax2:
            self.on_release_zoom(event)
            return
        
        return

    def add_dot_at_last(self, newtype):
        if self.selectdotarea == None and self.selectloc != None:
            newdot = self.ax.plot(self.selectloc[0], self.selectloc[1], "o", linewidth=2, fillstyle='none', color=self.colorify[newtype])
            newdot[0].pointtype = newtype
            if self.ax2_type == "Zoom":
                xlim = self.ax2.get_xlim()
                ylim = self.ax2.get_ylim()
                newdot2 = self.ax2.plot(self.selectloc[0], self.selectloc[1], "o", linewidth=2, fillstyle='none', color=self.colorify[newtype])
                newdot2[0].pointtype = newtype
                self.ax2.set_xlim(xlim)
                self.ax2.set_ylim(ylim)
            #add dot to f_points list of master
            #"first", "max", "min", "last"
            self.selectareaopen = False
            self.selectdotarea = None
            self.selectloc = None
            self.tempresult = False
            self.figure.canvas.draw()
            messagebox.showinfo("Dot Added", "Dot added successfully!")
            return True
        else:
            print("self.selectdotarea")
            print(self.selectdotarea)
            print("self.selectloc")
            print(self.selectloc)
            self.selectdotarea = None
            self.selectloc = None
            self.tempresult = False
            messagebox.showerror("Error", "Dot already exists at this position")
            return False
        self.selectdotarea = None
        self.selectloc = None
        return False


    def change_dot_at_last(self, newtype):
        if self.selectdotarea != None:
            self.selectdotarea.pointtype = newtype
            self.selectdotarea.set_color(self.colorify[newtype])
            complement_ax = self.ax
            complement_selected = None
            if self.ax2_type == "Zoom" and self.selectdotarea.axes == self.ax:
                complement_ax = self.ax2
            if self.ax2_type == "Zoom":
                for child in complement_ax.get_children():
                    if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data() == self.selectdotarea.get_data():
                        complement_selected = child
                        break
                if complement_selected != None:
                    complement_selected.pointtype = newtype
                    complement_selected.set_color(self.colorify[newtype])
            self.selectdotarea = None
            self.selectloc = None
            self.figure.canvas.draw()
            messagebox.showinfo("Dot Added", "Dot type changed successfully!")
            return True
        else:
            messagebox.showerror("Error", "No dots were clicked!")
            self.selectdotarea = None
            self.selectloc = None
            return False
        self.selectdotarea = None
        self.selectloc = None
        return False


    def remove_dot_at_last(self):
        if self.selectdotarea != None:
            complement_ax = self.ax
            complement_selected = None
            if self.ax2_type == "Zoom" and self.selectdotarea.axes == self.ax:
                complement_ax = self.ax2
            if self.ax2_type == "Zoom":
                for child in complement_ax.get_children():
                    if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data() == self.selectdotarea.get_data():
                        complement_selected = child
                        break
                if complement_selected != None:
                    complement_selected.remove()
            self.selectdotarea.remove()
            self.selectdotarea = None
            self.selectloc = None
            self.figure.canvas.draw()
            messagebox.showinfo("Dot Removed", "Dot removed successfully!")
            return True
        else:
            messagebox.showerror("Error", "No dots were clicked!")
            self.selectdotarea = None
            self.selectloc = None
            return False
        self.selectdotarea = None
        self.selectloc = None
        return False


    def on_press_moveevent(self, event, thisax):
        print("")
        print("on_press_moveevent")
        print("event print")
        print(event)
        print("self.selectareaopen")
        print(self.selectareaopen)
        if event.button == 3 and event.dblclick == False and self.selectareaopen == False:
            #check for circle clicked and save artist if so
            print("inside first")
            selected = None
            
            arrayz = np.asarray(list(range(len(self.data))))
            idx = (np.abs(arrayz - event.xdata)).argmin()
            self.selectloc = (int(idx), self.data[int(idx)])
            # print("event.xdata")
            # print(event.xdata)
            # print("idx")
            # print(idx)
            # print("self.selectloc")
            # print(self.selectloc)
            for child in thisax.get_children():
                if isinstance(child, Line2D) and child.contains(event)[0] and child.get_marker() == "o":
                    selected = child
                    if int(idx)  == int(child.get_data()[0][0]):
                        self.selectloc = None
                    break
            #open menu for move event
            print("selected")
            print(selected)
            self.selectdotarea = selected
            try:
                self.selectareaopen = True
                #x = self.master.winfo_pointerx()
                #y = self.master.winfo_pointery()
                abs_coord_x = self.master.winfo_pointerx() - self.master.winfo_vrootx()
                abs_coord_y = self.master.winfo_pointery() - self.master.winfo_vrooty()
                #self.master.currentpopup = self.selectCMenu
                self.master.currentpopup = self.master.current_frame.selectCMenu
                self.master.popuplock = True
                #self.selectCMenu.tk_popup(int(abs_coord_x + (self.selectCMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
                self.master.current_frame.selectCMenu.tk_popup(int(abs_coord_x + (self.master.current_frame.selectCMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
            finally:
                print("finally has run for selectCMenu")
                self.master.current_frame.selectCMenu.grab_release()
                if self.master.currentpopup is None:
                    print("popup does not even exist anymore")
                    if self.selectareaopen == True:
                       print("unsetting selectareaopen")
                       self.selectareaopen = False
            return
        elif event.button == 1 and event.dblclick == False and self.areaopen == False:
            selected = None
            # for child in self.ax.get_children():
                # if isinstance(child, Line2D) and child.contains(event)[0] and child.get_marker() == "o":
                    # selected = child
                    # break

            for child in thisax.get_children():
                if isinstance(child, Line2D) and child.contains(event)[0] and child.get_marker() == "o":
                    selected = child
                    break
            if selected == None: return
            self.circle = selected
            # self.circle.set_markersize(self.circle.get_markersize() * 2)
            self.circle.set_markersize(self.double_dot)

            self.complement_circle = None
            complement_selected = None
            self.complement_ax = None

            if self.ax2_type == "Zoom" and thisax == self.ax:
                self.complement_ax  = self.ax2
                for child in self.complement_ax.get_children():
                    if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data() == selected.get_data():
                        complement_selected = child
                        break
            elif self.ax2_type == "Zoom" and thisax == self.ax2:
                self.complement_ax  = self.ax
                for child in self.complement_ax.get_children():
                    if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data() == selected.get_data():
                        complement_selected = child
                        break
            if complement_selected != None:
                self.complement_circle = complement_selected
                # self.complement_circle.set_markersize(self.complement_circle.get_markersize() * 2)
                self.complement_circle.set_markersize(self.double_dot)

            xdata, ydata = self.circle.get_data()
            # print('event contains', self.circle.xy)
            x0, y0 = xdata[0], ydata[0]
            # x0, y0 = self.circle.get_data()
            self.press = x0, y0, event.xdata, event.ydata

            self.lock = self.circle

            # draw everything but the selected circleangle and store the pixel buffer
            canvas = self.figure.canvas
            # axes = self.ax
            self.current_selax = thisax
            # axes = thisax

            self.circle.set_animated(True)

            if self.complement_circle:
                self.complement_circle.set_animated(True)

            canvas.draw()

            # self.background = canvas.copy_from_bbox(self.ax.bbox)
            self.background = canvas.copy_from_bbox(self.current_selax.bbox)

            self.complement_background = None
            if self.complement_circle:
                self.complement_background = canvas.copy_from_bbox(self.complement_ax.bbox)
            # now redraw just the circleangle
            # axes.draw_artist(self.circle)
            self.current_selax.draw_artist(self.circle)
            
            if self.complement_circle:
                self.complement_ax.draw_artist(self.complement_circle)

            # and blit just the redrawn area
            # canvas.blit(axes.bbox)
            canvas.blit(self.current_selax.bbox)

            if self.complement_circle:
                canvas.blit(self.complement_ax.bbox)
        return

    def on_motion_moveevent(self, event):
        # 'on motion we will move the circle if the mouse is over us'
        if self.lock is not self.circle:
            return
        print("on_motion_moveevent")
        # print("event.inaxes")
        # print(event.inaxes)
        # print("self.ax.lims")
        # print(self.ax.get_xlim())
        # print(self.ax.get_ylim())
        # print("event.xdata")
        # print(event.xdata)
        # print("event.ydata")
        # print(event.ydata)
        if event.inaxes != self.current_selax:
            print("1 NOT IN AXES!!!")
            print("2 NOT IN AXES!!!")
            self.on_release_moveevent(event)
            return
        # if event.inaxes != self.circle.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.circle.set_xdata([x0+dx])
        self.circle.set_ydata([y0+dy])
        if self.complement_circle:
            #check if outofbounds for ax2 and remove dot if so
            self.complement_circle.set_xdata([x0+dx])
            self.complement_circle.set_ydata([y0+dy])
        self.old_pos = x0,y0

        canvas = self.figure.canvas
        # axes = self.ax

        # restore the background region
        canvas.restore_region(self.background)
        if self.complement_circle:
            canvas.restore_region(self.complement_background)

        # redraw just the current circleangle
        # axes.draw_artist(self.circle)
        self.current_selax.draw_artist(self.circle)
        
        if self.complement_circle:
            self.complement_ax.draw_artist(self.complement_circle)

        # blit just the redrawn area
        # canvas.blit(axes.bbox)
        canvas.blit(self.current_selax.bbox)
        if self.complement_circle:
            canvas.blit(self.complement_ax.bbox)
        return

    def on_release_moveevent(self, event):
        print("on_release_moveevent")
        # 'on release we reset the press data'

        if self.lock is not self.circle:
            return

        new_pos_x = int(self.circle.get_xdata()[0])
        if abs(self.circle.get_xdata()[0] - math.ceil(self.circle.get_xdata()[0])) < abs(self.circle.get_xdata()[0] - math.floor(self.circle.get_xdata()[0])):
            new_pos_x = int(math.ceil(self.circle.get_xdata()[0]))
        else:
            new_pos_x = int(math.floor(self.circle.get_xdata()[0]))
        if new_pos_x > 0 and new_pos_x < len(self.data):
            new_pos_y = self.data[new_pos_x]
        else:
            new_pos_x = self.old_pos[0]
            new_pos_y = self.old_pos[1]
        #new_pos = (new_pos_x, new_pos_y)

        self.circle.set_xdata([int(new_pos_x)])
        self.circle.set_ydata([new_pos_y])
        # self.circle.set_markersize(self.circle.get_markersize() / 2)
        self.circle.set_markersize(self.original_dot)

        if self.complement_circle:
            self.complement_circle.set_xdata([int(new_pos_x)])
            self.complement_circle.set_ydata([new_pos_y])
            # self.complement_circle.set_markersize(self.complement_circle.get_markersize() / 2)
            self.complement_circle.set_markersize(self.original_dot)
            # if self.ax2_type == "Zoom" and self.current_selax != self.ax2:
            #     #check if outofbounds for ax2 and remove dot if so
            #     print("self.ax2.get_xlim()")
            #     print(self.ax2.get_xlim())
            #     xl0 = self.ax2.get_xlim()[0]
            #     xll = self.ax2.get_xlim()[1]
            #     print("self.ax2.get_ylim()")
            #     print(self.ax2.get_ylim())
            #     yl0 = self.ax2.get_ylim()[0]
            #     yll = self.ax2.get_ylim()[1]
            #     if int(new_pos_x) < xl0 or int(new_pos_x) > xll or new_pos_y < yl0 or new_pos_y > yll:
            #         print("REMOVED!!!!")
            #         self.complement_circle.remove()

        # if self.ax2_type == "Zoom" or self.ax2_type == "FFT" or self.ax2_type=="PeakNoise":
            # self.drawAx2()
        # if self.ax2 and if self.current_selax == self.ax2:
            #get circle and redraw on ax1
            # self.redrawCircle(self.circle, self.ax)
        

        self.press = None
        self.lock = None
        self.old_pos = None
        self.current_selax = None

        # turn off the circle animation property and reset the background
        self.circle.set_animated(False)
        self.background = None

        if self.complement_circle:
            self.complement_circle.set_animated(False)
            self.complement_background = None

        # redraw the full figure
        self.figure.canvas.draw()
        return
    
    def on_press_buffer(self, event):
        # print("on_press_buffer")
        if event.inaxes != self.ax: return
        # print("on_press_buffer in ax")
        # print("Condition 1:")
        # print(self.lockrect)
        # print("event")
        # print(event)
        if event.button == 1 and event.dblclick == False and self.lockrect == False and event.xdata != None and event.ydata != None:
            print("on_press_buffer start rect")
            self.lockrect = True
            self.rectx0 = event.xdata
            return
        # if event.button == 1 and event.dblclick == False and self.lockrect == True and self.areaopen == True:
        # #     #this releases context menu on any single click inside or outside plot
        #     self.areaCMenu.unpost()
        #     self.lockrect = False
        #     self.areaopen = False
        #     return
        if event.button == 1 and event.dblclick == True and self.areaopen == False:
            #check if any rect on double click area and delete accordingly
            print("on_press_buffer double clicking")
            self.lockrect = False
            self.rectx0 = None
            # for i in range(len(self.drawnrects)):
            #DEBUGGING:
            # print("self.drawnrects")
            # print(self.drawnrects)
            # print("event.xdata")
            # print(event.xdata)
            # print()
            if len(self.drawnrects) > 0:
                for i in range(len(self.drawnrects)-1, -1, -1):
                    r = self.drawnrects[i]
                    rstart = r.get_x()
                    rend = r.get_x() + r.get_width()
                    if event.xdata >= rstart and event.xdata <= rend:
                        self.drawnrects[i].remove()
                        del self.drawnrects[i]
                        self.figure.canvas.draw()
                        # break
                if self.ax2_type == "Zoom" or self.ax2_type == "FFT":
                    self.drawAx2()
            return
        if event.button == 2:
            #mouse scroll event, at the moment does nothing
            self.lockrect = False
            self.rectx0 = None
            return
        if event.button == 3 and event.dblclick == False and self.areaopen == False:
            print("on_press_buffer right click and not double click")
            #check if any rect on double click area and open context menu if inside
            insiderect = False
            for r in self.drawnrects:
                rstart = r.get_x()
                rend = r.get_x() + r.get_width()
                if event.xdata >= rstart and event.xdata <= rend:
                    self.arearect_x0 = int(math.ceil(rstart))
                    self.arearect_x1 = int(math.floor(rend))
                    insiderect = True
                    break
            if insiderect:
                self.areaopen = True
                self.rectx0 = None
                try:
                    #x = self.master.winfo_pointerx()
                    #y = self.master.winfo_pointery()
                    abs_coord_x = self.master.winfo_pointerx() - self.master.winfo_vrootx()
                    abs_coord_y = self.master.winfo_pointery() - self.master.winfo_vrooty()
                    #self.master.currentpopup = self.areaCMenu
                    self.master.currentpopup = self.master.current_frame.areaCMenu
                    self.master.popuplock = True
                    #self.areaCMenu.tk_popup(int(abs_coord_x + (self.areaCMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
                    self.master.current_frame.areaCMenu.tk_popup(int(abs_coord_x + (self.master.current_frame.areaCMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
                finally:
                    print("areaCMenu finally")
                    #self.areaCMenu.grab_release()
                    self.master.current_frame.areaCMenu.grab_release()
                    self.lockrect = True
            # print("")
            # print("")
            return
        # print("")
        # print("")
        return
    
    def on_motion_buffer(self, event):
        if event.inaxes != self.ax: return
        # print("on_motion_buffer in ax")
        # print("condition 1:")
        # print(self.lockrect)
        # print("Condition init self.rectx0:")
        # print(self.rectx0)
        # print("event")
        # print(event)
        # print("event.xdata")
        # print(event.xdata)
        # print("event.ydata")
        # print(event.ydata)
        if self.lockrect == True and event.xdata != None and event.ydata != None and self.rectx0 != None:
            # print("on_motion_buffer")
            self.rectx1 = event.xdata
            # self.recty1 = event.ydata

            curlims = (self.ax.get_xlim(), self.ax.get_ylim())

            new_rect = Rectangle((0,np.min(self.data)), 1, 1)
            new_rect.set_facecolor(self.colorify['rect_color'])    
            new_rect.set_width(self.rectx1 - self.rectx0)
            
            # nheight = np.max(self.data) + abs(np.min(self.data)) + 0.5
            nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5

            new_rect.set_height(nheight)
            
            # new_rect.set_xy((self.rectx0, np.min(self.data)))
            new_rect.set_xy((self.rectx0, curlims[1][0]))

            if self.rect:
                self.rect.remove()
            self.rect = None
            self.rect = self.ax.add_patch(new_rect)

            self.ax.set_xlim(curlims[0])
            self.ax.set_ylim(curlims[1])

            self.figure.canvas.draw()
        # print("")
        # print("")
        return
    
    def on_release_buffer(self, event):
        print("on_release_buffer start")
        if event.inaxes != self.ax: return
        # print("on_release_buffer in ax")
        # print("condition 1:")
        # print(self.lockrect)
        # print("Condition init self.rectx0:")
        # print(self.rectx0)
        # print("event")
        # print(event)
        # print("event.xdata")
        # print(event.xdata)
        # print("event.ydata")
        # print(event.ydata)
        if self.lockrect == True and event.xdata != None and event.ydata != None and self.rectx0 != None:
            # print("on_release_buffer start")
            self.rectx1 = event.xdata
            if event.xdata < self.rectx0:
                nr1 = self.rectx0 + 0.0
                self.rectx0 = event.xdata
                self.rectx1 = nr1
            self.recty1 = event.ydata
            if abs(self.rectx1 - self.rectx0) > 1:
                #check if any rectangle on area and merge both rectangles if they exist

                curlims = (self.ax.get_xlim(), self.ax.get_ylim())


                new_rect = Rectangle((0,np.min(self.data)), 1, 1)
                new_rect.set_facecolor(self.colorify['rect_color'])
                new_rect.set_width(self.rectx1 - self.rectx0)
                
                # nheight = np.max(self.data) + abs(np.min(self.data)) + 0.5
                nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5
                new_rect.set_height(nheight)
                # new_rect.set_xy((self.rectx0, np.min(self.data)))
                new_rect.set_xy((self.rectx0, curlims[1][0]))

                nrectstart = new_rect.get_x()
                nrectend = new_rect.get_x() + new_rect.get_width()

                nstarts = [nrectstart]
                nends = [nrectend]
                i = 0
                redraw = []
                if len(self.drawnrects) >= 1:
                    for i in range(len(self.drawnrects)):
                        r = self.drawnrects[i]
                        rstart = r.get_x()
                        rend = r.get_x() + r.get_width()
                        if (nrectstart >= rstart and nrectstart <= rend) or (nrectend >= rstart and nrectend <= rend) or (nrectstart <= rstart and nrectend >= rend):
                            nstarts.append(rstart)
                            nends.append(rend)
                            try:
                                self.drawnrects[i].remove()
                            except Exception as e:
                                print("### PLEASE SEND ME TO DEV ###")
                                print(e)
                                print("### PLEASE SEND ME TO DEV ###")

                        else:
                            redraw.append(self.drawnrects[i])
                        i += 1
                self.drawnrects = None
                self.drawnrects = redraw.copy()

                if self.rect:
                    self.rect.remove()
                self.rect = None
                
                smallest = np.min(nstarts)
                highest = np.max(nends)
                new_rect.set_width(highest - smallest)
                
                # new_rect.set_xy((smallest, np.min(self.data)))
                new_rect.set_xy((smallest, curlims[1][0]))

                #save rect to rects
                self.rect = self.ax.add_patch(new_rect)
                # if not overlap:
                self.drawnrects.append(self.rect)
                #if zoom or fft area, draw data on subplot area (ax2)
                if self.ax2_type == "Zoom" or self.ax2_type == "FFT" or self.ax2_type=="PeakNoise":
                    self.drawAx2()
                self.rect = None

                self.ax.set_xlim(curlims[0])
                self.ax.set_ylim(curlims[1])

                self.figure.canvas.draw()
                self.lockrect = False
        # print("")
        # print("")
        return

    def drawAx2(self):
        print("def drawAx2 started")
        self.ax2.clear()
        self.subplotartist = None
        self.ax2.set_title("")
        self.ax2.set_xlabel("")
        self.ax2.set_ylabel("")
        print("def drawAx2 cleared ax2")
        print("def drawAx2 type is: " + self.ax2_type)
        if self.ax2_type == "Zoom":
            print("def drawAx2 rectangle selection about to check")
            curlims = None
            self.ax2.set_xlabel("Time ("+ self.master.current_timescale +")")
            self.ax2.set_ylabel("Average Speed ("+ self.master.current_speedscale+")")
            if len(self.drawnrects) > 0:
                print("def drawAx2 rectangle selection exist")
                rect_to = self.drawnrects[-1]
                smallest_int = int(math.ceil(rect_to.get_x()))
                if smallest_int < 0:
                    smallest_int = 0
                highest_int = int(math.floor(rect_to.get_x() + rect_to.get_width()))
                if highest_int > len(self.data):
                    highest_int = len(self.data) - 1
                print("def drawAx2 smallest_int")
                print(smallest_int)
                print("def drawAx2 highest_int (range is + 1)")
                print(highest_int)
                ax2_data = self.data[smallest_int: highest_int+1]
                
                print("def drawAx2 len ax2_data")
                print(len(ax2_data))

                #here data is plotted just to jet x and y lims for subplotdata
                self.subplotartist = self.ax2.plot(range(smallest_int, highest_int+1) ,ax2_data, color=self.colorify["main"])
                curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())
                
                # fake_plot_data is plotted to ax2 but not saved in any variable for exporting if data has only a single dot (self.subplotartist)
                if len(ax2_data) == 1:
                    print("def drawAx2 about to gen fake_plot_data")
                    next_smallest_int = np.max([0, smallest_int-1])
                    print("def drawAx2 next_smallest_int")
                    print(next_smallest_int)
                    next_highest_int = np.min([len(self.data)-1, highest_int+1])
                    print("def drawAx2 next_highest_int")
                    print(next_highest_int)
                    fake_plot_data = self.data[next_smallest_int:next_highest_int+1]
                    print("def drawAx2 fake_plot_data")
                    print(fake_plot_data)
                    self.ax2.plot(range(next_smallest_int, next_highest_int+1), fake_plot_data, color=self.colorify["main"])
                    print("def drawAx2 fake_plot_data plotted")

                print("def drawAx2 Zoom curlims: ")
                print(curlims)

                print("def drawAx2 adding ax dots to ax2 plot start")
                #get all dots between data and add them to plot
                for child in self.ax.get_children():
                    # if isinstance(child, Line2D) and child.get_marker() == "o" and child.get_data()[0][0] > smallest_int and child.get_data()[0][0] < highest_int:
                    if isinstance(child, Line2D) and child.get_marker() == "o":
                        xdata, ydata = child.get_data()
                        self.ax2.plot(child.get_data()[0][0], child.get_data()[1][0], "o", linewidth=2, fillstyle='none', color=child.get_color())
                        # self.ax2.plot(child.get_data()[0][0], child.get_data()[1][0], "o", color=child.get_color())
                print("def drawAx2 adding ax dots to ax2 plot done")
                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])
                print("def drawAx2 plot x,y limits set to current user selection")

                #update dot visibility
                print("def drawAx2 plotDots about to be run on master")
                self.master.current_frame.plotDots()
                print("def drawAx2 plotDots done on master")
                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])

                self.master.current_frame.plotRegressions()
                self.master.current_frame.plotMeanNoise()
                self.master.current_frame.plotMaxfiltering()
                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])

                self.figure.canvas.draw()
                print("def drawAx2 canvas updated")

                print("def drawAx2 initial labels:")
                print([item.get_text().replace("−", "-") for item in self.ax2.get_xticklabels()])

                if self.master.current_timescale == "s":
                    # labels = [float(item.get_text().replace("−", "-")) / self.FPS for item in self.ax2.get_xticklabels()]
                    labels = [ float("{:.3f}".format(float(item.get_text().replace("−", "-")) / self.FPS)) for item in self.ax2.get_xticklabels() ]

                elif self.master.current_timescale == "ms":
                    # labels = [(float(item.get_text().replace("−", "-")) / self.FPS)*1000 for item in self.ax2.get_xticklabels()]
                    labels = [ float("{:.3f}".format((float(item.get_text().replace("−", "-")) / self.FPS)*1000)) for item in self.ax2.get_xticklabels() ]

                print("def drawAx2 about to update labels:")
                self.ax2.set_xticklabels(labels)
                print("def drawAx2 update labels done")
            print("def drawAx2 drawing baseline and grid if user selected so start")
            if self.ax2baseline != None:
                self.ax2baseline.remove()
                self.ax2baseline = None
            if self.plotconf["zero"] == True:
                self.ax2baseline = self.ax2.axhline(y=0.0, color=self.plotconf["zero_color"], linestyle='-')

            if self.ax2grid != None:
                self.ax2grid.remove()
                self.ax2grid = None
            if self.plotconf["grid"] == True:
                self.ax2grid = self.ax2.grid(linestyle="-", color=self.plotconf["grid_color"], alpha=0.5)
            else:
                self.ax2.grid(False)
            print("def drawAx2 drawing baseline and grid if user selected so done")
            if curlims:
                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])
            self.figure.canvas.draw()
            print("def drawAx2 update canvas")
            print("def drawAx2 done")
        elif self.ax2_type == "FFT":
            smallest_int = 0
            highest_int = len(self.data)
            if len(self.drawnrects) > 0:
                rect_to = self.drawnrects[-1]
                smallest_int = int(math.ceil(rect_to.get_x()))
                if smallest_int < 0:
                    smallest_int = 0
                highest_int = int(math.floor(rect_to.get_x() + rect_to.get_width()))
                if highest_int > len(self.data):
                    highest_int = len(self.data) - 1
            ax2_data = self.data[smallest_int: highest_int+1]
            ax2_data = np.array(ax2_data)
            F = np.fft.fft(ax2_data, norm = "ortho")
            Fabs=np.abs(F)
            AmpFFT=np.sqrt(Fabs)
            # AmpFFT=Fabs
            
            t = np.linspace(0, len(ax2_data), len(ax2_data), endpoint=True)
            

            cut_fft = int(len(AmpFFT) * 0.3)
            if self.delta_fft == None:
                self.delta_fft = np.mean(AmpFFT[:cut_fft])
            self.delta_fft = float("{:.3f}".format(self.delta_fft))

            # delta is average from detected maximum values to closest baseline minimum / 2
            maxtab, mintab = peakdet(AmpFFT, self.delta_fft)

            # freq = np.fft.fftfreq(ax2_data.shape[-1])
            freq = np.fft.fftfreq(ax2_data.shape[-1], d=t[1])
            freq = [f * self.FPS for f in freq]
            # maxes = argrelextrema(AmpFFT, np.greater)
            maxes = maxtab.copy()
            if len(maxes) < 2:
                maxes = argrelextrema(AmpFFT, np.greater)

            maxes_x = []
            maxes_x = [freq[i] for i in maxes[1:]]
            maxes_y = []
            maxes_y = [AmpFFT[i] for i in maxes[1:]]
            if len(maxes_x) > 0 and len(maxes_y) > 0 and self.master.current_frame.plotFFTAll == False:
                maxes_x = [maxes_x[self.master.current_frame.plotFFTSelection]]
                maxes_y = [maxes_y[self.master.current_frame.plotFFTSelection]]
            elif len(maxes_x) == 0 or len(maxes_y) == 0:
                maxes_x = []
                maxes_y = []

            self.ax2.set_xlabel("Frequency (Hz)");
            self.ax2.set_ylabel("Amplitude Density");
            self.subplotartist = self.ax2.plot(freq, AmpFFT, color=self.colorify["main"])

            fdot = False
            maxesi = 0
            self.FFTxData = {}
            print("##")
            print("self.master.current_frame.plotFFTSelection")
            print(self.master.current_frame.plotFFTSelection)
            print("##")
            for x, y in zip(maxes_x, maxes_y):
                curc = self.colorify["fft"]
                # if fdot == False:
                if maxesi == self.master.current_frame.plotFFTSelection and self.master.current_frame.plotFFTAll == True:
                    curc = self.colorify["fft_selection"]                 
                    fdot = True
                elif fdot == False and self.master.current_frame.plotFFTAll == False:
                    curc = self.colorify["fft_selection"]                    
                    fdot = True
                # dot = self.ax2.plot(x, y, "o", color=curc, picker=1)
                self.FFTxData[x] = maxesi
                dot = self.ax2.plot(x, y, "o", linewidth=2, fillstyle='none', color=curc, picker=3)
                maxesi += 1
            
            if len(maxes_x) > 0 and self.master.current_frame.plotFFTAll == True:
                self.ax2.set_title("Selected Wave Frequency: " + "{:.3f}".format(maxes_x[self.master.current_frame.plotFFTSelection]) )
            elif len(maxes_x) > 0:
                self.ax2.set_title("Selected Wave Frequency: " + "{:.3f}".format(maxes_x[0]) )
            else:
                self.ax2.set_title("Selected Wave Frequency: None")

            # self.ax2.plot(maxes_x, maxes_y, "o", color=self.colorify["fft"])
            if self.ax2grid != None:
                self.ax2grid.remove()
                self.ax2grid = None
            if self.plotconf["grid"] == True:
                self.ax2grid = self.ax2.grid(linestyle="-", color=self.plotconf["grid_color"], alpha=0.5)
            else:
                self.ax2.grid(False)

            xlim_end = self.ax2.get_xlim()[1]
            self.ax2.set_xlim(0, xlim_end)

            #update dot visibility
            self.master.current_frame.plotDots()

            self.figure.canvas.draw()            

        elif self.ax2_type == "PeakNoise":
            self.ax2.set_xlabel("Time ("+ self.master.current_timescale +")")
            self.ax2.set_ylabel("Average Speed ("+ self.master.current_speedscale+")")

            # if self.ax2baseline != None:
            #     self.ax2baseline.remove()
            #     self.ax2baseline = None
            # if self.plotconf["zero"] == True:
            #     self.ax2baseline = self.ax2.axhline(y=0.0, color=self.plotconf["zero_color"], linestyle='-')

            # if self.ax2grid != None:
            #     self.ax2grid.remove()
            #     self.ax2grid = None
            # if self.plotconf["grid"] == True:
            #     self.ax2grid = self.ax2.rc('grid', linestyle="-", color=self.plotconf["grid_color"])
            #TODO DEBUG ARRAY LENGTH
            
            smallest_int = 0
            highest_int = len(self.data)

            if len(self.drawnrects) > 0:
                rect_to = self.drawnrects[-1]
                smallest_int = int(math.ceil(rect_to.get_x()))
                highest_int = int(math.floor(rect_to.get_x() + rect_to.get_width()))

            if smallest_int < 0:
                smallest_int = 0

            if highest_int >= len(self.data):
                highest_int = len(self.data) - 1
            
            ax2_data = self.data[smallest_int: highest_int+1]

            #DEBUGGING:

            self.subplotartist = self.ax2.plot(range(smallest_int, highest_int+1) ,ax2_data, color=self.colorify["main"])
            curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())
            # fake_plot_data is plotted to ax2 but not saved in any variable for exporting if data has only a single dot (self.subplotartist)
            if len(ax2_data) == 1:
                print("def drawAx2 about to gen fake_plot_data")
                next_smallest_int = np.max([0, smallest_int-1])
                print("def drawAx2 next_smallest_int")
                print(next_smallest_int)
                next_highest_int = np.min([len(self.data)-1, highest_int+1])
                print("def drawAx2 next_highest_int")
                print(next_highest_int)
                fake_plot_data = self.data[next_smallest_int:next_highest_int+1]
                print("def drawAx2 fake_plot_data")
                print(fake_plot_data)
                self.ax2.plot(range(next_smallest_int, next_highest_int+1), fake_plot_data, color=self.colorify["main"])
                print("def drawAx2 fake_plot_data plotted")
            curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())

            newlydrawnrects = []
            for erect in self.noisedrawnrects:
                esmallest_int = int(math.ceil(erect.get_x()))
                if esmallest_int < 0:
                    esmallest_int = 0
                ehighest_int = int(math.floor(erect.get_x() + erect.get_width()))
                if ehighest_int > len(self.data):
                    ehighest_int = len(self.data) - 1
                if esmallest_int >= smallest_int and ehighest_int <= highest_int:
                    #patches are added according to zoom level
                    nrect = Rectangle((0,np.min(self.data)), 1, 1)
                    nrect.set_facecolor(self.colorify['rect_color'])
                    nrect.set_width(erect.get_width())
                    nrect.set_height(erect.get_height())
                    nrect.set_xy(erect.get_xy())
                    self.ax2.add_patch(nrect)
                    newlydrawnrects.append(nrect)
            self.noisedrawnrects = newlydrawnrects.copy()

            not_in_indexes = sorted(list(set(self.user_selected_noise) - set(self.noiseindexes)))

            noise_in_ax2 = [i for i in self.noiseindexes if i in range(smallest_int, highest_int+1)]
            noise_in_ax2_val = [self.data[i] for i in noise_in_ax2]

            self.ax2.plot(noise_in_ax2, noise_in_ax2_val, "x", color=self.colorify["noise_true"])

            if len(not_in_indexes) > 0:
                self.ax2.plot(not_in_indexes, [self.data[i] for i in not_in_indexes], "x", color=self.colorify["noise_false"])

            if self.ax2baseline != None:
                self.ax2baseline.remove()
                self.ax2baseline = None
            if self.plotconf["zero"] == True:
                self.ax2baseline = self.ax2.axhline(y=0.0, color=self.plotconf["zero_color"], linestyle='-')

            if self.ax2grid != None:
                self.ax2grid.remove()
                self.ax2grid = None
            if self.plotconf["grid"] == True:
                self.ax2grid = self.ax2.grid(linestyle="-", color=self.plotconf["grid_color"], alpha=0.5)
            else:
                self.ax2.grid(False)
            print("def drawAx2 drawing baseline and grid if user selected so done")
            if curlims:
                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])
            self.figure.canvas.draw()
            print("def drawAx2 initial labels:")
            print([item.get_text().replace("−", "-") for item in self.ax2.get_xticklabels()])

            if self.master.current_timescale == "s":
                labels = [ float("{:.3f}".format(float(item.get_text().replace("−", "-")) / self.FPS)) for item in self.ax2.get_xticklabels() ]

            elif self.master.current_timescale == "ms":
                labels = [ float("{:.3f}".format((float(item.get_text().replace("−", "-")) / self.FPS)*1000)) for item in self.ax2.get_xticklabels() ]

            print("def drawAx2 about to update labels:")
            self.ax2.set_xticklabels(labels)
            print("def drawAx2 update labels done")
            self.figure.canvas.draw()

    #Mode FFT -> Click on maximum points and set them as frequency title of plot
    def on_press_ax2fft(self, event):
        self.ax2.set_title("")
        selected = None
        for child in self.ax2.get_children():
            if isinstance(child, Line2D) and child.contains(event)[0] and child.get_marker() == "o" and selected == None:
                child.set_color(self.colorify["fft_selection"])
                selected = child
                # break
            elif isinstance(child, Line2D) and child.get_marker() == "o":
                child.set_color(self.colorify["fft"])
        self.figure.canvas.draw()
        if selected == None: return
        xdata, ydata = selected.get_data()
        self.master.current_frame.plotFFTSelection = self.FFTxData[float(xdata[0])]
        # self.ax2.set_title("Frequency: " + "{:.3f}".format(float(xdata[0])) )
        self.ax2.set_title("Selected Wave Frequency: " + "{:.3f}".format(float(xdata[0])) )
        self.figure.canvas.draw()

    #Mode Noise -> Draw areas and set them as noise or not noise
    def on_press_ax2noise(self, event):
        if event.inaxes != self.ax2: return

        if event.button == 1 and event.dblclick == False and self.locknoiserect == False and event.xdata != None and event.ydata != None:
            self.locknoiserect = True
            self.noiserectx0 = event.xdata
            return

        if event.button == 1 and event.dblclick == True and self.areaopen == False:
            #check if any rect on double click area and delete accordingly
            self.locknoiserect = False
            self.noiserectx0 = None
            if len(self.noisedrawnrects) > 0:
                for i in range(len(self.noisedrawnrects)-1, -1, -1):
                    r = self.noisedrawnrects[i]
                    print("r")
                    print(r)
                    rstart = r.get_x()
                    rend = r.get_x() + r.get_width()
                    if event.xdata >= rstart and event.xdata <= rend:
                        self.noisedrawnrects[i].remove()
                        del self.noisedrawnrects[i]
                        self.figure.canvas.draw()
            return

        if event.button == 2:
            #mouse scroll event, at the moment does nothing
            self.locknoiserect = False
            self.noiserectx0 = None
            return

        if event.button == 3 and event.dblclick == False and self.noiseareaopen == False:
            #check if any rect on double click area and open context menu if inside
            insiderect = False
            rend =  None
            for r in self.noisedrawnrects:
                rstart = r.get_x()
                rend = r.get_x() + r.get_width()
                if event.xdata >= rstart and event.xdata <= rend:
                    self.noisearearect_x0 = int(math.ceil(rstart))
                    self.noisearearect_x1 = int(math.floor(rend))
                    insiderect = True
                    break
            if insiderect:
                self.noiseareaopen = True
                self.noiserectx0 = None
                try:
                    #x = self.master.winfo_pointerx()
                    #y = self.master.winfo_pointery()
                    abs_coord_x = self.master.winfo_pointerx() - self.master.winfo_vrootx()
                    abs_coord_y = self.master.winfo_pointery() - self.master.winfo_vrooty()
                    self.master.popuplock = True
                    #self.master.currentpopup = self.areaNMenu
                    self.master.currentpopup = self.master.current_frame.areaNMenu
                    self.master.current_frame.areaNMenu.tk_popup(int(abs_coord_x + (self.master.current_frame.areaNMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
                    #self.areaNMenu.tk_popup(int(abs_coord_x + (self.areaNMenu.winfo_width()/2) + 10 ), abs_coord_y, 0)
                finally:
                    #self.areaNMenu.grab_release()
                    self.master.current_frame.areaNMenu.grab_release()
                    self.locknoiserect = True

    def on_motion_ax2noise(self, event):
        if event.inaxes != self.ax2: return
        if self.locknoiserect == True and event.xdata != None and event.ydata != None and self.noiserectx0 != None:

            curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())

            self.noiserectx1 = event.xdata
            # self.recty1 = event.ydata
            
            # new_rect = Rectangle((0,0), 1, 1)
            new_rect = Rectangle((0,np.min(self.data)), 1, 1)
            new_rect.set_facecolor(self.colorify['rect_color'])
            new_rect.set_width(self.noiserectx1 - self.noiserectx0)

            # nheight = np.max(self.data) + abs(np.min(self.data)) + 0.5
            nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5

            new_rect.set_height(nheight)

            # new_rect.set_xy((self.noiserectx0, np.min(self.data) ))
            new_rect.set_xy((self.noiserectx0, curlims[1][0] ))

            if self.noiserect:
                self.noiserect.remove()
            self.noiserect = None
            self.noiserect = self.ax2.add_patch(new_rect)

            self.ax2.set_xlim(curlims[0])
            self.ax2.set_ylim(curlims[1])

            self.figure.canvas.draw()

    def on_release_ax2noise(self, event):
        if event.inaxes != self.ax2: return
        if self.locknoiserect == True and event.xdata != None and event.ydata != None and self.noiserectx0 != None:
            self.noiserectx1 = event.xdata
            if event.xdata < self.noiserectx0:
                nr1 = self.noiserectx0 + 0.0
                self.noiserectx0 = event.xdata
                self.noiserectx1 = nr1
            self.noiserecty1 = event.ydata
            if abs(self.noiserectx1 - self.noiserectx0) > 1:
                #check if any rectangle on area and merge both rectangles if they exist
                # new_rect = Rectangle((0,0), 1, 1)

                curlims = (self.ax2.get_xlim(), self.ax2.get_ylim())

                new_rect = Rectangle((0,np.min(self.data)), 1, 1)
                new_rect.set_facecolor(self.colorify['rect_color'])
                new_rect.set_width(self.noiserectx1 - self.noiserectx0)
                
                nheight = curlims[1][1] + abs(curlims[1][0]) + 0.5

                new_rect.set_height(nheight)

                # new_rect.set_xy((self.noiserectx0, np.min(self.data)))
                new_rect.set_xy((self.noiserectx0, curlims[1][0]))
                

                nrectstart = new_rect.get_x()
                nrectend = new_rect.get_x() + new_rect.get_width()

                nstarts = [nrectstart]
                nends = [nrectend]
                i = 0
                redraw = []
                if len(self.noisedrawnrects) >= 1:
                    for i in range(len(self.noisedrawnrects)):
                        r = self.noisedrawnrects[i]
                        rstart = r.get_x()
                        rend = r.get_x() + r.get_width()
                        if (nrectstart >= rstart and nrectstart <= rend) or (nrectend >= rstart and nrectend <= rend) or (nrectstart <= rstart and nrectend >= rend):
                            nstarts.append(rstart)
                            nends.append(rend)
                            self.noisedrawnrects[i].remove()
                        else:
                            redraw.append(self.noisedrawnrects[i])
                        i += 1
                self.noisedrawnrects = None
                self.noisedrawnrects = redraw.copy()

                if self.noiserect:
                    self.noiserect.remove()
                self.noiserect = None
                
                smallest = np.min(nstarts)
                highest = np.max(nends)
                new_rect.set_width(highest - smallest)

                # new_rect.set_xy((smallest, np.min(self.data)))
                new_rect.set_xy((smallest, curlims[1][0]))

                #save rect to rects
                self.noiserect = self.ax2.add_patch(new_rect)
                # if not overlap:
                self.noisedrawnrects.append(self.noiserect)
                #if zoom or fft area, draw data on subplot area (ax2)

                self.noiserect = None


                self.ax2.set_xlim(curlims[0])
                self.ax2.set_ylim(curlims[1])

                self.figure.canvas.draw()
                self.locknoiserect = False

    def reset_zoom_variables(self):
        if self.zoomrect != None:
            self.zoomrect.remove()
            self.zoomrect = None
        self.zoomrectx0 = None
        self.zoomrecty0 = None
        self.zoomrectx1 = None
        self.zoomrecty1 = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        
    def on_press_zoom(self, event):
        self.reset_zoom_variables()
        if event.button == 1 and event.dblclick == False and self.lockzoom == False and event.xdata is not None and event.ydata is not None:
            self.lockzoom = True
            self.zoomrectx0 = event.xdata
            self.zoomrecty0 = event.ydata
            return
        elif event.button == 1 and event.dblclick == True:
            self.lockzoom = False
            self.zoomed = False
            if self.zoomrect != None:
                self.zoomrect.remove()
                self.zoomrect = None
            if event.inaxes == self.ax:
                self.ax.set_xlim(self.axcurlims[0])
                self.ax.set_ylim(self.axcurlims[1])
            if event.inaxes == self.ax2:
                self.ax2.set_xlim(self.ax2curlims[0])
                self.ax2.set_ylim(self.ax2curlims[1])
            #get master original zoom and return to original state on selected axis zoom
            self.figure.canvas.draw()
            return
        elif self.lockzoom == True:
            self.lockzoom = False
        return

    def on_motion_zoom(self, event):
        if self.lockzoom == True and event.xdata != None and event.ydata != None and self.zoomrectx0 != None and self.zoomrecty0 != None:
            curlims = (event.inaxes.get_xlim(), event.inaxes.get_ylim())
            self.zoomrectx1 = event.xdata
            self.zoomrecty1 = event.ydata
            
            self.x0 = self.zoomrectx0
            self.x1 = self.zoomrectx1
            if self.zoomrectx1 < self.zoomrectx0:
                self.x0 = self.zoomrectx1
                self.x1 = self.zoomrectx0
            xwidth = self.x1 - self.x0

            self.y0 = self.zoomrecty0
            self.y1 = self.zoomrecty1
            if self.zoomrecty1 < self.zoomrecty0:
                self.y0 = self.zoomrecty1
                self.y1 = self.zoomrecty0
            yheight = self.y1 - self.y0

            new_zrect = Rectangle((self.x0,self.y0), xwidth, yheight, fill=False, alpha=1)
            if self.zoomrect:
                self.zoomrect.remove()
            self.zoomrect = None
            self.zoomrect = event.inaxes.add_patch(new_zrect)

            event.inaxes.set_xlim(curlims[0])
            event.inaxes.set_ylim(curlims[1])

            self.figure.canvas.draw()
            return
        elif self.lockzoom == True:
            self.lockzoom = False
            self.reset_zoom_variables()
        return

    def on_release_zoom(self, event):
        if self.lockzoom == True and event.xdata != None and event.ydata != None and self.zoomrectx0 != None and self.zoomrecty0 != None and self.zoomrectx1 != None and self.zoomrecty1 != None:
            event.inaxes.set_xlim((self.x0, self.x1))
            event.inaxes.set_ylim((self.y0, self.y1))
            self.figure.canvas.draw()
        self.reset_zoom_variables()
        self.lockzoom = False
        return
#https://stackoverflow.com/questions/12052379/matplotlib-draw-a-selection-area-in-the-shape-of-a-rectangle-with-the-mouse
#https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
#https://stackoverflow.com/questions/12014210/tkinter-app-adding-a-right-click-context-menu
#http://effbot.org/zone/tkinter-popup-menu.htm
#https://stackoverflow.com/questions/48431070/can-tkinter-menus-send-the-event-in-which-they-were-called
#https://stackoverflow.com/questions/21200516/python3-tkinter-popup-menu-not-closing-automatically-when-clicking-elsewhere
        
