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
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from customframe import ttkScrollFrame
from sys import platform as _platform

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

class HelpDialog(tkDialog.Dialog):
    def body(self, master):
        print("class AboutDialog def body creation")
        self.title_font = tkfont.Font(family='Helvetica', size=18)
        self.normal_font = tkfont.Font(family='Helvetica', size=12)

        pagenames = {
            "StartPage": [
            "After opening the Contraction Wave program, several Buttons can be seen at the Initial Screen, each with different purposes. To load New Data into the program, click the “New Data” Button highlighted below to move to the New Data Screen.",
            "Fig 1 - The New Data Button highlighted in Red in the program’s Initial Screen.",
            "Alternatively, the “New Data”  Option Button can be selected from the Top Bar File Menu at anytime to move to the New Data Screen.",
            "Fig 2 - The New Data Option Button highlighted in Red in the program’s File Menu.",
            "Other options can be explored by the Help menu in each Screen.",
            "Title Comparing Processed Waves",
            "Title Exporting Data",
            "This option can be accessed from the Initial Screen by clicking in the Merge Results Button.",
            "Fig 61 - Merge Results Button in the Initial Screen",
            "Tables exported from the Visualizing Wave Parameters Screen (See section in this Document) can be added or removed in a Pop Up Window. After all desired Tables are included, experiments from different Data Types can be summarized and compared in a final table when the 'Ok' Button is clicked. The user will be prompted to select a file name and destination folder for the Summary table.",
            "Fig 62 - Merge Results Pop Up Windows with the Add and Delete Buttons for selecting tables for creating a Summary.",
            "Title Customizing Plots",
            "Title Editing Plot Settings",
            "Custom Plot Settings can be edited at anytime by clicking the Plot Settings Menu at the Top Bar. The user then has the option to either Edit, Save or Load a Plot Setting.",
            "Fig 63 - Plot Settings Menu at the Top Bar options.",
            "Editing a Plot setting allows the user to change colors for any of the Wave Points, Lines, Selection Areas, to select whether a horizontal baseline is drawn at each plot at 0.0 and whether a grid is drawn in each plot and to customize the baseline and grid colors. Time units can be changed between Seconds and Milliseconds.",
            "Fig 64 - Editing Plot Settings option Pop Up Window.",
            "The Peak Absolute (in relation to the whole analysis) or Relative time (starting at 0.0) can be displayed in the Visualizing Wave Parameters and the Jet and Quiver Plots window also according to the user’s preference.",
            "Saving or Loading a Plot Setting prompts the user to select a filename or directory for their respective operation. The window plots are automatically refreshed after a new Plot Setting is loaded."
            ],
            
            "PageOne": ["Many options can also be found in the New Data Screen. Three main buttons allow the user to load a new data type (Load Data Button ), remove all added data types (Delete All Button) from the processing list or run the added data types (Run Button.) in the Processing Queue. Click the Load Data Button for loading a new data type.",
            "Fig 3 - The New Data Screen and it’s three main buttons.",
            "Title Data from an Image folder",
            "A Popup Window containing three options will appear above the New Data Screen. For processing a Folder containing multiple images click the 'Folder' Button on the Popup Window. Readable Image types include all OpenCV supported Image types: '.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif'.",
            "Fig 4 - New options in the New Data Screen after clicking the Load Data Button.",
            "A System Folder Selection dialog selection window will then appear. Please navigate and select the Image containing Folder to be analysed and click OK for the next step. IMPORTANT: Folders containing Non-ASCII characters such as: '^ ' ` ç' are not well supported by the program.",
            "The selected Image Folder will then be briefly pre-processed so that relevant information can be extracted. The right side of the New Data Screen is then updated with information extracted from the Image Folder. Various parameters which need to be adjusted by each user also appear on the screen. ",
            "Fig 5 - Updated New Data Screen after loading an Image Folder Data Type.",
            "Title Data from an Video file",
            "A Popup Window containing three options will appear above the New Data Screen. For processing a Video file click the 'Video' Button on the Popup Window.  Only the '.avi' video type is supported, due to it’s OpenCV support in multiple platforms.",
            "Fig 4 - New options in the New Data Screen after clicking the Load Data Button.",
            "A System File Selection dialog selection window will then appear. Please navigate to the Video file containing Folder to be analysed, select the Video File and click OK for the next step. IMPORTANT: Folders containing Non-ASCII characters such as: '^ ' ` ç' are not well supported by the program.",
            "The selected Video File will then be briefly pre-processed so that relevant information can be extracted. The right side of the New Data Screen is then updated with information extracted from the Video File. Various parameters which need to be adjusted by each user also appear on the screen. ",
            "Fig 6 - Updated New Data Screen after selecting a Video File as input.",
            "Title Data from an TIFF Directory",
            "A Popup Window containing three options will appear above the New Data Screen. For processing a TIFF Directory file click the 'Compressed TIFF' Button on the Popup Window.  Both  '.tiff' and '.tif' files are supported.",
            "Fig 4 - New options in the New Data Screen after clicking the Load Data Button.",
            "A System File Selection dialog selection window will then appear. Please navigate to the TIFF Directory containing Folder to be analysed, select the TIFF Directory File and click OK for the next step. IMPORTANT: Folders containing Non-ASCII characters such as: '^ ' ` ç' are not well supported by the program.",
            "The selected TIFF Directory File will then be briefly pre-processed so that relevant information can be extracted. The right side of the New Data Screen is then updated with information extracted from the TIFF Directory File. Various parameters which need to be adjusted by each user also appear on the screen. ",
            "Fig 7 - Updated New Data Screen after selecting a TIFF Directory File as input.",
            "Title Editing a Data Group’s Name and Deleting Data Groups",
            "All added Data Groups are automatically named according to their current Files/Folder names. For editing a given Data Group name, click the 'Rename' Menu Button after Right Clicking a Data Group. ",
            "Fig 8 - Right Click context Menu for a Data Group with the 'Rename' Button option highlighted.",
            "A new Popup Window will then open in which the Data Group’s name can be edited. After typing a new name, click the 'OK' option for saving.",
            "Fig 9  - Pop up Menu for a editing a Data Group’s name.",
            "For deleting a given Data Group name, click the 'Delete' Menu Button after Right Clicking a Data Group. This will remove the Pre-processed Data Group from the Data Groups List View at the left screen side.",
            "Fig 10 - Right Click context Menu for a Data Group with the 'Delete' Button option.",
            "Title Setting the Frames Per Second and Pixel Size Parameters",
            "The Frames Per Second parameter is automatically extracted from the '.avi' files in case of Video Inputs but needs to be properly set for TIFF Directory and Image Folders Data types. This parameter refers to the total frame capture of each image in these inputs for a complete second which is particular to the camera used during the Contractility experiment.",
            "Similarly the Pixel Size parameter also refers to the size of a pixel given the capturing device used during the Contractility experiment. Both FPS and Pixel Size values are not bounded by any means in the program and as such caution is needed for setting these parameters.",
            "Adjusting the Optical Flow Parameters",
            "All other parameters present in the Advanced Settings tab refer to the OpenCV Gunner Farneback Dense Optical Flow algorithm parameters. Users should refer to the OpenCV Documentation for more details on how setting these parameters: OpenCV Gunner Farneback Docs.",
            "Title Saving and Loading Optical Flow Parameters",
            "All other parameters present in the Advanced Settings tab set by a given Data Group can be saved as a new preset. Saved presets can be applied to any other Data Group. Both options are done by Right Clicking a Data Group at the Screen Left side.",
            "Fig 11 - Right Click context Menu for a Data Group with the 'Add to Preset' and 'Apply Preset' Button options highlighted.",
            "For adding a new Data Group Advanced Parameters as a new Preset,click the 'Add to Preset' Menu Button after Right Clicking a Data Group. A new Popup Window will then open a a Preset name will be asked. After typing a name, click the 'OK' option for saving the selected group’s Advanced Parameters tab as a new Preset.",
            "Fig 12 - Pop Up Window for adding a new Preset.",
            "For applying a previous preset to a Data Group, click the 'Apply Preset' Menu Button after Right Clicking a Data Group. A new Popup Window will then open containing all previously saved Presets and their corresponding Advanced Parameters tab values. After selecting an adequate Preset, click the 'OK' option for applying these values to the selected group.",
            "Fig 13 - Pop Up Window for applying a previous Preset.",
            "Processing the Input Data",
            "All input Data can be processed by clicking the 'Run All' Button at the bottom right corner of the screen.",
            "Fig 14 - Running all selected Data.",
            "The program will automatically Save each of the queued Data Groups. Data Groups are processed in a multithreaded processing queue.",
            "Fig 15 - Running all queued Data. Confirmation Window."],
            "PageTwo": ["Title Checking the Processing Queue",
            "After running all the selected Data Groups, the Processing status for each group can be checked by either clicking the 'Check Progress' Button at the bottom of the New Data Screen.",
            "Fig 16 - Checking Progress from the New Data Screen.",
            "Or clicking the 'Check Progress' Button directly from the Initial Screen.",
            "Fig 17 - Checking Progress directly from the Initial Screen.",
            "Or also by clicking the Check Progress Menu Button from the File Menu of the program’s top bar.",
            "Fig 18 - Checking Progress by clicking the Check Progress Menu Button.",
            "Various information regarding a Data Group’s processing can be seen at the Check Progress Screen. These include a progress bar indicating the completed processing status, The elapsed and estimated finish times and an 'Analysis Button'  that allows the user to start the analysis on a given Data Group after it’s processing is finished.",
            "Fig 19 - Check Progress Screen. In highlights are: 1. Data Group’s name; 2. Processing Progress Bar; 3. Time elapsed and Estimated Finish Time for Processing; 4. Analysis Button for a Data Group"],
            "PageThree": ["Title Loading Groups for Analysis",
            "Fig 20 - Starting Analysis by clicking the Start Analysis Button.",
            "Which can also be accessed by clicking the Load Analysis Menu Button from the File Menu of the program’s top bar.",
            "Fig 21 - Loading Analysis by clicking the Load Analysis Menu Button.",
            "Starting Analysis from the Initial Screen or from the Load Analysis Menu Button allows the user to check a Data Group’s Time vs Raw Average Speed data generated from the Dense Optical Flow algorithm in a scrollable table.",
            "Fig 22 - Selecting processed Data Groups saved on Memory or Disk for Analysis.",
            "The Data Group’s Time vs Raw Average Speed data generated from the Dense Optical Flow algorithm can also be downloaded by either the Download Table Button or the Export Current Table Menu Button from the Export Menu.",
            "Fig 23 - Exporting processed Data Groups raw Data.",
            "Both .CSV and .XLS or .XLSX exporting formats are supported and can be selected from a Pop up Window",
            "Fig 24 - Pop up Window for selecting export table format."],
            "PageFour": ["Title Analysing Time vs Average Speed plot",
            "By either clicking the Start Analysis Button on the Start Analysis Screen, the Analysis Button on the Check Progress Screen or the Load Analysis Menu Button on the File Menu of the Top Bar, the user can navigate to the Wave Points Definition Screen.",
            "Fig 25 A) Clicking the Start Analysis Button on the Start Analysis Screen. B) Clicking the Analysis Button on the Check Progress Screen. C) Clicking the Load Analysis Menu Button on the File Menu of the Top Bar.",
            "The main objective of this Screen is allowing the User to select Waves of interest from the Processed Data by an automatic Wave detection algorithm combined with the manual adjustment by the User when needed.",
            "The user can click and drag the mouse for selecting Wave areas of interest to be analysed in the posterior plots. Releasing the mouse inside the plot creates a Plot Selection that is initially zoomed in the below subplot. The User can display the automatically detected points by clicking the Plot Dots Check Box at the bottom menu. More on this algorithm and how to better adjust it to fit the data of interest is written in the following section.",
            "Fig 26 A) Initial Plot display for the starting analysis. B) Clicking and dragging in the Plot Area creates a Plot Selection that is zoomed in the initial sub-plot below. C) Automatically detected points are shown in the plot by clicking the Plot Dots Check Box at the bottom menu",
            "Multiple Plot Selections can be created in a single plot. Plot Selections can be deleted by double mouse clicking and can be merged by overlapping the areas of a currently drawn Plot Selection with a previously drawn Plot Selection. Two Mouse Modes are currently available in this Screen: Plot Area Select and Plot Point Select in the Top Bar under the Plot Mouse Mode Menu. For drawing selections, please ensure that the Plot Area Select Mouse Mode is selected by clicking under this Menu.",
            "Title Separating Wave from Noise and the Wave Points defining Algorithm",
            "The automatic Wave detection algorithm first step is to separate frames related to Contraction-Relaxation (Waves) from baseline state oscillations during measurement (Noise). This is done by a fixed threshold whose first guess is calculated as the median of all data points values. When first opening the Wave Points Definition Screen this value can be seen in the 'Noise Cutoff' top menu Spin Box highlighted below.",
            "Fig 27 - 'Noise Cutoff' top menu Spin Box highlighted in red.",
            "As one can imagine, this value is not suited for all Data Groups. The user can help the algorithm to select an adequate Noise Cutoff by editing the values in the top menu Spin Box. Visualization of values defined as Noise can be done by clicking the Plot Noise Max Line check box in the top menu. ",
            "Fig 28 - 'Plot Noise Max Line' visualization in the plot. Wave points 'mislabelled' (see below) as Noise are highlighted in red.",
            "You may notice some of points inside the Wave area were detected as Noise. The algorithm accounts for such variation and this generally does not affect the Point Definition step. One of the ways this is accounted for can be seen in the Noise Detection Options in the Data Options top bar Menu.",
            "Fig 29 A) - 'Noise Detection Options'  in the top bar Menu. B) 'Noise Detection Options' Pop Up Window with labelled options.",
            "The Noise Areas Min. Size Filtering is the second step of the Noise defining algorithm. In this step, Noise sequences of points (or areas) containing a Time smaller than the Average Time of ALL Noise areas are defined as possible Wave areas. This step is Optional to the analysis and may be adjusted by the user in the 'Noise Detection Options' Pop Up Window.",
            "Other options in this Window include decreasing the Average Speed of ALL Noise areas or a Custom User Input Value from the Data after the Noise defining algorithm and the Point Definition step are done. This may be useful for cases in which the User believes the obtained Wave Speed values are suffering from interference from the baseline values since these interferences can affect various Peak Parameters in the following Peak Parameters Screen Analysis.",
            "The algorithms third step starts the Point Definition steps. In these steps, five points of four possible point types are detected for each possible Wave areas. Firstly, possible local maximums are extracted from an open source Peak Detection Algorithm which utilizes a parameter called Delta, which is the difference from a possible local minimum to a following local maximum. The initial value guessed by the program for this parameter is the Average Speed of all Wave defined points divided by 3.",
            "The User can display the automatically detected points by clicking the Plot Dots Check Box at the bottom menu. If these dots look clearly wrong, the first correcting measure should be editing the Delta value at the top menu. It should be raised when more dots are being displayed than it should and lowered when fewer dots are being displayed than it should.",
            "Fig 30 A) - 'Plot Dots' Check Box  in the bottom menu. B) 'Delta' Spin Box in the top menu.",
            "In the following steps for Point Definition, possible local maximums are filtered and only those with zero-derivative in Wave defined areas remaining for analysis.",
            "For each pair of remaining Maximum points, the Start Point of a Wave is defined as the closest zero-derivative minimum Noise point preceding the first Maximum.",
            "The Minimum Point is defined as the point with the smallest Speed between the two Maximums.",
            "And the Last Point is finally defined by fitting an exponential function to all points succeeding the Second Maximum until the end of the following Noise area and defining a Stop Criteria for when the exponential has stabilized. This approach was chosen due to the Waves double peak pattern which consists on a very fast increase from the baseline in the double peak beginning but a slow decay in the end.",
            "The initial Stop Criteria is 0.01 or 1% which means that when a zero-derivative minimum point following the Second Maximum has a difference in the fitted function of less than 1% to the anteceding neighbour point it is defined as the Last Point of the Wave.",
            "This Stop Criteria generally works fine with the tested cases but should also be customized according to the data in the Stop Criteria Spin Box shown in the Figure below. Waves displaying a slower decay should have a smaller Stop Criteria and Waves with a faster decay should have a higher Stop Criteria. The Exponential Fit can also be visualized for each individual Peak by clicking the Plot Exponential Regressions Check Box highlighted in the Figure below.",
            "Fig 31 - 'Stop Criteria' Spin Box and the Plot Exponential Regressions Check Box in the top menu.",
            "Each of the four point types (First, Maximum, Minimum and Last) are colored differently in the plot. Default color configurations are Green, Red, Blue and Purple but can be changed anytime along with other Plot Settings (See Customizing Plots in this Document).",
            "After Waves are properly assigned and selected, the user can move to the next analysis as described in the final session for this Analysis.",
            "Despite all these steps, highly noisy data groups with Noise oscillations close to the Wave values may still difficult Wave assignment from the Speed data calculated by the Optical Flow Algorithm. These particularly arise from multidirectional Contraction-Relaxation data such as those seen in Neonatal cardiomyocytes. The program presents multiple options for contouring such issues which are all exemplified below.",
            "Title Applying Smoothing/Denoising Algorithms",
            "One option for correcting highly noise data is applying a Smoothing/Denoising Algorithm on the data. The offset of this method is that the Raw Speed data is clearly modified when such an algorithm is applied, which may lead to wrong conclusions regarding the methodology results. As such these methods are to be applied with care, specially when the objective is comparing data groups from different experiments/cell types which is the most often case.",
            "Fig 32 - A highly noisy case with clear bad point detection",
            "Three main Smoothing algorithms are present in this program and are the Savitzky-Golay filter, a Fast Fourier Transform finite impulse response (FIR) filter, and an Average Window denoising. These can be opened from the Data Options > Smooth/Denoise Menus in the Top Bar. ",
            "Fig 33 - Data Options > Smooth/Denoise Menus in the Top Bar",
            "Clicking each filter opens up a Pop Up Window for applying such filter. Customizing options for these filters include Window length for the Savitzky-Golay filter and Average Window denoising, the Polynomial Order used by the Savitzky-Golay filter, various possible Window Scaling types for the Average Window denoising and the Percentage of the highest frequencies kept in the Fast Fourier Transform filter.",
            "Fig 34 - Selecting one of the Smooth/Denoise options opens up three possible pop ups for the the Average Window, Savitzky-Golay, and FFT Frequency filters (A-C respectively)",
            "The Savitzky-Golay filter and Average Window denoising procedure can also be applied to an user selected Plot Selection by clicking the Plot Selection with the right mouse button and selecting any of the two options regarding these filters as seen in the Figure below.",
            "Fig 35 - Right clicking a Plot Selection allows the user to apply the Savitzky-Golay filter and Average Window denoising procedure on a specific plot area as highlighted in red.",
            "The data can be restored to the original configuration by clicking in the Data Options > Restore Original  Menu Option in the Top Bar. After Waves are properly assigned and selected, the user can move to the next analysis as described in the final session for this Analysis.",
            "Title Separating Waves from Noise by area selection",
            "The following two options for correcting noise/peak definition by the program is by manually setting noise/peak areas or each of the five peak points after adjusting the parameters in the first section of this Screen tutorial (Separating Wave from Noise and the Wave Points defining Algorithm).",
            "For setting the noise/peak areas first set the Sub-Plot Mode to Noise/Peak Areas in the Sub-Plot Mode Menu of the Top Bar. This will update the below Sub-Plot to show points defined as Noise by the Noise defining algorithm in blue crosses as shown below.",
            "Fig 36 - Sub-Plot Modes in the Sub-Plot Mode Menu of the Top Bar",
            "Also ensure that the Plot Mouse Mode is set to Plot Area Select in the Plot Mouse Mode Menu of the Top Bar.",
            "Fig 37 - Plot Mouses Modes in the Plot Mouse Mode Menu of the Top Bar",
            "Now Sub-Plot Selection Areas can be generated in the same way Plot Selection areas are by clicking, dragging and released with the left mouse button.",
            "Fig 38 - Sub-Plot Selections in the Noise/Peak Areas Sub-Plot generated by clicking and dragging.",
            "Right clicking a Sub-Plot Selection Area allows the user to set individual parts of the Plot as Wave or Noise, which greatly helps the Point Detection algorithm in defining where the true Maximum points are located. An important thing to note is that changing any of the automatic detection options resets any of the user manual editions. After Waves are properly assigned and selected, the user can move to the next analysis as described in the final session for this Analysis.",
            "Fig 39 - 'Set as Wave' and 'Set as Noise' Menu Options in the Noise/Peak Areas Sub-Plot generated by right clicking a Sub-Plot Selection.",
            "Title Editing and Creating Wave points manually",
            "Another way to edit wrongly or non assigned Wave points is to manually add, remove or move points in the plot. First, ensure that the Plot Mouse Mode is set to Plot Point Select in the Plot Mouse Mode Menu of the Top Bar. ",
            "Fig 37 - Plot Mouses Modes in the Plot Mouse Mode Menu of the Top Bar",
            "Also check that the Sub-Plot Mode is assigned to Peak Zoom in the Sub-Plot Mode Menu of the Top Bar.",
            "Fig 36 - Sub-Plot Modes in the Sub-Plot Mode Menu of the Top Bar",
            "Now points can be moved by dragging and dropping with the left mouse button.",
            "Fig 40 - Clicking and dragging a Wave Point.",
            "The plot can be right clicked at any position for Adding a new Point. Points in the Plot can also be right clicked and Removed or have their type Changed to another one.",
            "Fig 41 - A) Right clicking in the Plot in the Plot Point Select Mouse Mode opens up a new Menu. B) Pop Up Window for creating a new point or changing a point’s type.",
            "This is a more radical way to assign Waves which anyhow can be useful since all the assignment here is completely done by the user. An important thing to note is that changing any of the automatic detection options resets any of the user manual editions. After Waves are properly assigned and selected, the user can move to the next analysis as described in the final session for this Analysis.",
            "Title Checking Wave Frequency by Fast Fourier Transform of the Data",
            "Finally the Wave frequency of the plot can be checked using a Fast Fourier Transform of the Data. First, ensure that the Plot Mouse Mode is set to Plot Point Select in the Plot Mouse Mode Menu of the Top Bar. ",
            "Fig 37 - Plot Mouses Modes in the Plot Mouse Mode Menu of the Top Bar",
            "Then check that the Sub-Plot Mode is assigned to FFT in the Sub-Plot Mode Menu of the Top Bar.",
            "Fig 36 - Sub-Plot Modes in the Sub-Plot Mode Menu of the Top Bar",
            "This displays the Fast Fourier Transform of the Plot Data. Clicking any single point in the subplot highlights this point and updates the Plot’s title with the corresponding selecting Frequency.",
            "Fig 42 - A) FFT Sub-Plot Mode. B) Clicking a new dot ",
            "The Fast Fourier Transform is applied to the last created Plot Selection in the Plot if any has been previously selected.",
            "Fig 43 - FFT Sub-Plot Mode is now applied to the last Plot Selection created.",
            "Title Exporting Data",
            "Multiple Data can be exported from this Screen if the user desires to do so. This includes the whole or individual Plot+Sub Plot images as well as their data. Supported exported figure data includes various possible formats according to the user’s computer matplotlib installation. Like the previous Screen, Data is exported in tabular format in either the .csv or .xls / .xlsx formats.",
            "Fig 44 - Export Menu in the Top Bar with several export options.",
            "Fig 45 - Pop Up Window for exporting Figures in Contraction Wave. After clicking the ok button, the user is prompted to select a file name and a folder for saving.",
            "Moving to the Next Analysis",
            "One or more Waves are required to be selected (under a Plot Selection) before progressing for further analysis. A complete Wave is defined by the presence of the following Wave Points in this order: First Point, Maximum Point, Minimum Point, Maximum Point and Last Point.",
            "Fig 46 - A) A valid selection of three Waves before progressing to the next analysis. B) Clicking the Analyse Wave Areas button allows the user to progress to the next analysis.",
            "For drawing selections, please ensure that the Plot Area Select Mouse Mode is selected by clicking under this Menu and click and drag the Top Plot."],
            "PageFive": ["Title Time, Speed and Area Parameters",
            "In this Screen, the user can visualize each of the previously selected Waves in the below Plot by clicking any of the Top Table rows.",
            "Fig 47 - Visualizing Wave Parameters Screen after selecting various Waves.",
            "Various parameters of interest regarding the Waves Time, Speed and Area are calculated for each Wave based on the previously defined: First Point, Maximum Point, Minimum Point, Maximum Point and Last Point.",
            "Switching between these three types of tables containing Time, Speed and Area parameters can be done by clicking on the Top Radio Buttons.",
            "Fig 48 - Switching between Top Radio Buttons of the Visualizing Wave Parameters Screen.",
            "A Middle table allows seeing the Average values of each of the selected Parameters for the currently selected Waves. ",
            "Fig 49 - Highlighted average parameter table in the middle of the screen.",
            "Incorrectly assigned or selected Waves can be deleted by right clicking to open a context Menu and selecting the Delete option.",
            "Fig 50 - Deleting a misassigned Wave by right clicking and selecting Remove Peak.",
            "The Plot Figure/Data and all the tabular Data can be exported in the Top Bar Export Menu.",
            "Fig 51 - Export options for this Screen.",
            "In order to move to the final Analysis, the user must select a valid Wave row and click the Quiver/Jet Plots button at the bottom of the screen. By doing so, the program will ask whether the user desires to Save the current Waves to the disk. This is highly recommended for accessing this data at a later time.",
            "Fig 52 - Moving forward to the last analysis by clicking the Quiver/Jet Plots button",
            "Saved Waves can be accessed at anytime by clicking the Load Saved Waves option in the File Menu of the Top Bar. From the Initial Screen it is also possible to load saved Waves by clicking the Load Saved Waves Button.",
            "Fig 53 - Load Saved Waves Menu Option (A) and Button at the Initial Screen (B)"],
            "PageSix": ["Title Visualizing the full Contraction-Relaxation Cycle",
            "Various possible visualizations of the full Contraction-Relaxation Cycle of a selected Wave are possible in this Screen. By clicking the first three Top Check Boxes, three different visualizations can be merged in a total of seven different combinations. The Jet Check Box  displays Speed per each pixel in the selected frame. The Quiver Check Box displays Speed Vectors with both velocities and the optical flow calculated directions over a window in the X (horizontal) and Y (vertical) image axes. The Image Check Box  displays the current plot Image the Optical flow was calculated from.",
            "Fig 54 - Three Top Check Boxes in red highlighting the three different visualizations: Image (A), Jet (B) and Quiver (C).",
            "An Legend automatically scaled from the previously selected Noise threshold to the Maximum possible Speed in any of the image’s pixels can also be drawn or removed by clicking the fourth Check Box.",
            "Fig 55 - Fourth Check Box in red and legend drawn for the Jet visualization.",
            "The Jet/Quiver Plots can also be automatically contoured by clicking the last top Check Box allowing the user to highlight the cell when such single cell contouring is possible.",
            "Fig 56 - Last Check Box in red and contouring drawn for the Jet visualization.",
            "The user can also move between different frames of the Contraction-Relaxation cycle by clicking and dragging the bottom slider or by clicking directly into the bottom plot.",
            "Fig 57 - Moving between different frames of the Image/Jet visualization via the slider or plot clicks highlighted in red.",
            "Alternatively, the user can also play an animation of the full cycle by clicking the Play/Stop button and adjusting the Frame Per Second rate Spin Box at the bottom of the Screen. This can sometimes be laggy since plots are dynamically drawn and not pre-bufferized at this time.",
            "Fig 58 - The Play/Stop Button and the Frame Per Second rate Spin Box highlighted in red.",
            "Title Adjusting Automatic Contour Settings",
            "Advanced settings can be adjusted by clicking the Advanced Configs Menu Option in the Top Bar and editing the parameters at the next Pop Up Window. The Quiver X and Y windows can be adjusted for controlling the size of the Quiver Plot horizontal and vertical directions.",
            "Fig 59 - Advanced settings Pop up Window.",
            "Blur Size, Kernal Dilation, Kernel Erosion, Kernal Smoothing Contours, are all parameters related to the respective OpenCV functions: OpenCV Docs Filtering - See: blur(), erode(), dilate(), morphologyEx() in order.",
            "Likewise, the Border thickness parameter refers to the Contour Thickness of the respective OpenCV function: OpenCV Docs Structural Analysis and Shape Descriptors - See drawContours().",
            "The Scale Min. and Scale Max. parameters refer to the drawn Colorbar Legend limits and can be changed to match a user’s needs when drawing publication ready figures.",
            "All parameters are updated in real time when changed in this Pop Up Window.",
            "Title Exporting Data",
            "Various export options exist for this Window. Besides being able to export the current plot and plot data as a figure and a table, respectively,  each of the seven possible image combinations can be exported in various formats either for the currently selected frame or for all frames of the selected Wave. An additional exporting option is available is possible when all frames are selected which is exporting as a video ('.avi' extension). The Legend colorbox can also be separately exported.",
            "Fig 60 - Exporting Data Options for the Jet/Quiver Plot Screen."]
        }
        
        self.helpframe = ttkScrollFrame(master, 'greyBackground.TFrame', self.parent._get_bg_color())
        self.helpframe.viewPort.columnconfigure(0, weight=1)
        self.helpframe.grid(row=1,column=0, rowspan=1, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

        tlbl = ttk.Label(master,text="Help:", wraplength=600, width=100)
        tlbl.grid(row=0, column=0, rowspan=1, columnspan=1)#, sticky=tk.W+tk.E+tk.N+tk.S)

        master.update()
        self.parent.update()
        self.update_idletasks()
        self.update()
        self.imglbls = []
        helpframewidth = tlbl.winfo_width()
        towrite = pagenames[self.literals["current_help"]]
        rown = 0
        for i, paragraph in enumerate(towrite):
            thefont = self.normal_font
            if paragraph[:5] == "Title":
                thefont = self.title_font
                paragraph = paragraph[5:]
            paragraph = "\n" + paragraph + "\n"

            newlbl = ttk.Label(self.helpframe.viewPort, font=thefont, text=paragraph, wraplength=800, anchor="w", justify=tk.LEFT ) 
            newlbl.grid(row=rown, column=0, rowspan=1, columnspan=1)
            self.helpframe.viewPort.rowconfigure(rown, weight=1)
            rown +=1
            # print(paragraph[1:4])
            # print(paragraph[0])
            # print(paragraph[1])
            # print(paragraph[2])
            # print(paragraph[3])
            # print(paragraph[4])
            if paragraph[1:4] == "Fig":
                print("FIG!!!")
                newlblframe = ttk.Frame(self.helpframe.viewPort)
                photo = load_img("tutorial_imgs/" + paragraph.split()[0] + "_" +  paragraph.split()[1] + ".png", int(helpframewidth * 0.8))
                self.imglbls.append(photo)
                newlblimg = ttk.Label(newlblframe, image=self.imglbls[-1])
                newlblimg.image = self.imglbls[-1] # keep a reference!
                newlblimg.pack()
                newlblframe.grid(row=rown, column=0, rowspan=1, columnspan=1)
                self.helpframe.viewPort.rowconfigure(rown, weight=1)
                rown += 1
        self.packbtns = False

    def validate(self):
        print("class AboutDialog def validate start")
        self.valid = False
        return 0

    def apply(self):
        print("class AboutDialog def apply start")
        pass

class AboutDialog(tkDialog.Dialog):
    def body(self, master):
        print("class AboutDialog def body creation")
        ttk.Label(master, text='ContractionWave: Free Tool to easy visualize, quantify and analyze cell contractility\n').grid(row=0, column=1)
        ttk.Label(master, text='Version: 1.0Py\n').grid(row=1, column=1)
        ttk.Label(master, text='Scalzo, S.; Lima Afonso, M.Q.L.; Fonseca, N.J.;Jesus, I. C. G.; Alves, A. P.;Teixeira,V.P.;\nMarques, F.A.M.;Mesquita, O. N.; Kushmerick, C; Bleicher, L.;Agero, U.; Guatimosim, S.\n').grid(row=2, column=1)
        ttk.Label(master, text='To be published\n').grid(row=3, column=1)
        ttk.Label(master, text='This software is registered under GPL3.0.\nContact: sergiosc1789 at gmail.com\n').grid(row=4, column=1)
        ttk.Label(master, text='Third party copyrights are property of their respective owners.\n\nCopyright (C) 2020, Ionicons, Released under MIT License.\nCopyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, All rights reserved.\nCopyright (c) 2012-2013 Matplotlib Development Team; All Rights Reserved\nCopyright (C) 2000-2019, Intel Corporation, all rights reserved.\n Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.\n Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.\n Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.\n Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.\n Copyright (C) 2015-2016, Itseez Inc., all rights reserved.\n Copyright (C) 2019-2020, Xperience AI, all rights reserved.\nCopyright (C) 2018 Uwe Klimmek\nCopyright Regents of the University of California, Sun Microsystems, Inc., Scriptics Corporation, and other parties\nCopyright © 1997-2011 by Secret Labs AB\nCopyright © 1995-2011 by Fredrik Lundh\nCopyright © 2010-2020 by Alex Clark and contributors\nCopyright (c) 2013-2020, John McNamara <jmcnamara@cpan.org> All rights reserved.\nCopyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola, All rights reserved. \n Copyright (c) 2005, NumPy Developers\n Copyright (c) 2010-2020, PyInstaller Development Team\n Copyright (c) 2005-2009, Giovanni Bajo\n Based on previous work under copyright (c) 2002 McMillan Enterprises, Inc.\n Copyright © 2001, 2002 Enthought, Inc. All rights reserved. \n Copyright © 2003-2019 SciPy Developers. All rights reserved.\n ').grid(row=5, column=1)

        self.packbtns = False

    def validate(self):
        print("class AboutDialog def validate start")
        self.valid = False
        return 0

    def apply(self):
        print("class AboutDialog def apply start")
        pass

class FolderSelectDialog(tkDialog.Dialog):
    def body(self, master):
        print("class FolderSelectDialog def body creation")
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
        print("class FolderSelectDialog def validate start")
        if self.result != None:
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        print("class FolderSelectDialog def apply start")
        pass

class CoreAskDialog(tkDialog.Dialog):

    def body(self, master):
        print("class CoreAskDialog def body creation")
        ttk.Label(master, text='Start Processing Queue:').grid(row=0, column=0)

        self.formatvar = tk.StringVar(master)
        self.formatchoices = range(1, self.literals["maxcores"]+1)
        self.formatvar.set(self.literals["maxcores"])
        self.optmenu = ttk.OptionMenu(master, self.formatvar, self.literals["maxcores"], *self.formatchoices)
        ttk.Label(master, text='Core Number:').grid(row=1, column=0)
        self.optmenu.grid(row = 1, column = 1)

    def validate(self):
        print("class CoreAskDialog def validate start")
        #nothing to validate
        if int(self.formatvar.get()) > 0:
            self.valid = True
            return 1
        else:
            self.valid = False
            return 0

    def apply(self):
        print("class CoreAskDialog def apply start")
        #save configs
        self.result = int(self.formatvar.get())

class SelectMenuItem(tkDialog.Dialog):

    def body(self, master):
        print("class SelectMenuItemN def body creation")
        ttk.Label(master, text='Type New Name:').grid(row=0, column=1)
        self.AnswerVar = tk.StringVar()
        self.AnswerVar.set(self.literals["current_name"])
        ttk.Entry(master, width=10, textvariable=self.AnswerVar).grid(row=1,column=1)
        self.AnswerVar.set(self.literals["current_name"])

    def validate(self):
        print("class SelectMenuItem def validate start")
        #nothing to validate
        if len(str(self.AnswerVar.get())) > 0:
            self.valid = True
            return 1
        else:
            self.valid = False
            return 0

    def apply(self):
        print("class SelectMenuItemN def apply start")
        #save configs
        self.result = str(self.AnswerVar.get())

class AddPresetDialog(tkDialog.Dialog):

    def body(self, master):
        print("class AddPresetDialog def body creation")
        ttk.Label(master, text='New Preset Name:').grid(row=0, column=1)
        self.AnswerVar = tk.StringVar()
        self.AnswerVar.set("")
        ttk.Entry(master, width=10, textvariable=self.AnswerVar).grid(row=1,column=1)
        self.result = None

    def validate(self):
        print("class AddPresetDialog def validate start")
        #nothing to validate
        if self.AnswerVar.get() != "":
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        print("class AddPresetDialog def apply start")
        #save configs
        if self.valid == True:
            self.result = self.AnswerVar.get()
        else:
            self.result = None

class DotChangeDialog(tkDialog.Dialog):

    def body(self, master):
        print("class DotChangeDialog def body creation")
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
        print("class DotChangeDialog def validate start")
        #nothing to validate
        if self.DType.get() != "":
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        print("class DotChangeDialog def apply start")
        #save configs
        if self.valid == True:
            self.result = self.DType.get().lower()
        else:
            self.result = None

class SelectPresetDialog(tkDialog.Dialog):

    def body(self, master):
        print("class SelectPresetDialog def body creation")
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
        print("class SelectPresetDialog def validate start")
        #nothing to validate
        if self.PresetSelect.get() != "":
            self.valid = True
            return 1
        self.valid = False
        return 0

    def apply(self):
        print("class SelectPresetDialog def apply start")
        #save configs
        if self.valid == True:
            self.result = self.literals["preset_dicts"][self.PresetSelect.get()]
        else:
            self.result = None

class PlotSettingsProgress(tkDialog.Dialog):

    def body(self, master):
        print("class PlotSettingsProgress def body creation")
        ttk.Label(master, text='Edit Plot Settings:').grid(row=0, column=0, columnspan=2)
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
            "rect_color": 'Set Sel. Areas Color'
        }
        self.current_settings = self.literals["plotsettingscolors"]
        self.current_lineopts = self.literals["plotsettingsline"]
        i = 0
        #i = 1
        for j, k in enumerate(sorted(self.current_settings.keys())):
            coln = 1
            if j % 2 == 0:
                i += 1
                coln = 0
            thisButton = ttk.Button(master, text=keytextconversion[k],width=20)
            thisButton.texttype = k
            thisButton.bind("<Button-1>", self.change_color)
            thisButton.grid(row=i, column=coln)
            # i += 1
        i += 1
        print("create sep")
        sep = ttk.Separator(master, orient='horizontal')
        sep.grid(row=i, column=0, columnspan=2, sticky="ew", ipadx=20)
        i += 1
        # for l in sorted(self.current_lineopts.keys()):
        checklvarstatus = 0
        if self.current_lineopts["zero"] == True:
            checklvarstatus = 1
        self.thisCheck_line_var = tk.IntVar()
        self.thisCheck_line_var.set(checklvarstatus)
        self.thisCheck_line = ttk.Checkbutton(master, text="Plot X Axis Baseline", variable = self.thisCheck_line_var,width=20)
        self.thisCheck_line.texttype = "zero"
        self.thisCheck_line.grid(row=i, column=0)

        self.zeroButton = ttk.Button(master, text="Baseline Color",width=20)
        self.zeroButton.texttype = "zero_color"
        self.zeroButton.bind("<Button-1>", self.change_color2)
        self.zeroButton.grid(row=i, column=1)

        i += 1
        checkgvarstatus = 0
        if self.current_lineopts["grid"] == True:
            checkgvarstatus = 1
        self.thisCheck_grid_var = tk.IntVar()
        self.thisCheck_grid_var.set(checkgvarstatus)
        self.thisCheck_grid = ttk.Checkbutton(master, text="Plot Gridlines", variable = self.thisCheck_grid_var,width=20)
        self.thisCheck_grid.texttype = "grid"
        self.thisCheck_grid.grid(row=i, column=0)

        self.gridButton = ttk.Button(master, text="Gridlines Color",width=20)
        self.gridButton.texttype = "grid_color"
        self.gridButton.bind("<Button-1>", self.change_color2)
        self.gridButton.grid(row=i, column=1)
        i += 1

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
        ttk.Label(master, text="Used Time Unit:").grid(row=i, column=0, columnspan=2)
        i+=1
        self.optmenu.grid(row=i, column=0, columnspan=2)
        self.TimeSelect.set(fkey)
        i+=1

        checkavarstatus = 0
        if self.current_lineopts["absolute_time"] == True:
            checkavarstatus = 1
        self.thisCheck_absolute_var = tk.IntVar()
        self.thisCheck_absolute_var.set(checkavarstatus)
        self.thisCheck_absolute = ttk.Checkbutton(master, text="Peak Absolute Time", variable = self.thisCheck_absolute_var,width=20)
        self.thisCheck_absolute.texttype = "absolute_time"
        self.thisCheck_absolute.grid(row=i, column=0, columnspan=2)

        print("class PlotSettingsProgress def body created")

        # self.current_lineopts 

    def change_color(self, event):
        print("class PlotSettingsProgress def change_color started")
        k = event.widget.texttype
        newcolor = colorchooser.askcolor(title="Select Color for " + k, color=self.current_settings[k])
        print("class PlotSettingsProgress def change_color current color is: " + self.current_settings[k])
        if newcolor[1] != None:
            print("class PlotSettingsProgress def change_color color changed!")
            self.current_settings[k] = newcolor[1]
            print("class PlotSettingsProgress def change_color new color is:" + newcolor[1])
        print("class PlotSettingsProgress def change_color done")
    
    def change_color2(self, event):
        print("class PlotSettingsProgress def change_color2 started")
        k = event.widget.texttype
        newcolor = colorchooser.askcolor(title="Select Color for " + k, color=self.current_lineopts[k])
        print("class PlotSettingsProgress def change_color2 current color is: " + self.current_lineopts[k])
        if newcolor[1] != None:
            print("class PlotSettingsProgress def change_color2 color changed!")
            self.current_lineopts[k] = newcolor[1]
            print("class PlotSettingsProgress def change_color2 new color is:" + newcolor[1])
        print("class PlotSettingsProgress def change_color2 done")
    
    def set_timetype(self, event):
        print("class PlotSettingsProgress def set_timetype started")
        self.current_lineopts["time_unit"] = self.full_time_dict[self.TimeSelect.get()]
        print("class PlotSettingsProgress def set_timetype done")
    
    def validate(self):
        print("class PlotSettingsProgress def validate start")
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
        print("class PlotSettingsProgress def validate done")
        return 1

    def apply(self):
        print("class PlotSettingsProgress def apply start")
        #save configs
        self.result = {}
        print(self.current_settings.copy())
        print(self.current_lineopts.copy())
        self.result["peak_plot_colors"] = self.current_settings.copy()
        self.result["plotline_opts"] = self.current_lineopts.copy()
        print("class PlotSettingsProgress def apply done")
        # return self.current_settings[k]

class WaitDialogProgress(tkDialog.DialogBlockNonGrab):
# class WaitDialogProgress(tkDialog.DialogNonBlock):

    def body(self, master):
        ttk.Label(master, text="Please wait until processing is done...").grid(row=0,column=0, columnspan=3)

    def validate(self):
        print("class AboutDialog def validate start")
        self.valid = False
        return 0

    def apply(self):
        print("class AboutDialog def apply start")
        pass

class QuiverJetMaximize(tkDialog.DialogNonBlockMax):

    def body(self, master):
        print("class QuiverJetMaximize def body creation")
        self.master_window = master

        # self.figmax = plt.figure(figsize=(4, 3), dpi=100 ,facecolor=master.cget('bg'), edgecolor="None")
        # self.mbg = master.cget('bg')
        self.figmax = plt.figure(figsize=(4, 3), dpi=100 ,facecolor="black", edgecolor="None")
        
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
        print("class QuiverJetMaximize def body done")

    def set_figure(self, figx, figy):
        self.figmax.set_size_inches(figx/100, figy/100, forward=True)
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
        print("class QuiverJetMaximize def validate start")
        self.valid = True
        return 1

    def apply(self):
        print("class QuiverJetMaximize def apply start")
        self.literals["updatable_frame"].clear_maximize()
        pass

class QuiverJetSettings(tkDialog.DialogNonBlock):

    def body(self, master):
        print("class QuiverJetSettings def body creation")
        ttk.Label(master, text='Advanced Configurations:').grid(row=0, column=0, rowspan=1)
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
        ttk.Label(master,text=self.literals["blur_size"][0]).grid(row=rown, column=0)
        self.blur_size_spin = tk.Spinbox(master, from_=self.literals["config"]["blur_size"][0], to=self.literals["config"]["blur_size"][1], increment=2, width=10, command=lambda: self.up_frame(self.blur_size_spin, "blur_size"))
        self.blur_size_spin.thistype = "blur_size" 
        self.blur_size_spin.grid(row=rown, column=1)
        self.blur_size_spin.delete(0,"end")
        self.blur_size_spin.insert(0,self.literals["blur_size"][1])
        self.blur_size_spin.bind('<Return>', lambda *args: self.up_frame(self.blur_size_spin, "blur_size"))

        rown += 1
        ttk.Label(master,text=self.literals["kernel_dilation"][0]).grid(row=rown, column=0)
        self.kernel_dilation_spin = tk.Spinbox(master, from_=self.literals["config"]["kernel_dilation"][0], to=self.literals["config"]["kernel_dilation"][1], increment=1, width=10, command=lambda: self.up_frame(self.kernel_dilation_spin, "kernel_dilation"))
        self.kernel_dilation_spin.thistype = "kernel_dilation" 
        self.kernel_dilation_spin.grid(row=rown, column=1)
        self.kernel_dilation_spin.delete(0,"end")
        self.kernel_dilation_spin.insert(0,self.literals["kernel_dilation"][1])
        self.kernel_dilation_spin.bind('<Return>', lambda *args: self.up_frame(self.kernel_dilation_spin, "kernel_dilation"))
       
        rown += 1
        ttk.Label(master,text=self.literals["kernel_erosion"][0]).grid(row=rown, column=0)
        self.kernel_erosion_spin = tk.Spinbox(master, from_=self.literals["config"]["kernel_erosion"][0], to=self.literals["config"]["kernel_erosion"][1], increment=1, width=10, command=lambda: self.up_frame(self.kernel_erosion_spin, "kernel_erosion"))
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
        ttk.Label(master,text=self.literals["minscale"][0]).grid(row=rown, column=0)
        self.minscale_spin = tk.Spinbox(master, from_=self.literals["config"]["minscale"][0], to=self.literals["config"]["minscale"][1], increment=1, width=10, command=lambda: self.up_frame(self.minscale_spin, "minscale"))
        self.minscale_spin.thistype = "minscale"
        self.minscale_spin.grid(row=rown, column=1)
        self.minscale_spin.delete(0,"end")
        self.minscale_spin.insert(0,self.literals["minscale"][1])
        self.minscale_spin.bind('<Return>', lambda *args: self.up_frame(self.minscale_spin, "minscale"))

        rown += 1
        ttk.Label(master,text=self.literals["maxscale"][0]).grid(row=rown, column=0)
        self.maxscale_spin = tk.Spinbox(master, from_=self.literals["config"]["maxscale"][0], to=self.literals["config"]["maxscale"][1], increment=1, width=10, command=lambda: self.up_frame(self.maxscale_spin, "maxscale"))
        self.maxscale_spin.thistype = "maxscale"
        self.maxscale_spin.grid(row=rown, column=1)
        self.maxscale_spin.delete(0,"end")
        self.maxscale_spin.insert(0,self.literals["maxscale"][1])
        self.maxscale_spin.bind('<Return>', lambda *args: self.up_frame(self.maxscale_spin, "maxscale"))
        
        print("class QuiverJetSettings def body creation done")
        
        # tk.Label(master, text=literals[k][0]).grid(row=rown, column=0)
        # tk.Spinbox(master, from_=self.literals["config"][k][0], to=self.literals["config"][k][0], increment=1, width=10).grid(row=rown, column=1)

    def up_frame(self, current, event=None):
        #TODO: CORRECT THIS
        print("class QuiverJetSettings def up_frame")
        if event != None:
            print("class QuiverJetSettings def up_frame event")
            print(event)
            valid = self.validate()
            if valid == True:
                self.literals["updatable_frame"].update_config(event,int(current.get().replace(",", ".")))
            # print(event.widget)
            # print(event.widget.thistype)
        # self.literals["updatable_frame"].update_frame()
        # self.literals["updatable_frame"].update()
        pass

    def validate(self):
        print("class QuiverJetSettings def validate start")
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
            kernel_erosion_spin_val = int(self.kernel_erosion_spin.get().replace(",", "."))
            if kernel_erosion_spin_val < self.literals["config"]["kernel_erosion"][0] or kernel_erosion_spin_val > self.literals["config"]["kernel_erosion"][1]:
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
            maxscale_val = float(self.maxscale_spin.get().replace(",", "."))
            if maxscale_val < self.literals["config"]["maxscale"][0] or maxscale_val > self.literals["config"]["maxscale"][1]:
                self.valid = False
                messagebox.showwarning(
                    "Bad input",
                    "Illegal values, please try again"
                )
                return 0
            if maxscale_val <= minscale_val:
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
        print("class QuiverJetSettings def apply start")
        #save configs
        self.result = None
        print("self.valid")
        print(self.valid)
        if self.valid == True:
            self.result = {}
            print(self.result)
            self.result["current_windowX"] = int(self.current_windowX_spin.get().replace(",", "."))
            self.result["current_windowY"] = int(self.current_windowY_spin.get().replace(",", "."))
            self.result["blur_size"] = int(self.blur_size_spin.get().replace(",", "."))
            self.result["kernel_dilation"] = int(self.kernel_dilation_spin.get().replace(",", "."))
            self.result["kernel_erosion"] = int(self.kernel_erosion_spin.get().replace(",", "."))
            self.result["kernel_smoothing_contours"] = int(self.kernel_smoothing_contours_spin.get().replace(",", "."))
            self.result["border_thickness"] = int(self.border_thickness_spin.get().replace(",", "."))
            self.result["minscale"] = float(self.minscale_spin.get().replace(",", "."))
            self.result["maxscale"] = float(self.maxscale_spin.get().replace(",", "."))
            print(self.result)
            self.literals["updatable_frame"].update_all_settings(self.result)
        return True

class AdjustNoiseDetectDialog(tkDialog.Dialog):

    def body(self, master):
        print("class AdjustNoiseDetectDialog def body creation")
        ttk.Label(master, text='Noise Advanced Parameters:').grid(row=0, column=0)
        self.checkfvar = tk.IntVar(value=self.literals["noiseareasfiltering"])
        self.checkf = ttk.Checkbutton(master, text = "Noise Areas Min. Size filtering", variable = self.checkfvar, \
                         onvalue = 1, offvalue = 0)
        self.check_d_var = tk.IntVar(value=self.literals["noisedecrease"])
        self.checkd = ttk.Checkbutton(master, text = "Decrease Avg. Noise from Plot", variable = self.check_d_var, \
                         onvalue = 1, offvalue = 0, command=self.hidespin)
        
        self.check_u_var = tk.IntVar(value=self.literals["userdecrease"])
        self.checku = ttk.Checkbutton(master, text = "Decrease Custom Value from Plot", variable = self.check_u_var, \
                         onvalue = 1, offvalue = 0, command=self.showspin)
        
        self.spinlbl = ttk.Label(master, text= "Custom Value: ")
        self.spinu = tk.Spinbox(master,from_=-10000000000, to=10000000000, increment=1, width=10)
        
        self.spinu.bind('<Return>', lambda *args: self.validate())

        print('self.literals["noisedecreasevalue"]')
        print(self.literals["noisedecreasevalue"])
        self.spinu.delete(0,"end")
        self.spinu.insert(0,self.literals["noisedecreasevalue"])

        self.checkf.grid(row=1, column=0)
        self.checkd.grid(row=2, column=0)
        self.checku.grid(row=3, column=0)

        self.spinlbl.grid(row=4, column=0)
        self.spinu.grid(row=4, column=1)
        if self.check_u_var.get() == 0:
            self.spinlbl.grid_forget()
            self.spinu.grid_forget()
        print("class AdjustNoiseDetectDialog def body created")
    
    def showspin(self):
        print("class AdjustNoiseDetectDialog def showspin start")
        if self.check_u_var.get() == 0:
            self.spinu.grid_forget()
            self.spinlbl.grid_forget()
        else:
            self.check_d_var.set(0)
            self.spinlbl.grid(row=4, column=0)
            self.spinu.grid(row=4, column=1)
        print("class AdjustNoiseDetectDialog def showspin end")

    def hidespin(self):
        print("class AdjustNoiseDetectDialog def hidespin start")
        # if self.check_u_var.get() == 0:
        # else:
        #     self.check_d_var.set(0)
        #     self.spinlbl.grid(row=4, column=0)
        #     self.spinu.grid(row=4, column=1)
        if self.check_d_var.get() == 1:
            self.check_u_var.set(0)
            self.spinu.grid_forget()
            self.spinlbl.grid_forget()
        print("class AdjustNoiseDetectDialog def hidespin end")

    def validate(self):
        print("class AdjustNoiseDetectDialog def validate start")
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
        print("class AdjustNoiseDetectDialog def validate done")
        return 1

    def apply(self):
        print("class AdjustNoiseDetectDialog def apply start")
        #save configs
        self.result = {}
        if self.checkfvar.get() == 1:
            self.result["adjustnoisevar"] = True
        else:
            self.result["adjustnoisevar"]= False
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
        print("class AdjustNoiseDetectDialog def apply done")

class SaveFigureVideoDialog(tkDialog.Dialog):

    def body(self, master):
        print("class SaveFigureVideoDialog def body creation")
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
        self.dpi = tk.StringVar()
        self.dpi.set("300")
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
        print("args")
        print(args)
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
        print("class SaveFigureVideoDialog def validate start")
        try:
            first= int(self.dpi.get())
            #third = int(self.e3var.get())
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
        print("class SaveFigureVideoDialog def apply start")
        #save configs
        self.result = None
        if self.valid == True:
            self.result = {}
            fname = filedialog.asksaveasfile(defaultextension="." + self.formatvar.get())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
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
            self.result["quality"] = float(self.quality.get())
            bboxv = self.bboxvar.get()
            if bboxv == "None":
                bboxv = None
            self.result["bbox"] = bboxv

class SaveFigureDialog(tkDialog.Dialog):

    def body(self, master):
        print("class SaveFigureDialog def body creation")

        nrow = 0
        ttk.Label(master, text='Figure Format:').grid(row=nrow, column=0)
        self.formatvar = tk.StringVar(master)
        self.formatchoices = self.literals["formats"]
        self.formatvar.set('png')
        self.optmenu = ttk.OptionMenu(master, self.formatvar, "png", *self.formatchoices, command=self.jpg_detection)
        self.optmenu.grid(row = nrow, column = 1)
        nrow += 1
        self.dpi = tk.StringVar()
        self.dpi.set("300")
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
        print("args")
        print(args)
        self.formatvar.set(args)
        if self.formatvar.get() == "jpg" or self.formatvar.get() == "jpeg":
            self.jpglabel.grid(row=3, column=0)
            self.qualityspin.grid(row=3, column=1)
        else:        
            self.jpglabel.grid_forget()
            self.qualityspin.grid_forget()

    def validate(self):
        print("class SaveFigureDialog def validate start")
        try:
            first= int(self.dpi.get())
            #third = int(self.e3var.get())
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
        print("class SaveFigureDialog def apply start")
        #save configs
        self.result = None
        if self.valid == True:
            self.result = {}
            fname = filedialog.asksaveasfile(defaultextension="." + self.formatvar.get())
            if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            fname.close()
            fnamename = str(fname.name)
            filename = r'%s' %fnamename
            os.remove(str(filename))
            self.result["name"] = str(filename).split(".")[0] + "." + self.formatvar.get()
            self.result["format"] = self.formatvar.get()
            self.result["dpi"] = int(self.dpi.get())
            self.result["quality"] = float(self.quality.get())
            bboxv = self.bboxvar.get()
            if bboxv == "None":
                bboxv = None
            self.result["bbox"] = bboxv

class SaveTableDialog(tkDialog.Dialog):

    def body(self, master):
        print("class SaveTableDialog def body creation")
        # group = self.literals["current_group"]
        ttk.Label(master, text='Table Format:').grid(row=0, column=0)
        # Create a Tkinter variable
        self.formatvar = tk.StringVar(master)

        # Dictionary with options
        if _platform == "linux" or _platform == "linux2":
            self.formatchoices = { 'CSV','XLS'}
        else:
            self.formatchoices = { 'CSV','XLSX'}
        self.formatvar.set('CSV') # set the default option

        ttk.OptionMenu(master, self.formatvar, "CSV", *self.formatchoices).grid(row = 0, column = 1)

    def validate(self):
        print("class SaveTableDialog def validate start")
        self.valid = True
        return 1

    def apply(self):
        print("class SaveTableDialog def apply start")
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
        try:
            format_sel = self.formatvar.get()
            data_type = self.literals["data_t"]
            data = np.array(self.literals["data"])
            print("data")
            print(data)
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
                print("sheets")
                print(sheets)
                fname = filedialog.asksaveasfile(defaultextension="." + format_sel.lower())
                if fname is None: # asksaveasfile return `None` if dialog closed with "cancel".
                    return
                fname.close()
                fnamename = str(fname.name)
                filename = r'%s' %fnamename
                os.remove(str(filename))
                print("filename")
                print(filename)
                writer = None
                workbook = None
                header_format = None
                if format_sel == "XLS" or format_sel == "XLSX":
                    writer = pd.ExcelWriter(str(filename).split(".")[0] + "." + format_sel.lower(), engine='xlsxwriter', mode='w')
                    workbook = writer.book
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'align': 'center','border': 1})
                for e_header, e_data, e_sheet in zip(headers, data, sheets):
                    new_dict = {}
                    new_df = None
                    for header, data_col in zip(e_header, e_data):
                        print("header")
                        print(header)
                        print("data_col")
                        print(data_col)
                        new_dict[header] = data_col.copy()
                    new_df = pd.DataFrame(new_dict)
                    if format_sel == "CSV":
                        new_df.to_csv(str(filename).split(".")[0] + "_" + e_sheet + "." + str(filename).split(".")[1], index=False)
                    elif format_sel == "XLS" or format_sel == "XLSX":
                        new_df.to_excel(writer, sheet_name=e_sheet,index=False)
                        worksheet = writer.sheets[e_sheet]

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
                if format_sel == "XLS" or format_sel == "XLSX":
                    writer.save()
                messagebox.showinfo(
                    "File saved",
                    "File was successfully saved"
                )
        except Exception as e:
            messagebox.showerror("Error", "Could not save Data file\n" + str(e))

class SummarizeTablesDialog(tkDialog.Dialog):
    def body(self, master):
        print("class SummarizeTablesDialog def body creation")

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
        print("class SummarizeTablesDialog def body created")
    
    def add_table(self, event=None):
        print("class SummarizeTablesDialog def add_table start")
        print("class SummarizeTablesDialog def add_table open file dialog selection")
        filenames = filedialog.askopenfilenames(parent=self,title='Choose Data Tables to Summarize:',filetypes = (("XLS Files","*.xls"),("XLSX Files","*.xlsx")))
        print("class SummarizeTablesDialog def add_table open file dialog done")
        print("filenames")
        print(filenames)
        if filenames:
            # filenames = [r'%s' for fna in filenames]
            newfilenames = []
            for fna in filenames:
                newfilenames.append(r'%s' %fna)
            print("class SummarizeTablesDialog def add_table 1 or more files selected")
            for filename in newfilenames:
                print("class SummarizeTablesDialog def add_table inserting file in listbox")
                self.listbox.insert(tk.END, filename)
        print("class SummarizeTablesDialog def add_table done")

    
    def delete_table(self, event=None):
        print("class SummarizeTablesDialog def delete_table start")
        self.listbox.delete(self.listbox.curselection()[0])
        print("class SummarizeTablesDialog def delete_table done")

    def validate(self):
        print("class SummarizeTablesDialog def validate start")
        items = self.listbox.get(0, tk.END)
        print("items")
        print(items)
        if len(items) > 0:
            for item in items:
                xl = pd.ExcelFile(item)
                names = xl.sheet_names  # see all sheet names
                if "Avg. Time" not in names or "Avg. Speed" not in names or "Avg. Area" not in names:
                    messagebox.showerror("Error", "No tables selected")
                    self.valid = False
                    return 0
            self.valid = True
            return 1
        else:
            messagebox.showerror("Error", "No tables selected")
            self.valid = False
            return 0

        print("class SummarizeTablesDialog def validate done")

    def apply(self):
        print("class SummarizeTablesDialog def apply start")
        timeavgdf = {
            "Name": []
        }
        speedvgdf = {
            "Name": []
        }
        areaavgdf = {
            "Name": []
        }
        for item in self.listbox.get(0, tk.END):
            xl = pd.ExcelFile(item)
            timedf = xl.parse("Avg. Time")  # read a specific sheet to DataFrame
            k1 = timedf.keys()[0]
            v1 = timedf[k1]
            for v in v1:
                timeavgdf["Name"].append(os.path.basename(item))
            for key in timedf.keys():
                if key not in timeavgdf.keys():
                    timeavgdf[key] = []
                values = timedf[key]
                for value in values:
                    timeavgdf[key].append(value)
            speeddf = xl.parse("Avg. Speed")  # read a specific sheet to DataFrame
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
            areadf = xl.parse("Avg. Area")  # read a specific sheet to DataFrame
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
            print("filename")
            print(filename)

            writer = pd.ExcelWriter(str(filename).split(".")[0] + format_do, engine='xlsxwriter', mode='w')
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'align': 'center','border': 1})
            timeavgdf = pd.DataFrame(timeavgdf)
            timeavgdf.to_excel(writer, sheet_name="Time. Concat",index=False)
            worksheet = writer.sheets["Time. Concat"]
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
            speedvgdf.to_excel(writer, sheet_name="Speed. Concat",index=False)
            worksheet = writer.sheets["Speed. Concat"]
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
            areaavgdf.to_excel(writer, sheet_name="Area. Concat",index=False)
            worksheet = writer.sheets["Area. Concat"]
            for idx, col in enumerate(areaavgdf):  # loop through all columns
                series = areaavgdf[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
            for col_num, value in enumerate(areaavgdf.columns.values):
                worksheet.write(0, col_num, value, header_format)

            writer.save()
            messagebox.showinfo(
                "File saved",
                "File was successfully saved"
            )
        except Exception as e:
            print(str(e))
            messagebox.showerror("Error", "Could not save Data file\n" + str(e))

        print("class SummarizeTablesDialog def apply end")

class SavGolDialog(tkDialog.Dialog):

    def body(self, master):
        print("class SavGolDialog def body creation")

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
        print("class SavGolDialog def validate start")
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
        print("class SavGolDialog def apply start")
        first = int(self.e1.get().replace(",", "."))
        second = int(self.e2.get().replace(",", "."))
        self.result = first, second

        # print first, second # or something

class NpConvDialog(tkDialog.Dialog):

    def body(self, master):
        print("class NpConvDialog def body creation")

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
        print("class NpConvDialog def validate start")
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
        print("class NpConvDialog def apply start")
        first = int(self.e1.get().replace(",", "."))
        self.result = first, self.formatvar.get()

class FourierConvDialog(tkDialog.Dialog):

    def body(self, master):
        print("class FourierConvDialog def body creation")
        ttk.Label(master, text="% of Frequencies kept:").grid(row=0, column=0)

        self.e1 = tk.Spinbox(master, from_=self.literals["freqstart"], to=self.literals["maxvalues"], increment=0.1, width=10)
        self.e1.grid(row=0, column=1)
        self.e1.bind('<Return>', lambda *args: self.validate())


        self.valid = False

        print("class FourierConvDialog def body created")
        return self.e1 # initial focus

    def validate(self):
        print("class FourierConvDialog def validate start")
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
            print("class FourierConvDialog def validate true")
            return 1
        except ValueError:
            messagebox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )
            self.valid = False
            print("class FourierConvDialog def validate false")
            return 0

    def apply(self):
        print("class FourierConvDialog def apply start")
        first = float(self.e1.get().replace(",", "."))
        self.result = first
        print("class FourierConvDialog def apply done")