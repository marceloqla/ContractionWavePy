# ContractionWavePy

## ABOUT

CONTRACTIONWAVE (CW) is a free software developed in Python Programming Language that allows the user to visualize, quantify, and analyze cell contractility parameters in a simple and intuitive format. The software enables the user to acquire membrane kinetics data of cell contractility during contraction-relaxation cycles through image capture and a dense optical flow algorithm. Both method and software were developed using multidisciplinary knowledge, which resulted in a robust data extraction protocol.

For more information access: https://sites.icb.ufmg.br/cardiovascularrc/contractionwave/

## GETTING STARTED
Contractionwave can be installed from an executable file (see below section – for
Windows or Ubuntu) or from Anaconda environment installation (see second section below –
for Windows, Ubuntu, or Mac-OS).

### Installation – Executable file
**Step 1**: The latest stable version compiled executables **for Windows and Ubuntu (64-
bit)** can be downloaded from:
https://sites.icb.ufmg.br/cardiovascularrc/contractionwave (Choose the Operating
System and click on Download)

**Step 2**: Unpack the File

**Step 3**: The extracted directory contains a library folder and one executable file (.exe
for Windows)

**Step 4**: Double-click the CW icon to start the executable file

### Installation – Anaconda environment
#### I- Anaconda environment installation (should only be executed
once)

The latest stable version **for Windows and Ubuntu (64-bit) or Mac-OS**

**Step 1: Download the program**

**a)** Access the page https://github.com/marceloqla/ContractionWavePy

**b)** Click on the "Code" green button on the right hand side of the screen

**c)** Click on the "Download ZIP" button.

**d)** Unpack the file and complete the extraction process to a desired target directory

**Step 2: Installation using Anaconda**

Follow the instructions for installing Anaconda:

**Windows**: https://docs.anaconda.com/anaconda/install/windows/

**Mac-OS**: https://docs.anaconda.com/anaconda/install/mac-os/

**Linux**: https://docs.anaconda.com/anaconda/install/linux/

#### **II- ContractionWave environment installation with Anaconda Prompt (should only be executed once)**

**Step 3: Open Anaconda Prompt**

**Windows**: Click Start, search, or select Anaconda Prompt from the menu.

**Mac-OS:** Cmd+Space to open Spotlight Search and type “Navigator” to open the
program.

**Ubuntu:** Open the Dash by clicking the upper left Ubuntu icon, then type “terminal”.

Please refer to other materials for opening the Terminal in other Operating Systems.

**Step 4: Access the directory in which the ZIP file was extracted**

For example, if the full path for the directory is:

*"/Users/PC-name/Desktop/ContractionWavePy/"*

All systems: you should type in the Anaconda Prompt the following command

> cd /Users/PC-name/Desktop/ContractionWavePy-master/

Press *“Enter”* to access this folder using the Anaconda Prompt window.

**Important:** type: *"cd "* before the extracted program folder full path for accessing this
folder on *“Terminal”*.

**Hint:**
If you are unsure of the full path for the extraction you can use:

**Windows:** Right mouse click on desired file/folder > Select and click Properties
Full path should be under the “Location” tab.

**MacOS:** the "Finder" application to access the containing folder of the extraction target
directory and follow the instructions on:

https://www.josharcher.uk/code/find-path-to-folder-on-mac/

Instructions include basically pressing the *"Command ⌘"* or *"cmd ⌘"* and the "i"
keyboard keys at the same time AFTER selecting the extraction target directory
containing the program. A new window will open and the full path for the directory will
be shown in the "General > Where" tab.

Ubuntu: Right mouse click on desired file/folder > Select and click Properties
Full path should be under the “Parent folder” tab.

**FAQ**

**Q:** I've been getting: "python: can't open file "ContractionWave.py':[Errno 2]: No such
file or directory"

**A:** Check that you are in the correct directory for running the program. Remember to
type:
*"cd "*
before the extracted program folder full path for accessing this folder on Anaconda
Prompt.

**Step 5: Install all dependencies for ContractionWave using Anaconda**

**Important:** make sure you have a stable internet connection.
ContractionWave dependecies be easily install by typing the following command on the
previously opened Anaconda Prompt window:

**Windows:**
>conda env create -f ContractionWavePy-windows.yml
Press “Enter” to start the installation.

**Mac-OS:**
>conda env create -f ContractionWavePy-mac.yml
Press “Enter” to start the installation.

**Ubuntu:**
>conda env create -f ContractionWavePy-ubuntu.yml
Press “Enter” to start the installation.

Wait until all installations are concluded (it might take some time – 10 to 40 minutes)

#### **III- Executing ContractionWave (Should executed every time to open CW)**

**Step 6: Executing ContractionWave**

1. Open the Anaconda Prompt as instructed in Step 3.

2. Access the Extracted folder as instructed in section Step 4.

3. Activate the Anaconda environment by typing on the previously opened Anaconda

Prompt window:
> conda activate ContractionWavePy

Press “Enter” to start the environment.

**Step 7: Run the ContractionWave program by typing on the previously opened Anaconda Prompt window**

> python ContractionWave.py
Press “Enter” to start the program.

**FAQ**

**Q:** ContractionWave is not opening, what can be done?

**A:** If for any reasons ContractionWave does not open, try to remove the environment in the Anaconda Prompt

> conda env remove -n ContractionWavePy
Press “Enter” to remove the environment.

Or try to update Anaconda in the Anaconda Prompt
> conda update -n base -c defaults conda
Press “Enter” to update Anaconda.

and start again the steps from: II- ContractionWave environment installation with Anaconda Prompt (Should only be executed once)
