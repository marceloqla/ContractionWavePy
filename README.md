# ContractionWavePy

## ABOUT

CONTRACTIONWAVE (CW) is a free software developed in Python Programming Language that allows the user to visualize, quantify, and analyze cell contractility parameters in a simple and intuitive format. The software enables the user to acquire membrane kinetics data of cell contractility during contraction-relaxation cycles through image capture and a dense optical flow algorithm. Both method and software were developed using multidisciplinary knowledge, which resulted in a robust data extraction protocol.

For more information access: https://sites.icb.ufmg.br/cardiovascularrc/contractionwave/

## GETTING STARTED
Contractionwave can be installed from an executable file (see below section – for
Windows or Ubuntu) or from Anaconda environment installation (see second section below –
for Windows, Ubuntu, or Mac-OS).

These installations steps can also be found on the program's manual on: https://sites.icb.ufmg.br/cardiovascularrc/contractionwave/

### Installation – Executable file

### Installation – Executable file for Windows or Ubuntu

**Step 1:** Sign in to Mendeley Data website: https://data.mendeley.com
If you are a first time user, you will need to register to log in.
The link to the executable file in the Mendeley website is presented below:
https://data.mendeley.com/datasets/hswwzgw6rp/draft?a=0b0c2dc1-5b68-4655-b4c1-
6cb792675b08

In Experiment data files click on CONTRACTIONWAVE - Executable file (Windows or
Ubuntu) folder and choose the Operating System to download.

**Step 2:** Unpack the zip file

**Step 3:** The extracted directory contains a library folder and one executable file (.exe
for Windows)

**Step 4:** Double-click on the CW icon to start the executable file

Wait until the ContractionWave window opens (it might take some time). 

**Then, you can proceed to section 3.3- Quick Start Guide.**

### Installation – Anaconda environment for Mac-OS
#### I- Anaconda environment installation (should only be executed
once)

The latest stable version **for Windows and Ubuntu (64-bit) or Mac-OS**

**Step 1: Download Contractionwave code**

**a)** Access the page https://github.com/marceloqla/ContractionWavePy

**b)** Click on the "Code" green button on the right hand side of the screen

**c)** Click on the "Download ZIP" button.

**d)** Unpack the file and complete the extraction process to a desired target directory

**Step 2: Installation using Anaconda**

Follow the instructions for installing Anaconda:

https://docs.anaconda.com/anaconda/install/mac-os/

After the installation is complete, close the Anaconda dialog box and then go to step II.
(no need to register)

#### **II-Executing ContractionWave using Anaconda in “Terminal”*

**Step 3: Open Anaconda Prompt**

**Mac-OS:** Cmd+Space to open Spotlight Search and type “Navigator” to open the
program.

**Step 4: Access the directory in which the ZIP file was extracted**

To access the **ContractionWavePy-master** directory in **MacOS:** follow the instructions below:

Use the "Finder" application to access the containing folder of the extraction target
directory and follow the instructions on:

https://www.josharcher.uk/code/find-path-to-folder-on-mac/

Instructions include basically pressing the *"Command ⌘"* or *"cmd ⌘"* and the "i"
keyboard keys at the same time AFTER selecting the extraction target directory
containing the program. A new window will open and the full path for the directory will
be shown in the "General > Where" tab.

**For example**, if the full path for the directory is:

*"/Users/PC-name/Desktop/ContractionWavePy/"*

you should type in the Anaconda Prompt the following command (**Example**):

> cd /Users/PC-name/Desktop/ContractionWavePy-master/

And then press *“Enter”* to access this folder using the Anaconda Prompt window.

**Important:** type: *"cd "* before the extracted program folder full path for accessing this
folder on the *“Anaconda Prompt”*.

**Step 5: Install all dependencies for ContractionWave using Anaconda**

**Important: should only be executed once.** Make sure you have a stable internet connection.

ContractionWave dependecies be easily installed by typing the following command:

>conda env create -f ContractionWavePy-mac.yml
Press “Enter” to start the installation.

Wait until all installations are concluded (it might take some time – 5 to 30 minutes)

**Step 6: Step 6: Activating Anaconda to execute ContractionWave**

**Important:** This step should be executed after typing the command on step 5 if you are
doing it for the first time, or after the command on step 4, if you have already followed
step 5 previously.

Activate the Anaconda environment by typing the the following command:

> conda activate ContractionWavePy
And then press “Enter” to start the environment activation.

**Important:** You should be able to see **(base)** prefix on Terminal change to **(ContractionWavePy)**

**Step 7: Run the ContractionWave software**

ContractionWave software can be opened by typing the command:

> python ContractionWave.py
And then Press “Enter” to start the program. Wait until the ContractionWave window opens (it might take some time: 1 to 5 minutes)

**FAQ**

**Q:** I've been getting: "python: can't open file "ContractionWave.py':[Errno 2]: No such
file or directory"

**A:** Check that you are in the correct directory for running the program. Remember to
type:"cd " with a space before the extracted program folder full path for accessing this
folder on “Terminal”.

**Q:** ContractionWave is not opening, what can be done?

**A:** If for any reasons ContractionWave does not open, try to remove the environment in the Anaconda Prompt

Execute the instructions in **Step 3** and **Step 4** and then:

> conda env remove -n ContractionWavePy
Press “Enter” to remove the environment.

Or try to update Anaconda in the Anaconda Prompt
> conda update -n base -c defaults conda
Press “Enter” to update Anaconda.

and start again the steps from: II- Executing ContractionWave using Anaconda in “Terminal”
