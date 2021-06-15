# ContractionWavePy

## ABOUT

CONTRACTIONWAVE (CW) is a free software developed in Python Programming Language that allows the user to visualize, quantify, and analyze cell contractility parameters in a simple and intuitive format. The software enables the user to acquire membrane kinetics data of cell contractility during contraction-relaxation cycles through image capture and a dense optical flow algorithm. Both method and software were developed using multidisciplinary knowledge, which resulted in a robust data extraction protocol.

For more information access: https://sites.icb.ufmg.br/cardiovascularrc/contractionwave/

## 1. GETTING STARTED
Contractionwave can be installed from an executable file (3.1 section – for
**Windows or Ubuntu**) or from Anaconda environment installation (3.2 section below –
for **Windows, Ubuntu, or Mac-OS**).

These installations steps can also be found on the program's manual on: https://sites.icb.ufmg.br/cardiovascularrc/contractionwave/

### 1.1 Installation – Executable file for Windows or Ubuntu

**Step 1:** The latest stable version compiled executable for **Windows and Ubuntu (64-bit)** can be downloaded from: 
The link to the executable file in the Mendeley website is presented below:
https://sites.icb.ufmg.br/cardiovascularrc/contractionwave (Choose the Operating System and click on Download)

In Experiment data files click on CONTRACTIONWAVE - Executable file (Windows or
Ubuntu) folder and choose the Operating System to download.

**Step 2:** Unpack the zip file

**Step 3:** The extracted directory contains a library folder and one executable file (*.exe*
for Windows)

**Step 4:** Double-click on the CW icon to start the executable file

Wait until the CONTRACTIONWAVE window opens (it might take some time). Then, you can proceed to section 3.3- Quick Start Guide.

### 1.2 Installation – Anaconda environment for Windows, Ubuntu or Mac-OS
#### I- Anaconda environment installation (should only be executed once)


**Step 1: Download CONTRACTIONWAVE code**

**a)** Access the page https://github.com/marceloqla/ContractionWavePy

**b)** Click on the "Code" green button on the right hand side of the screen

**c)** Click on the "Download ZIP" button.

**d)** Unpack the file and complete the extraction process to a desired target directory.

**Step 2: Installation using Anaconda**

Follow the instructions for installing Anaconda:

**Windows:** https://docs.anaconda.com/anaconda/install/windows/

**Mac-OS:** https://docs.anaconda.com/anaconda/install/mac-os/

**Linux:** https://docs.anaconda.com/anaconda/install/linux/

After the installation is complete, close the Anaconda dialog box and then go to step II.
(no need to register)

#### **II- CONTRACTIONWAVE environment installation with Anaconda Prompt (should only be executed once)**

**Step 3: Open Anaconda Prompt**

**Windows:** Click Start, search, or select Anaconda Prompt from the menu.
**Mac-OS:** Cmd+Space to open Spotlight Search and type “Navigator” to open the program.
**Ubuntu:** Open the Dash by clicking the upper left Ubuntu icon, then type “terminal”.

Please refer to other materials for opening the Terminal in other Operating Systems.

**Step 4: Access the directory in which the ZIP file was extracted**

To access the **ContractionWavePy-master directory** follow the instructions below:

**Windows:** Right mouse click on desired file/folder > Select and click Properties
The full path should be under the “Location” tab.

**MacOS:** the "Finder" application to access the containing folder of the extraction target directory and follow the instructions on:
https://www.josharcher.uk/code/find-path-to-folder-on-mac/
Instructions include pressing the "Command ⌘" or "cmd ⌘" and the "i" keyboard keys at the same time AFTER selecting the extraction target directory containing the program. A new window will open and the full path for the directory will be shown in the "General > Where" tab.

**Ubuntu:**  Right mouse click on desired file/folder > Select and click Properties
The full path should be under the “Parent folder” tab.

For example, if the full path for the directory is:

*"/Users/PC-name/Desktop/ContractionWavePy/"*

you should type in the Anaconda Prompt the following command (**Example**):

> cd /Users/PC-name/Desktop/ContractionWavePy-master/

And then press *“Enter”* to access this folder using the Anaconda Prompt window.

**Important:** type: *"cd " with space* before the extracted program folder full path for accessing this
folder on the *“Anaconda Prompt”*.


**FAQ**

**Q:** I've been getting: "python: can't open file "ContractionWave.py':[Errno 2]: No such
file or directory"

**A:** Check that you are in the correct directory for running the program. Remember to
type:
"cd "
before the extracted program folder full path for accessing this
folder on “Terminal”.

**Step 5: Install all dependencies for ContractionWave using Anaconda**

**Important:** Make sure you have a stable internet connection.

ContractionWave dependecies be easily installed by typing the following command on the previously opened Anaconda Prompt window:

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

#### III- Executing ContractionWave (It should be executed every time to open CW)

**Step 6: Executing ContractionWave**

I. Open the Anaconda Prompt as instructed in Step 3.

II. Access the Extracted folder by typing the full path for the ContractionWavePy-master directory as instructed in section Step 4.1

III. Activate the Anaconda environment by typing on the previously opened Anaconda Prompt window.

Type in the Anaconda Prompt the following command:

> conda activate ContractionWavePy
And then press “Enter” to start the environment and then go to step 7.

**Important:** You should be able to see the Terminal (base) prefix change to **(ContractionWavePy)**

**Step 7: Run the ContractionWave program by typing on the previously opened Anaconda Prompt window**

Type in the Anaconda Prompt the following command:

> python ContractionWave.py

And then Press “Enter” to start the program.
Wait until the ContractionWave window opens (it might take some time). Then, you can proceed to section 3.3- Quick Start Guide.

**FAQ**

**Q:** ContractionWave is not opening, what can be done?

**A:** If for any reasons ContractionWave does not open, try to remove the environment in the Anaconda Prompt

> conda env remove -n ContractionWavePy
Press “Enter” to remove the environment.

Or try to update Anaconda in the Anaconda Prompt
> conda update -n base -c defaults conda
Press “Enter” to update Anaconda.

and start again the steps from: II- ContractionWave environment installation with Anaconda Prompt (Should only be executed once) 
