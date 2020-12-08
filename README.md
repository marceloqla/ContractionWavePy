# ContractionWavePy
## GETTING STARTED
### Installation - Executable file 
1. The latest stable version compiled executables for Windows and Ubuntu (64-bit) can be downloaded from: [https://sites.icb.ufmg.br/cardiovascularrc/contractionwave](https://sites.icb.ufmg.br/cardiovascularrc/contractionwave) (Choose the Operating System and click on Download)
2. Unpack the File
3. The extracted directory contains a library folder and one executable file (.exe for Windows)
4. Double-click the CW icon to start the executable file
### Installation - Using Source
 
1. **Download the program:**  
a) Access the page https://github.com/marceloqla/ContractionWavePy  
b) Click on the "Code" green button on the right hand side of the screen  
c) Click on the "Download ZIP" button.  
d) Double click the downloaded ZIP file and complete the extraction process to a desired target directory  


2. **Installation using Anaconda**  
Follow the instructions for installing Anaconda:  

**Windows:**  
 [https://docs.anaconda.com/anaconda/install/windows/)](https://docs.anaconda.com/anaconda/install/windows/)  
**Mac-OS:**  
[https://docs.anaconda.com/anaconda/install/mac-os/](https://docs.anaconda.com/anaconda/install/mac-os/)  
**Linux:**  
[https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)  


3. **Open the "Terminal" program.**  

**Windows:**  
Instructions can be found here:  
[https://www.computerhope.com/issues/chusedos.htm](https://www.computerhope.com/issues/chusedos.htm)  

Or, alternatively, here:  
[https://www.digitalcitizen.life/open-cmd/](https://www.digitalcitizen.life/open-cmd/)  

**Mac-OS:**
Instructions can be found here:  
[https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac](https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)  

Or, alternatively, here:  
[https://www.idownloadblog.com/2019/04/19/ways-open-terminal-mac/](https://www.idownloadblog.com/2019/04/19/ways-open-terminal-mac/)  

**Ubuntu:**  
Instructions can be found here:  
[https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal](https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal)  

Please refer to other materials for opening the Terminal in other Operating Systems. 

4. **Access the directory in which the ZIP file was extracted.**  
For example, if the full path for the directory is:  
"/Users/Neli/Desktop/ContractionWavePy/",  


**All systems:**    
you should type in the Terminal the following command:  
`cd /Users/Neli/Desktop/ContractionWavePy/`

and press enter to access this folder using the Terminal Window.

**Hint:**
If you are unsure of the full path for the extraction you can use:

**MacOS:** the "Finder" application to access the containing folder of the extraction target directory and follow the instructions on:

[https://www.josharcher.uk/code/find-path-to-folder-on-mac/](https://www.josharcher.uk/code/find-path-to-folder-on-mac/)

Instructions include basically pressing the "Command ⌘" or "cmd ⌘" and the "i" keyboard keys at the same time AFTER selecting the extraction target directory containing the program. A new window will open and the full path for the directory will be shown in the "General > Where" tab.

**Windows:** Right mouse click on desired file/folder > Select and click Properties
Full path should be under the “Location” tab.

**Ubuntu:**  Right mouse click on desired file/folder > Select and click Properties
Full path should be under the “Parent folder” tab.

5. **Install all dependencies for ContractionWave using Anaconda.**

a) This can be easily done by typing the following command on the previously opened "Terminal" window:  


`conda env create -f ContractionWavePy.yml`  
Press “Enter” to start the installation.


Wait until all installations are concluded.

b) Executing ContractionWave
1. Open the "Terminal" Program as instructed in section "1.b)" above.
2. Access the Extracted folder as instructed in section "1.c)" above.
3. Activate the Anaconda environment by typing on the previously opened "Terminal" Window:  
`conda activate ContractionWavePy`  
Press “Enter” to start the environment.


4. Run the ContractionWave program by typing on the previously opened "Terminal" Window:  
`python ContractionWave.py`  
Press “Enter” to start the program.

### FAQ

**Q:** I've been getting: *"python: can't open file "ContractionWave.py':[Errno 2]: No such file or directory"*  

**A:** Check that you are in the correct directory for running the program. Remember to type:

`cd `  

before the extracted program folder full path for accessing this folder on "Terminal".
