# MITgcm-for-icy-ocean-worlds
This code modify MITgcm to simulate icy ocean world.
The MITgcm version for which this code is written can be downloaded from https://github.com/jm-c/MITgcm/tree/deep_factor_in_Smag3D. 
There are newer versions, but they may not be compatible with the code patch here.

For people who are familiar with MITgcm, you'll only need code_now where modified source codes are stored and input_now where namelist data files are stored. In the input_now folder, there is a gendata.m file. It takes some physical parameters as input and writes them to the namelist data files and generates needed input files. Please take a close look before using and make changes as needed.

Steps for people who are not familiar with MITgcm:
1. Download MITgcm source code from https://github.com/jm-c/MITgcm/tree/deep_factor_in_Smag3D

2. Create a folder called "my_exp" under the MITgcm folder, and copy everything in this folder there.

3. Download gsw package from https://www.teos-10.org/software.htm#1

4. Search for XX in the my_exp and fill in correct info. For compiler and MPI choices, you may want to ask your IT people.

5. To build a test case, put in the following command:
    ./buildmitgcm.sh [casename] [resolution] [ncpu] compilebuild
   If you want to remove an existing case
    ./buildmitgcm.sh [casename] [resolution] [ncpu] clean
   I recommend reading buildmitgcm.sh first to know what happens there.
   This code will configure and compile MITgcm for you. After it is done, a folder named [casename] will be created under my_exp.

    Folder structure:

        i) build: it contains the objective files source code files and executable, mitgcmuv.
    
        ii) code: it contains the modified source code adapted from the default MITgcm code. If you make changes to anything in the code folder, you’ll need to go back to the build folder and rebuild the model. To do so, go to the build folder, load proper modules, type "make" and hit enter.
    
        iii) input: it contains all the namelist files which specifies parameters such as ocean salinity, bathymetry, domain etc. A matlab file in this folder called gendata.m is used to set up the namelist files (whose names starting w/ “data.xxx”) and produce the input data files needed for the configuration (e.g., bathymetry profile). Please take a close look before using! Changes will be needed if you want to alter the model configuration.
    
        iv) data_[casename]: this is a link pointing to a folder that stores your data and run the experiment.
    
        v) jupyter_template.ipynb: this is a sample code to visualize the model output. In order for this to work, you need to use my analysis library (make it visible to python by setting the PYTHONPATH env variable). Caution: this package assumes that the model output is written in a very specific way as specified in the data.diagnostics. For MIT svante user, you need to go to: https://svante-ood.mit.edu/pun/sys/dashboard/batch_connect/sessions to start a jupyter notebook session.
    
        vi) newrundir: this script helps you create a new experiment which uses the same source code and thus the same executable (mitgcmuv under the build folder) but different namelist configurations (done in the input folder). To do so, just type ./newrundir [expname] create [reference expname]. 

6. Run gendata.m by typing in "matlab<gendata.m"

7. to run the model, you need to get into the run directory, data_[casename]_[expname], and sbatch run.sub (may vary in other systems, check with IT people). It will run the model until totaliteration is reached.

Good luck! 
