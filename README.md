


https://github.com/user-attachments/assets/8f76f637-60b7-4662-b88c-0e58c94f19c6
Log Default


# Stellar_evolution_animation
Creates a stellar evolution animation for binary systems simulated using COMPAS.
Animation is reflective of the radius, separation, percieved color etc of the stars. 

ALL code related to temperature-color relationship is directly from the TULIPS code
https://bitbucket.org/elaplace/tulips/src/master/
https://ui.adsabs.harvard.edu/abs/2022A%26C....3800516L/abstract


Code is run using the run.sh file. 
Before doing so, update the data file input path in the preprocess.py file, it is currently set to a default example file provided with the code. 
in terminal run 

chmod +x "_______/Final Animation/run.sh"

where ______ should be replaced with your path up until that point



then you can run
./run.sh log tulips

this is just an example, you have the options of log/linear or tulips/default
and the animation will be created using pygame



things required for this... still needs to be updated  but so far

pip install numpy pygame pillow



https://github.com/user-attachments/assets/42940662-179b-461f-807a-38408b33354d
Linear Tulips


