# **** Imports ****
import numpy as np
import ase
from ase.io.trajectory import Trajectory
import Data_Set
# ************

class Model_Interpreter:
    
    def __init__(self, calculator, data_set: Data_Set,
                 name = 'not stated', evaluate_model = False):
        
        self.calculator = calculator # calculator used by ASE
        self.data_set = data_set # Data_Set object contaning the test data
        self.name = name # Model name (often good to use same name as model file name)
        
        if(evaluate_model == True):
            self.benchmark_model()
        
        return
        
    def benchmark_model(self):
        print('*** Benchmarking model:', self.name, '******')
        
        # *** Computing model energies and forces *******************
        # Initializeing energy and forces arrays;
        self.model_energies = np.array([])
        self.model_forces = np.array([])
        self.model_energies_avg = np.array([])
        
        i = 0
        for atoms in self.data_set.data:
        
            atoms.calc = self.calculator # Assigns calculator to ASE atoms object
            
            # Computing energy:
            new_energy = atoms.get_potential_energy()
            self.model_energies = np.append(self.model_energies, new_energy)
            self.model_energies_avg = np.append(self.model_energies_avg, new_energy/len(atoms))
        
            # Computing forces:
            if(len(self.model_forces) > 0):
                self.model_forces = np.vstack((self.model_forces,
                                               atoms.get_forces()))
            else:
                # For correct initilization of dimentions of forces array
                self.model_forces = atoms.get_forces()
                
            # for printing of progress:
            i = i + 1
            if(i%10 == 0):
                print('atoms objects evaluated:', i, '/', len(self.data_set.data))
        
        # *** Now computing errors and misc ************************
        self.N = len(self.model_forces) # Total number of atoms in data_set
        
        # Computing all energy and force diffrences:
        self.energies_diff = np.abs(self.data_set.energies - self.model_energies) # Energy diffrence
        self.forces_diff = self.data_set.forces - self.model_forces # All forces diffrence
        self.forces_diff_mag = np.linalg.norm(self.forces_diff, axis = 1) # All force magnitude diffrences
        
        self.forces_relative_err = self.forces_diff_mag/np.linalg.norm(self.data_set.forces, axis = 1)
        self.forces_RMSE = np.sqrt(np.sum(self.forces_diff_mag**2)/len(self.forces_diff_mag)) # Force RMSE
        
        # Finding all maximum forces:
        self.max_forces_diff_mag = np.array([])
        index = 0
        for atoms in self.data_set.data:
            n = len(atoms)
            new_max = self.forces_diff_mag[index: index + n]
            new_max = new_max.max()
            self.max_forces_diff_mag = np.append(self.max_forces_diff_mag, new_max)
            index = index + n
            
        self.max_forces_RMSE = np.sqrt(np.sum(self.max_forces_diff_mag**2)/len(self.max_forces_diff_mag))

        print('Model', self.name, ': !BENCHMARKED!')
        print('*********************************** \n')
                
        return
   