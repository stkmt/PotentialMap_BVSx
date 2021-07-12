# PotentialMap_BVSx
This program is going to create a potential map of mobile ions. This map (.rho format file) is used for visualizing diffusion pathways on the [VESTA](http://jp-minerals.org/vesta/en/) software.

![image](https://user-images.githubusercontent.com/80811293/125252205-0dd78f00-e333-11eb-8948-4f84d0c87178.png)


## Dependency
To run potentialmap.py
Python 3.7.7  <-- already checked
- module  
_shutil_  
_sys_  
_math_  
_numpy_  
_time_

- files (all files are from [_nap_](https://github.com/ryokbys/nap) made by [ryo kobayashi](https://github.com/ryokbys))  
_pmdini_  
_in.params.Coulomb_  
_in.params.Morse_  
_in.params.angular_


## Usage
```
$ python percolation.py Arg1 Arg2 Arg3 (Arg4) (Arg5) (Arg6) (Arg7) (Arg8) (Arg9) (Arg10)    (option)
```

Arg1    : pmd file (eg. pmdini)  
Arg2    : mobile ion (eg. Li)  
Arg3    : nominal charge of Arg2 ion (eg. 1.0)  
Arg4    : cutoff interatom distance for BVS-FF calc [ang] (def: 6.0)  
Arg5    : mesh resolution [mesh/ang] (def: 5.0)  
Arg6    : 3-body ions for angular potential (eg. Li-O-O,Li-Cl-Cl (it is okay to select more than 2))  (def: None)  
Arg7    : if remove mobile ion, True or False (def: True)  
Arg8    : if use sigmoid function, True or False (if Arg6 equals False, you should use this.  def: False)  
Arg9    : sigmoid width [ang] (def: 5.0)  
Arg10   : sigmoid shift [ang] (def: 3.0)

- output  
_BVSxmap.rho_  <--  the file to visualize diffusion pathways on VESTA  
_BVSx_xyzp.dat_  <--  the file to caluculate potential curve and migration barrier by percolation theory


## License
This software is released under the MIT License, see LICENSE.

## Authors
- Shuta Takimoto
- Creative Engineering Program, Department of Engineering
