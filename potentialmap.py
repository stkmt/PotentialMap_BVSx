#!/usr/bin/env python

import shutil
import sys
import math
import numpy as np
import time


#Date 2021/04
#Author Shuta Takimoto
#Add 3-body system 2021/05  ** this func is super slow.. isnt recommended.
#I wonder calculate anglepotential() should be recalled in bvspotential()

argvs = sys.argv

if len(argvs) < 2:
    print("\n"
          "         PROGRAM DESCRIPTION\n"
          "\n"
          "         This program is going to create a potential map.\n"
          "         This map is used for visualizing diffusion pathways on the VESTA software.\n"
          "         In this program, the BVSx technique as a new version potential will be used.\n"
          "         Following files are needed here,\n"
          "               - pmd file (eg. pmdini)\n"
          "               - in.params.Coulomb\n"
          "               - in.params.Morse\n"
          "               - in.params.angular\n"
          "\n"
          "         Here, there is two important options.\n"
          "         First one is the angular potential penalty on the mobile ions.\n"
          "         You can set the angular potential penalty among 'mobile ion - anion - anion'.\n"
          "         It's too slow to calculate actually... That's why this option is not recommended.\n"
          "\n"
          "         Second one is if you remove mobile ion or not.\n"
          "         If you want to remove mobile ions to calculate BVSx potential, Arg7 should be 'True' (default set is True).\n"
          "         You have to use sigmoid function (Arg8 - Arg10) to smear the interaction between original mobile ions\n"
          "         and new ions.\n"
          "\n"
          "         Two files, 'BVSxmap.rho' and 'BVSx_xyzp.dat' will be outputted.\n"
          "         'BVSxmap.rho' is useful to visualize pathways, 'BVSx_xyzp.dat' is useful to calculate the migration barrier.\n"
          "\n"
          "         Hope your success. Bye:)\n"
          )

if len(argvs) < 2 or len(argvs) > 10:
    print("\nUsage   : Arg1 Arg2 Arg3 (Arg4) (Arg5) (Arg6) (Arg7) (Arg8) (Arg9) (Arg10)")
    print("  Arg1    : pmd file (eg. pmdini)")
    print("  Arg2    : mobile ion (eg. Li)")
    print("  Arg3    : nominal charge of Arg2 ion (eg. 1.0)")
    print("  Arg4    : cutoff interatom distance for BVS-FF calc [ang] (def: 6.0)")
    print("  Arg5    : mesh resolution [mesh/ang] (def: 5.0)")
    print("  Arg6    : 3-body ions for angular potential (eg. Li-O-O,Li-Cl-Cl (it is okay to select more than 2))  (def: None)")
    print("  Arg7    : if remove mobile ion, True or False (def: True)")
    print("  Arg8    : if use sigmoid function, True or False (if Arg6 equals False, you should use this.  def: False)")
    print("  Arg9    : sigmoid width [ang] (def: 5.0)")
    print("  Arg10   : sigmoid shift [ang] (def: 3.0)")

    sys.exit()


class PotentialMap:
    def __init__(self, targetion, targetion_charge, max_r, mesh_reso):
        self.tion         = targetion
        self.tioncharge   = targetion_charge
        self.max_r        = max_r
        self.mesh_reso    = mesh_reso
        self.max_e        = 0
        self.min_e        = 99999

    
    def bvspotential(self, pmddata, atomcoord, param_morse, param_coulomb, sigmoid_switch=False, sigmoid_width=5.0, sigmoid_shift=3.0):
        self.mesh_x = int(pmddata["a_lat"] * mesh_reso)
        self.mesh_y = int(pmddata["b_lat"] * mesh_reso)
        self.mesh_z = int(pmddata["c_lat"] * mesh_reso)

        self.Morse_sum    = np.zeros((self.mesh_x, self.mesh_y, self.mesh_z))
        self.Coulomb_sum  = np.zeros((self.mesh_x, self.mesh_y, self.mesh_z))
        self.Total_sum    = np.zeros((self.mesh_x, self.mesh_y, self.mesh_z))

        print("a-axis: {:.5f} ang.".format(pmddata["a_lat"]))
        print("b-axis: {:.5f} ang.".format(pmddata["b_lat"]))
        print("c-axis: {:.5f} ang.".format(pmddata["c_lat"]))
        print("x-mesh: {0}, y-mesh: {1}, z-mesh: {2}".format(self.mesh_x, self.mesh_y, self.mesh_z))
        print("log----")

        nnx = int(max_r // pmddata["a_lat"])
        nny = int(max_r // pmddata["b_lat"])
        nnz = int(max_r // pmddata["c_lat"])

        self.lattice_vector = np.array([pmddata["a_vector"], pmddata["b_vector"], pmddata["c_vector"]])
        distance = {}

        for x in range(self.mesh_x):
            print("x  =  {0}/{1}".format(x+1, self.mesh_x))

            for y in range(self.mesh_y):
                for z in range(self.mesh_z):
                    voxel_rel = np.array([[x/self.mesh_x], [y/self.mesh_y], [z/self.mesh_z]])

                    for NX in range(-nnx, nnx+1):
                        for NY in range(-nny, nny+1):
                            for NZ in range(-nnz, nnz+1):
                                for key, value in atomcoord.items():
                                    n = 1

                                    for v in value:
                                        coord_main_rel = np.array([[float(v[1])], [float(v[2])], [float(v[3])]])

                                        diff_rel = coord_main_rel - voxel_rel

                                        if diff_rel[0, 0] <= -0.5:
                                            diff_rel[0, 0] += 1
                                        elif diff_rel[0, 0] > 0.5:
                                            diff_rel[0, 0] -= 1
                                        if diff_rel[1, 0] <= -0.5:
                                            diff_rel[1, 0] += 1
                                        elif diff_rel[1, 0] > 0.5:
                                            diff_rel[1, 0] -= 1
                                        if diff_rel[2, 0] <= -0.5:
                                            diff_rel[2, 0] += 1
                                        elif diff_rel[2, 0] > 0.5:
                                            diff_rel[2, 0] -= 1

                                        diff_rel[0, 0] += NX
                                        diff_rel[1, 0] += NY
                                        diff_rel[2, 0] += NZ

                                        diff_abs = np.dot(self.lattice_vector, diff_rel).reshape((1,3))
                                        distance[key+str(n)] = math.sqrt(diff_abs[0, 0]**2 + diff_abs[0, 1]**2 + diff_abs[0, 2]**2)

                                        if distance[key+str(n)] < max_r:
                                            if param_coulomb[key][0] < 0 and self.tioncharge > 0:
                                                self.Morse_sum[x, y, z] = self.Morse_pot(param_morse[self.tion+","+key][0], param_morse[self.tion+","+key][1], param_morse[self.tion+","+key][2], distance[key+str(n)]) + self.Morse_sum[x, y ,z]

                                            elif param_coulomb[key][0] > 0 and self.tioncharge < 0:
                                                self.Morse_sum[x, y, z] = self.Morse_pot(param_morse[self.tion+","+key][0], param_morse[self.tion+","+key][1], param_morse[self.tion+","+key][2], distance[key+str(n)]) + self.Morse_sum[x, y ,z]

                                            elif param_coulomb[key][0] >= 0 and self.tioncharge > 0:
                                                if distance[key+str(n)] < 0.1:
                                                    self.Coulomb_sum[x, y, z] = 999 + self.Coulomb_sum[x, y ,z]

                                                else:
                                                    if key == self.tion:
                                                        if sigmoid_switch:
                                                            sigmoid = self.sigmoid_func(sigmoid_width, sigmoid_shift, distance[key+str(n)])
                                                        
                                                        else:
                                                            sigmoid = 1.0

                                                    else:
                                                        sigmoid = 1.0

                                                    self.Coulomb_sum[x, y ,z] = self.Coulomb_pot(14.4, param_coulomb[key][0], distance[key+str(n)], param_coulomb[self.tion][1], param_coulomb[key][1], param_coulomb["fbvs"], sigmoid) + self.Coulomb_sum[x, y, z]

                                            elif param_coulomb[key][0] <= 0 and self.tioncharge < 0:
                                                if distance[key+str(n)] < 0.1:
                                                    self.Coulomb_sum[x, y, z] = 999 + self.Coulomb_sum[x, y, z]

                                                else:
                                                    if key == self.tion:
                                                        if sigmoid_switch:
                                                            sigmoid = self.sigmoid_func(sigmoid_width, sigmoid_shift, distance[key+str(n)])

                                                        else:
                                                            sigmoid = 1.0

                                                    else:
                                                        sigmoid = 1.0

                                                    self.Coulomb_sum[x, y, z] = self.Coulomb_pot(14.4, param_coulomb[key][0], distance[key+str(n)], param_coulomb[targetion][1], param_coulomb[key][1], param_coulomb["fbvs"], sigmoid) + self.Coulomb_sum[x, y, z]

                                    n += 1

                                    self.Total_sum[x, y, z] = self.Morse_sum[x, y, z] + self.Coulomb_sum[x, y, z]
                                    if self.Total_sum[x, y, z] >= self.max_e:
                                        self.Total_sum[x, y, z] = self.max_e
                                    if self.Total_sum[x, y, z] < self.min_e:
                                        self.min_e = self.Total_sum[x, y, z]


    def anglepotential(self, angle_trio, param_angular, atomcoord):
        self.angular_sum = np.zeros((self.mesh_x, self.mesh_y, self.mesh_z))

        for atom in angle_trio:
            atom_main = atom.split("-")[0]
            atom_subA = atom.split("-")[1]
            atom_subB = atom.split("-")[2]

            for x in range(self.mesh_x):
                print("x_angle  =  {0}/{1}".format(x+1, self.mesh_x))

                for y in range(self.mesh_y):
                    for z in range(self.mesh_z):
                        voxel_rel = np.array([[x/self.mesh_x], [y/self.mesh_y], [z/self.mesh_z]])

                        for coordA in atomcoord[atom_subA]:
                            coord_subA_rel = np.array([float(i) for i in coordA[1:4]]).reshape((3,1))

                            diff_A_rel = coord_subA_rel - voxel_rel

                            if diff_A_rel[0] <= -0.5:
                                diff_A_rel[0, 0] += 1
                            elif diff_A_rel[0, 0] > 0.5:
                                diff_A_rel[0, 0] -= 1
                            if diff_A_rel[1, 0] <= -0.5:
                                diff_A_rel[1, 0] += 1
                            elif diff_A_rel[1, 0] > 0.5:
                                diff_A_rel[1, 0] -= 1
                            if diff_A_rel[2, 0] <= -0.5:
                                diff_A_rel[2, 0] += 1
                            elif diff_A_rel[2, 0] > 0.5:
                                diff_A_rel[2, 0] -= 1

                            diff_A_abs = np.dot(self.lattice_vector, diff_A_rel).reshape((1,3))
                            ra = math.sqrt(diff_A_abs[0, 0]**2 + diff_A_abs[0, 1]**2 + diff_A_abs[0, 2]**2)

                            for coordB in atomcoord[atom_subB]:
                                coord_subB_rel = np.array([float(i) for i in coordB[1:4]]).reshape((3,1))
                                diff_B_rel = coord_subB_rel - voxel_rel

                                m = 0
                                for (subA, subB) in zip(coord_subA_rel, coord_subA_rel):
                                    if subA != subB:
                                        m += 1

                                if m != 0:
                                    if diff_B_rel[0] <= -0.5:
                                        diff_B_rel[0, 0] += 1
                                    elif diff_B_rel[0, 0] > 0.5:
                                        diff_B_rel[0, 0] -= 1
                                    if diff_B_rel[1, 0] <= -0.5:
                                        diff_B_rel[1, 0] += 1
                                    elif diff_B_rel[1, 0] > 0.5:
                                        diff_B_rel[1, 0] -= 1
                                    if diff_B_rel[2, 0] <= -0.5:
                                        diff_B_rel[2, 0] += 1
                                    elif diff_B_rel[2, 0] > 0.5:
                                        diff_B_rel[2, 0] -= 1

                                diff_B_abs = np.dot(self.lattice_vector, diff_B_rel).reshape((1,3))
                                rb = math.sqrt(diff_B_abs[0, 0]**2 + diff_B_abs[0, 1]**2 + diff_B_abs[0, 2]**2)

                                if ra <= param_angular[atom][0] and rb <= param_angular[atom][0]:
                                    theta = self.get_angular_inp(voxel_rel, coord_subA_rel, coord_subB_rel)

                                    self.angular_sum[x, y, z] = self.angular_pot(self.param_angular[atom][0], self.param_angular[atom][1], self.param_angular[atom][2], self.param_angular[atom][3], ra, rb, theta) + self.angular_sum[x, y, z]

                        self.Total_sum[x, y, z] = self.Morse_sum[x, y, z] + self.Coulomb_sum[x, y, z] + self.angular_sum[x, y, z]
                        if self.Total_sum[x, y, z] >= self.max_e:
                            self.Total_sum[x, y, z] = self.max_e
                        if self.Total_sum[x, y, z] < self.min_e:
                            self.min_e = self.Total_sum[x, y, z]


    def remove_ion_from_pmdfile(self, inputfile, outputfile):
        shutil.copy(inputfile, outputfile)

        with open(inputfile) as f:
            lines_default = f.readlines()

            ionlist = [s.split() for s in lines_default if "specorder" in s]
            ionnum = ionlist[0].index(self.tion) - 1
            lines_removed = [s for s in lines_default if not (s.split()[0].startswith(str(ionnum)) and len(s.split()) == 7)]
            removed_ion_list = [s for s in lines_default if (s.split()[0].startswith(str(ionnum)) and len(s.split()) == 7)]
            lines_removed = [s.replace(self.tion+" ", "") for s in lines_removed]

            lines_removed2 = []

            for s in lines_removed:
                if len(s.split()) == 7 and int(s.split(".")[0]) > ionnum:
                    lines_removed2.append(s.replace(s.split(".")[0]+".10000000", "   " + str(int(s.split(".")[0])-1)+".10000000"))

                else:
                    lines_removed2.append(s)

            lines_removed2[10] = lines_removed2[10].replace(lines_removed[10].replace("\n", "").split()[0], str(int(lines_removed[10].replace("\n", "").split()[0]) - len(removed_ion_list)))

        with open(outputfile, "w") as f:
            for i in lines_removed2:
                f.write(i)


    def Morse_pot(self, D0, alpha, r_min, r):
        return D0 * (math.exp(-2 * alpha * (r - r_min)) -2 * math.exp(-alpha * (r - r_min)))


    def Coulomb_pot(self, k, qB, r, rA, rB, fbvs, sigmoid_function):
        rho = (rA + rB) * fbvs
        return k * sigmoid_function * self.tioncharge * qB / r * math.erfc(r / rho)


    def angular_pot(self, rc, alpha, beta, gamma, ra, rb, theta):
        return alpha * math.exp(beta / (ra - rc) + beta / (rb - rc)) * (theta - gamma)**2


    def sigmoid_func(self, width, shift, distance):
        return 0.5 * (1 - math.exp((-width) * (distance - shift))) / (1 + math.exp((-width) * (distance - shift))) + 0.5

    
    def get_angular_inp(self, main, A, B):
        diff_ra = A - main
        diff_rb = B - main
        diff_ra_abs = math.sqrt(diff_ra[0, 0]**2 + diff_ra[1, 0]**2 + diff_ra[2, 0]**2)
        diff_rb_abs = math.sqrt(diff_rb[0, 0]**2 + diff_rb[1, 0]**2 + diff_rb[2, 0]**2)

        cos_theta = np.inner(diff_ra, diff_rb) / (diff_ra_abs * diff_rb_abs) 
        theta = np.arccos(cos_theta[0, 0])

        return theta

    
    def read_inparamsMorse(self, inputfile):
        with open(inputfile) as f:
            lines = f.readlines()

            parameters_dict = {}
            for l in lines:
                if not "#" in l:
                    parameters_dict[l.split()[0]+','+l.split()[1]] = [float(i.replace("\n", "")) for i in l.split()[2:]]

        mode = "normal"

        for key in parameters_dict.keys():
            if key.split(",")[1] == self.tion:
                mode = "invars"

        keys = []

        if mode == "invars":
            d = parameters_dict

            for key in d.keys():
                if key.split(",")[1] == self.tion:
                    keys.append(key.split(",")[1]+","+key.split(",")[0])

        for key in keys:
            parameters_dict[key] = d[key.split(",")[1]+","+key.split(",")[0]]

            del parameters_dict[key.split(",")[1]+","+key.split(",")[0]]

        return parameters_dict


    def read_inparamsCoulomb(self, inputfile):
        with open(inputfile) as f:
            lines = f.readlines()

            mode = None
            parameters_dict = {}
            interactions_list = []

            for l in lines:
                if "charges" in l:
                    mode = "parameters"

                elif "fbvs" in l:
                    fbvs = float(l.split()[1].replace("\n", ""))
                    parameters_dict["fbvs"] = fbvs

                elif "interactions" in l:
                    mode = "interactions"

                elif mode == "parameters" and len(l.split()) == 4:
                    parameters_dict[l.split()[0]] = [float(i.replace("\n", "")) for i in l.split()[1:]]

                elif mode == "interactions" and len(l.split()) == 2:
                    try:
                        interactions_list.append(i.replace("\n", "") for i in l.split())
                    except:
                        pass
                else:
                    pass

        return parameters_dict


    def read_inparamsangular(self, inputfile):
        with open(inputfile) as f:
            lines = f.readlines()

            parameters_dict = {}

            for l in lines:
                if not "#" in l:
                    parameters_dict[l.split()[1]+"-"+l.split()[2]+"-"+l.split()[3]] = [float(i.replace("\n", "")) for i in l.split()[4:]]

        return parameters_dict


    def read_datapmd(self, inputfile, outputfile):
        data_dict = {}
        species_dict = {}

        with open(outputfile) as f:
            lines = f.readlines()

            mode = None

            for l in lines:
                if "a vector  =" in l:
                    a_vector = [float(l.split()[4].replace(",", "")),float(l.split()[5].replace(",", "")),float(l.split()[6].replace("]", ""))]
                    data_dict["a_vector"] = a_vector

                elif "b vector  =" in l:
                    b_vector = [float(l.split()[4].replace(",", "")),float(l.split()[5].replace(",", "")),float(l.split()[6].replace("]", ""))]
                    data_dict["b_vector"] = b_vector

                elif "c vector  =" in l:
                    c_vector = [float(l.split()[4].replace(",", "")),float(l.split()[5].replace(",", "")),float(l.split()[6].replace("]", ""))]
                    data_dict["c_vector"] = c_vector

                elif "a =" in l and "A" in l:
                    a_lat = float(l.split()[2])
                    data_dict["a_lat"] = a_lat

                elif "b =" in l:
                    b_lat = float(l.split()[2])
                    data_dict["b_lat"] = b_lat

                elif "c =" in l:
                    c_lat = float(l.split()[2])
                    data_dict["c_lat"] = c_lat

                elif "alpha =" in l:
                    alpha = float(l.split()[2])
                    data_dict["alpha"] = alpha

                elif "beta  =" in l:
                    beta = float(l.split()[2])
                    data_dict["beta"] = beta

                elif "gamma =" in l:
                    gamma = float(l.split()[2])
                    data_dict["gamma"] = gamma

                elif "volume= " in l:
                    volume = float(l.split()[1])
                    data_dict["volume"] = volume

                elif "number of atoms   =" in l:
                    num_total_atom = int(l.split()[4])
                    data_dict["num_total_atom"] = num_total_atom

                elif "number of atoms per species:" in l:
                    mode = "num_atom"

                elif "density =" in l:
                    density = float(l.split()[2])
                    data_dict["density"] = density

                elif mode == "num_atom" and len(l.split()) <= 3:
                    species_dict[l.replace(":", "").split()[0]] = int(l.replace(":", "").split()[1])

        coordinate = {}

        with open(inputfile) as f:
            lines = f.readlines()

            if len(lines[10].split()) != 1 and len(lines[11].split()) != 7:
                sys.exit("ERROR: xyz coordinate should start at line 12.")

            head_lines = lines[:11]
            coord_lines = lines[11:]

            n = 1

            for key in species_dict.keys():
                coordinate[key] = [s for s in lines if s.split()[0].startswith(str(n)) and len(s.split()) == 7]

                n += 1

        return data_dict, coordinate


    def stock(self):
        pstock = []
        for x in range(self.mesh_x):
            for y in range(self.mesh_y):
                for z in range(self.mesh_z):
                    pstock.append(self.max_e-self.Total_sum[x, y, z])

        return pstock


    def write_rho(self, pmddata, pstock, outputfile="BVSxmap.rho"):
        with open(outputfile, "w") as f:
            f.write("cell\n")
            f.write("{0:.5f} {1:.5f} {2:.5f}\n".format(pmddata["a_lat"], pmddata["b_lat"], pmddata["c_lat"]))
            f.write("{0:.5f} {1:.5f} {2:.5f}\n".format(pmddata["alpha"], pmddata["beta"], pmddata["gamma"]))
            f.write("{0} {1} {2} {3:.5f} {4:.5f} {5:.5f}\n".format(self.mesh_x, self.mesh_y, self.mesh_z, pmddata["a_lat"]/0.52918, pmddata["b_lat"]/0.52918, pmddata["c_lat"]/0.52918))

            loop = len(pstock)//5
            loopre = len(pstock)%5
            for uu in range(loop):
                for uuu in range(5):
                    try:
                        f.write("{:.8e}  ".format(pstock[(uu+1)*5-5+uuu]))
                    except:
                        f.write("{:.8e}  ".format(0))
                f.write("\n")

            for uuu in range(loopre):
                try:
                    f.write("{:.8e}  ".format(pstock[loop*5+uuu]))
                except:
                    f.write("{:.8e}  ".format(0))


    def write_dat(self, pmddata, pstock, outputfile="BVSx_xyzp.dat"):
        with open(outputfile, "w") as f:
            f.write("cell ")
            f.write("{0:.5f} {1:.5f} {2:.5f} ".format(pmddata["a_lat"], pmddata["b_lat"], pmddata["c_lat"]))
            f.write("{0:.5f} {1:.5f} {2:.5f}\n".format(pmddata["alpha"], pmddata["beta"], pmddata["gamma"]))
            f.write("mesh {0:3d} {1:3d} {2:3d}\n".format(self.mesh_x, self.mesh_y, self.mesh_z))
            for x in range(self.mesh_x):
                for y in range(self.mesh_y):
                    for z in range(self.mesh_z):
                        f.write("{0:3d} {1:3d} {2:3d} {3:.5e}\n".format(x, y, z, self.Total_sum[x, y, z]-self.max_e))


class Analyze:
    def __init__(self, targetion):
        self.tion  = targetion


    def read_pmd(self, pmdfile):
        with open(pmdfile) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]

        if len(lines[10].split()) != 1 and len(lines[11].split()) != 7:
                sys.exit("ERROR: xyz coordinate should start at line 12.")

        n = 0
        ionnumlist = []
        coordinatelist = []

        for l in lines:
            if "specorder" in l:
                ionlist = l.split()[2:]

            elif not "!" in l:
                if n == 0:
                    self.ratio = float(l)

                elif n == 1 and len(l.split()) == 3:
                    self.a_vector  = [float(i) for i in l.split()]
                    self.a_vector_ = [v*self.ratio for v in self.a_vector]

                elif n == 2 and len(l.split()) == 3:
                    self.b_vector  = [float(i) for i in l.split()]
                    self.b_vector_ = [v*self.ratio for v in self.b_vector]

                elif n == 3 and len(l.split()) == 3:
                    self.c_vector  = [float(i) for i in l.split()]
                    self.c_vector_ = [v*self.ratio for v in self.c_vector]

                elif n == 4 and len(l.split()) == 3:
                    self.va_vector = [float(i) for i in l.split()]

                elif n == 5 and len(l.split()) == 3:
                    self.vb_vector = [float(i) for i in l.split()]

                elif n == 6 and len(l.split()) == 3:
                    self.vc_vector = [float(i) for i in l.split()]

                elif n == 7:
                    self.totalnumofion = int(l)

                elif n >= 8:
                    coordinatelist.append(l.split())
                    ionnumlist.append(int(l.split(".")[0]))

                n += 1

        self.a_lat = np.linalg.norm(self.a_vector_)
        self.b_lat = np.linalg.norm(self.b_vector_)
        self.c_lat = np.linalg.norm(self.c_vector_)
        self.vol = self.volume()
        self.alpha, self.beta, self.gamma = self.angle()
        

        ionid = [n+1 for n, i in enumerate(ionlist)]
        speciesdict = {}
        coordinatedict = {}

        for ion, iid in zip(ionlist, ionid):
            speciesdict[ion] = ionnumlist.count(iid)
            coordinatedict[ion] = [c for c in coordinatelist if c[0].startswith(str(iid))]

        self.speciesdict = speciesdict.copy()
        self.coordinatedict = coordinatedict.copy()


    def write_datapmd(self, outputfile="data.pmd"):
        with open(outputfile, "w") as f:
            f.write(" a vector  = [{0:10.3f}, {1:10.3f}, {2:10.3f}]\n".format(self.a_vector_[0], self.a_vector_[1], self.a_vector_[2]))
            f.write(" b vector  = [{0:10.3f}, {1:10.3f}, {2:10.3f}]\n".format(self.b_vector_[0], self.b_vector_[1], self.b_vector_[2]))
            f.write(" c vector  = [{0:10.3f}, {1:10.3f}, {2:10.3f}]\n".format(self.c_vector_[0], self.c_vector_[1], self.c_vector_[2]))
            f.write(" a = {0:10.3f} A\n".format(self.a_lat))
            f.write(" b = {0:10.3f} A\n".format(self.b_lat))
            f.write(" c = {0:10.3f} A\n".format(self.c_lat))
            f.write(" alpha = {0:7.2f} deg.\n".format(self.alpha))
            f.write(" beta  = {0:7.2f} deg.\n".format(self.beta))
            f.write(" gamma = {0:7.2f} deg.\n".format(self.gamma))
            f.write(" volume= {0:10.3f} A^3\n".format(self.vol))
            f.write(" number of atoms   = \n", self.totalnumofion)

            if self.speciesdict:
                f.wirte(" number of atoms per species:\n")
                for key, value in self.speciesdict.items():
                    f.write("   {0:<2s}: {1:>4d}\n".format(key, value))


    def volume(self):
        return self.ratio**3 * np.abs(np.dot(self.a_vector, np.cross(self.b_vector, self.c_vector)))


    def angle(self):
        alpha = np.arccos(np.dot(self.b_vector_, self.c_vector_) / self.b_lat / self.c_lat) / np.pi * 180.0
        beta  = np.arccos(np.dot(self.a_vector_, self.b_vector_) / self.a_lat / self.b_lat) / np.pi * 180.0
        gamma = np.arccos(np.dot(self.a_vector_, self.c_vector_) / self.a_lat / self.c_lat) / np.pi * 180.0

        return alpha, beta, gamma


if __name__ == "__main__":
    start_time  =  time.time()

    pmdfile = argvs[1]
    targetion = argvs[2]
    targetion_charge = float(argvs[3])

    max_r = 6.0
    try:
        max_r = float(argvs[4])

    except:
        pass

    mesh_reso = 5.0
    try:
        mesh_reso = float(argvs[5])

    except:
        pass

    angle_trio = None
    try:
        angle_trio = argvs[6].split(',')

    except:
        pass

    remove_switch = True
    try:
        if argvs[7] == "None":
            remove_switch = None

    except:
        pass

    sigmoid_switch = False
    try:
        if argvs[8] == "True":
            sigmoid_switch = True

    except:
        pass

    sigmoid_width = 5.0
    try:
        sigmoid_width = float(argvs[9])

    except:
        pass

    sigmoid_shift = 3.0
    try:
        sigmoid_shift = float(argvs[10])

    except:
        pass

    pm = PotentialMap(targetion, targetion_charge, max_r, mesh_reso)
    analyze = Analyze(targetion)

    if remove_switch:
        pm.remove_ion_from_pmdfile(pmdfile, pmdfile+"_removed")
        analyze.read_pmd(pmdfile+"_removed")

    else:
        analyze.read_pmd(pmdfile)

    data_dict = {
                "a_vector": analyze.a_vector_,
                "b_vector": analyze.b_vector_,
                "c_vector": analyze.c_vector_,
                "a_lat": analyze.a_lat,
                "b_lat": analyze.b_lat,
                "c_lat": analyze.c_lat,
                "alpha": analyze.alpha,
                "beta": analyze.beta,
                "gamma": analyze.gamma}

    pm.bvspotential(data_dict, analyze.coordinatedict, 
                    pm.read_inparamsMorse("in.params.Morse"), 
                    pm.read_inparamsCoulomb("in.params.Coulomb"),
                    sigmoid_switch=sigmoid_switch, sigmoid_width=sigmoid_width, sigmoid_shift=sigmoid_shift)

    if angle_trio:
        pm.anglepotential(angle_trio, pm.read_inparamsangular("in.params.angular"), analyze.coordinatedict)

    pstock = pm.stock()

    pm.write_rho(data_dict, pstock)
    pm.write_dat(data_dict, pstock)

    print("\n")
    print("---------------------")
    print("output: BVSxmap.rho")
    print("output: BVSx_xyzp.dat")

    end_time  =  time.time()
    elapsed_time  =  end_time - start_time
    print("\n")
    print("Elapsed time: {} s\n".format(int(elapsed_time)))
