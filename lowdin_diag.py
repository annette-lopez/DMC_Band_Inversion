#!/usr/bin/env python3
import os
import sys

import numpy as np
import xml.etree.ElementTree as ET

import csv
import pandas as pd

from debug import ci
from developer import error

from uncertainties import ufloat, unumpy
from numerics import simstats
from hdfreader import read_hdf

from qmcpack_input import QmcpackInput
from qmcpack_analyzer import QmcpackAnalyzer

from pwscf_postprocessors import ProjwfcAnalyzer


printing_on = True #use this if you want to see per k-point printing
ao_print    = False
qmc_on      = True #use this if you want to collect QMC charges
equil       = 0 
qmc_series  = 2 #vmc wJ = 0, dmc equilibration = 1, dmc final = 2


#Begin Definitions

#Access within runs/nscf/pwscf_output/pwscf.save/atomic_proj.xml
def collectValuesFromAtomicProj(xmlfile):

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    header = root.find('.//HEADER')

    # Find number of bands
    nBands = int(header.attrib['NUMBER_OF_BANDS'])
    # Find number of kpoints
    nKpoints = int(header.attrib['NUMBER_OF_K-POINTS'])
    print("\nThe number of k-points is    ", nKpoints)
    # Find number of atomic wave functions
    nAtomicWFC = int(header.attrib['NUMBER_OF_ATOMIC_WFC'])
    # Find number of spin components
    nSpin = int(header.attrib['NUMBER_OF_SPIN_COMPONENTS'])

    kWeights = np.empty((nKpoints),dtype=float)

    #Create a list of the k-points
    kValues = []
    for i in root.findall('EIGENSTATES/K-POINT'):
        k_tuple = list(map(float, i.text.split()))
        kValues.append(k_tuple)

    atomicProjections = np.empty((nKpoints,nSpin,nAtomicWFC,nBands),dtype=complex)
    # Find atomic projections
    for k in range(nKpoints):
        kWeights[k] = float(root.findall('EIGENSTATES/K-POINT')[k].attrib['Weight'])
        for s in range(nSpin):
            for awfc in range(nAtomicWFC):
                if nSpin==1:
                    for b, text in enumerate(root.findall('EIGENSTATES/PROJS')[k][awfc].text.strip().splitlines()):
                        proj = float(text.split()[0])
                        proj = proj+complex(0,float(text.split()[1]))
                        # zeroth element below is for spin-type. In this case there is only one
                        atomicProjections[k][0][awfc][b]=proj
                    #end for
                else:
                    for b, text in enumerate(root.findall('EIGENSTATES/PROJS')[s*nKpoints+k][awfc].text.strip().splitlines()):
                        proj = float(text.split()[0])
                        proj = proj+complex(0,float(text.split()[1]))
                        atomicProjections[k][s][awfc][b]=proj
                    #end for
                    #for b, text in enumerate(root.find('EIGENSTATES/PROJS')[k][s][awfc].text.strip().splitlines()):
                    #    proj = float(text.split()[0])
                    #    proj = proj+complex(0,float(text.split()[1]))
                    #    atomicProjections[k][s][awfc][b]=proj
                    ##end for
                #end if
            #end for
        #end for
    #end for


    atomicOverlaps = np.empty((nKpoints,nSpin,nAtomicWFC,nAtomicWFC),dtype=complex)
    # Find atomic overlaps
    for k in range(nKpoints):
        for s in range(nSpin):
            if nSpin==1:
                for o, text in enumerate(root.findall('OVERLAPS/OVPS')[k].text.strip().splitlines()):
                    ovlp = float(text.split()[0])
                    ovlp = ovlp+complex(0,float(text.split()[1]))
                    atomicOverlaps[k][0][o//nAtomicWFC][o%nAtomicWFC]=ovlp
                #end for
            else:
                for o, text in enumerate(root.findall('OVERLAPS/OVPS')[s*nKpoints+k].text.strip().splitlines()):
                    ovlp = float(text.split()[0])
                    ovlp = ovlp+complex(0,float(text.split()[1]))
                    atomicOverlaps[k][s][o//nAtomicWFC][o%nAtomicWFC]=ovlp
                #end for
            #end if
        #end for
    #end for

    ###Not used
    invAtomicOverlaps = np.copy(atomicOverlaps)
    tmp = np.copy(atomicOverlaps)
    # Store inverse of atomic overlaps
    for k in range(nKpoints):
        for s in range(nSpin):
            invAtomicOverlaps[k][s] = np.linalg.inv(tmp[k][s])
        #end for
    #end for

    assert len(kWeights)==nKpoints
    assert len(atomicProjections)==nKpoints
    assert len(atomicOverlaps)==nKpoints
    assert len(kValues)==nKpoints

    assert atomicProjections.shape==(nKpoints,nSpin,nAtomicWFC,nBands)

    ## jtk debug
    #nKpoints = 1
    #kWeights = kWeights[0:1]
    #atomicProjections = atomicProjections[0:1]
    #atomicOverlaps = atomicOverlaps[0:1]
    #kValues = kValues[0:1]
    
    return nBands,nKpoints,kWeights,nAtomicWFC,nSpin,atomicProjections,atomicOverlaps,invAtomicOverlaps, kValues
#end def

#Access within runs/nscf/pwscf_output/pwscf.xml
def collectValuesFromXML(xmlfile):

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    #Get the lattice parameter and lattice vectors
    children = {c.tag for c in root.findall('./')}
    #print("The children are: ", children)

    grandkids = {g.tag for g in root.findall('./input/*')}
    #print("The grandkids are: ", grandkids)

    descendents = {d.tag for d in root.findall('./input/atomic_structure/*')}
    #print("The descendents are: ", descendents)

    distant_relatives = {r.tag for r in root.findall('input/atomic_structure/cell/*')}
    #print("The distant relatives (twice removed) are: ", distant_relatives)

    alat = float(root.find('input/atomic_structure').attrib['alat']) 
    print("\nalat    ", alat)
   
    get_a1 = root.find('input/atomic_structure/cell/a1').text
    #print(get_a1)
    
    a1 = [float(a) for a in get_a1.split()] 
    print("a1   ", a1)

    get_a2 = root.find('input/atomic_structure/cell/a2').text

    a2 = [float(a) for a in get_a2.split()]
    print("a2   ", a2)
    get_a3 = root.find('input/atomic_structure/cell/a3').text

    a3 = [float(a) for a in get_a3.split()]
    print("a3   ", a3)

    #Collect atom names
    atoms = []
    for i in root.iter('atom'):
        atoms.append(str(i.get('name'))+str(i.get('index')))

    orbitals = ['s', 'pz', 'px', 'py', 'dz2', 'dxz','dyz','dx2-y2','dxy', 'fz3', 'fxz2','fyz2','fzx2-zy2','fxyz','fx3-3xy2','f3yx2-y3']
    tot_orbs = ['s', 'p', 'd', 'f']

    #Consider the noncollinear SOC case
    for tag in root.findall('input/spin/noncolin'):
        val = (tag.text).capitalize()
        booval = eval(val)
    if booval:
        print("\nThis is a noncollinear calculation")
        collinear = False
    else:
        print("\nThis is a collinear calculation")
        collinear = True

    spin_orbs = ['s u', 's d', 'pz u', 'px u', 'py u', 'pz d', 'px d', 'py d', 'dz2 u', 'dxz u', 'dyz u', 'dx2-y2 u', 'dxy u', 'dz2 d', 'dxz d', 'dyz d', 'dx2-y2 d', 'dxy d', 'fz3 u', 'fxz2 u', 'fyz2 u', 'fzx2-zy2 u', 'fxyz u', 'fx3-3xy2 u', 'f3yx2-y3 u', 'fz3 d', 'fxz2 d', 'fyz2 d', 'fzx2-zy2 d', 'fxyz d', 'fx3-3xy2 d', 'f3yx2-y3 d']
    
    #Collect total magnetization, if not present take absolute magnetization
    totmag = 0.0 
    assert root.find('.//magnetization') is not None
    s = root.find('.//magnetization/total')
    if s is not None:
        totmag = int(float(s.text))
    #end if
    else:
        s = root.find('.//magnetization/absolute')
        if s is not None:
            assert float(s.text)<1e-3
        #end if

    #totmag = int(float(root.find('.//magnetization/total').text))
    nElec = int(float(root.find('.//nelec').text))
    nAtom = int(float(root.find('.//atomic_structure').attrib['nat']))

    #Take only unique names
    del atoms[nAtom:]

    return nAtom,nElec,int((nElec+totmag)/2),int((nElec-totmag)/2),orbitals, atoms, tot_orbs, spin_orbs, collinear
#end def

def get_qmcpack_run_info(qmc_dir,qmc_prefix,series):
    assert os.path.exists(qmc_dir)
    ntwists = 0
    infile_single = None
    stat_files = []
    #print(qmc_dir)
    for d in os.listdir(qmc_dir):
        #print(d)
        if os.path.isdir(os.path.join(qmc_dir, d)): 
            for f in os.listdir(os.path.join(qmc_dir, d)):
                if not f.startswith(qmc_prefix):
                    continue
                #end if
                #if 'twistnum' in f and f.endswith('.in.xml'):
                if f.endswith('in.xml'):
                    ntwists += 1
                    infile_single = f
                    base = d
                    #print("base path ", base)
                #end if
                ss = 's{}'.format(str(series).zfill(3))
                if f.endswith(ss+'.stat.h5'):
                    stat_files.append(os.path.join(qmc_dir,d,f))
                #end if
            #end for f
        else: continue
    #end for d
    #print(stat_files)
    #print(ntwists,len(stat_files))
    assert ntwists==len(stat_files)
    #print(infile_single)
    #quit()
    qi = QmcpackInput(os.path.join(qmc_dir,base,infile_single))
    tm = qi.get('tilematrix')
    nprim_cells = np.abs(np.linalg.det(tm))
    supercell = np.abs(nprim_cells-1)>1e-3
    return supercell,stat_files
#end def

def read_einspline(file_path):
    text = open(file_path, 'r').read()

    #Split the first line as the header from the rest of the text
    header, text = text.split('\n',1)

    #make a list of the names, excluding the first '# '
    names = header.split()[1:]
    #print(names)

    #Take the big text block and split every entry into it's own string
    data = np.array(text.split())
    #print(data)

    #alter the shape so that we create a list of rows, where each row is a list itself
    #11 rows and 12 columns for diamond
    #print(len(names))
    #print(len(data))
    data.shape = len(data)//len(names), len(names)
    #print(data)

    #Correct the data so that we have a list of column lists
    data = data.T
    #print(data)
    dt = {}
    for name,d in zip(names, data):
        try:
            d = np.array(d,dtype = int)
        except:
            d = np.array(d,dtype = float)
        #end try
        dt[name] = d
    #end for

    df = pd.DataFrame(dt)
    #print(df)

    #Access the relevant columns in the einspline.dat file
    twistseries = df['TwistIndex']
    k1series    = df['K1']
    k2series    = df['K2']
    k3series    = df['K3']
    stateseries = df['State']

    #For each different TwistIndex create a list of k-point tuples
    twistIndex = pd.unique(twistseries) #these are our future dictionary keys
    #print(twistseries.values)
    #print("Unique twist indices: ", twistIndex)
    #print(type(twistIndex))

    #Identify the unique k-points
    kvals = []
    for i in range(len(twistIndex)):
        for j in range(len(twistseries)):
            if twistseries[j] in twistIndex:
                mytuple = (k1series[j], k2series[j],k3series[j])
                kvals.append(mytuple)
    kpoints = pd.unique(kvals)
    #print("The associated k-points for each twist index: ", kpoints)

    #Construct the set of states associated with each twistIndex
    state_collection = []
    for i in range(len(twistIndex)):
        templist = []
        for j in range(len(twistseries)):
            if twistseries[j] == twistIndex[i]:
                templist.append(stateseries[j])
                #print(templist)
        state_collection.append(templist)
    #print("The set of states per k-point and twist index: ", state_collection)

    #Next, create a dictionary, where for each supercell twist we can access the k-point
    k_dict = dict(zip(twistIndex, kpoints))
    #print("The associated k-points for each twist index: ", k_dict)

    #Next, create a dictionary, where for each supercell twist we can access the set of states (orbitals)
    s_dict = dict(zip(twistIndex, state_collection))
    #print("The set of states per k-point and twist index: ", s_dict)

    #Check the number of orbitals is correct
    #nStates = len(s_dict[twistIndex[0]])
    #print(nStates)
    #if nStates == 18: print("Good job!")
    return (k_dict, s_dict)
#end def

def read_density_matrix_from_stat_h5(filepath,equil=0):
    print(filepath)
    cwd = os.getcwd()
    print("current working directory ", cwd)
    assert os.path.exists(filepath)
    h = read_hdf(filepath)
    dm_names = ['DensityMatrices','OneBodyDensityMatrices']
    dm = None
    for dm_name in dm_names:
        if dm_name in h:
            dm = h[dm_name]
            break
        #end if
    #end for
    if dm is None:
        error('density matrix data is missing from stat.h5')
    #end if
    nm = dm.number_matrix
    for s in ('u','d'):
        if s in nm:
            val = nm[s].value
            nb,norb1,norb2,ntmp = val.shape
            v = val[:,:,:,0] + 1j*val[:,:,:,1]
            v.shape = nb,norb1,norb2
            nm[s] = v[equil:]
        #end if
    #end for
    return nm
#end def

def collect_data(matrix, num_k, k_values, start):                
    #print("checking inputs")
    #print("\n")
    #print(num_k, k_values, start, matrix)
    #quit()
    main_dict = dict(zip(k_values, [None]*len(k_values)))
    for k in range(num_k):
        if printing_on:
            print("k-point: " + str(k_values[k]) + "\n")
        #end if

        #Go through each atomic species Bi1, Bi2, Te3, Te4, Te5 and create a list of their corresponding wfcs
        full_occs = []
        for i in range(len(start) - 1):    
            j_list = list(range(start[i],start[i+1]))
            #Based on the atom, the shells can include contributions from s,p,d, and f with each orbital resolved into xyz contributions
            if collinear:
                if len(j_list) == 16: x = [1,3,5,7]
                elif len(j_list) == 9: x = [1,3,5]
                elif len(j_list) == 4: x = [1,3]
                else: x = [1]
            else:
                if len(j_list) == 32: x =[l*2 for l in [1,3,5,7]]
                elif len(j_list) == 18: x = [l*2 for l in [1,3,5]]
                elif len(j_list) == 8: x =[l*2 for l in [1,3]]
                else: x = [l*2 for l in [1]]
            #end if
            #print(x)

            #make a list of sublists of the wfcs that need to be added to produce the total occupation at a k-point. E.g. pz+px+py = p for collinear, or pz up + px up +py up + pz dwn + px dwn + py dwn = p
            b = 0
            sublist = []
            for a in x:
                sublist.append(j_list[b:b+a])
                b += a
            #end for

            #Use the previous sublist to extract the corresponding occupancies at each wfc number
            sumset = []
            for o in range(0, len(x)):
                subset = []
                for g in sublist[o]:
                    subset.append(matrix[k][sp][g][g].real)
                #end for
                sumset.append(subset)
            #end for

            #Create a list of the summed orbital occupancies at a specific k-point
            finalset = []
            for e in range(0, len(sumset)):
                finalset.append(sum(sumset[e]))
            #end for

            if printing_on:
                for j, n in enumerate(shells[i]):
                    print("         charge on " + keys[i] + " , " + n + " = " + str(finalset[j]))
                #end for
            #end if
        
            #Create a list of all occupations for all atoms to assign to our main dictionary
            full_occs += finalset
        #end for i
        #print(full_occs) 
        main_dict[k_values[k]] = full_occs
    #print(main_dict)
    #end for k  
    return main_dict
#end def

def save_data(sys_dict, names, input_dict, num_k, data_type, col_check = True):
    #Note: data_type is either 'qe' or 'qmc' or 'qmcerror'
    print("\n")
    print("Writing " + data_type + " k-point resolved data to file")

    #Make a list of starting point indices for the larger loop below
    strt_pts = [0]
    i = 0
    for k, v in sys_dict.items():
        strt_pts.append(strt_pts[i]+len(sys_dict[k]))
        i += 1
    #print(strt_pts)
    #end for

    kxvals = []
    kyvals = []
    kzvals = []
    for k in kValues:
        for i, v in enumerate(k):
            if i == 0: kxvals.append(v)
            elif i == 1: kyvals.append(v)
            elif i == 2: kzvals.append(v)
            #end if
        #end for i,v
    #end for k

    for i in range(len(strt_pts) -1):
        #Each atom will have a different number of orbitals, so we need different number of columns
        headers = ['KX', 'KY', 'KZ']
        for j in shells[i]:
            headers += j
        #end for j
        
        df = pd.DataFrame(columns = [n for n in headers])

        #Fill our lists of data
        s = []
        p = []
        d = []
        f = []
        for k, key in enumerate(list(input_dict.keys())):
            for n, v in enumerate(input_dict[key][strt_pts[i]:strt_pts[i+1]]):
                if n == 0: s.append(v*num_k)
                try:
                    if n == 1: p.append(v*num_k)
                except:
                    pass
                try:
                    if n == 2: d.append(v*num_k)
                except:
                    pass
                try:
                    if n == 3: f.append(v*num_k)
                except:
                    pass
            #end for n, v
        #end for k, key
        
        L = len(kxvals)
        l = len(s)
        
        if l != L: 
            print("\nnumber of k-points and occupancies unequal, check for repeating k-points") 
            print("\nLength of k-points: ", L)
            print("\nLength of occupancies: ", l)
        #end if

        df['KX'] = kxvals
        df['KY'] = kyvals
        df['KZ'] = kzvals

        col = []
        for c in df.columns:
            col.append(df.columns.get_loc(c))
        del col[0:3]
        heads = headers.copy()
        del heads[0:3]
        tmp = zip(col, heads)
        for c, h in tmp:
            if c == 3: df[h] = s
            try:
                if c == 4: df.iloc[:, c] = p
            except:
                pass
            try:
                if c == 5: df.iloc[:, c] = d
            except:
                pass
            try:
                if c == 6: df.iloc[:, c] = f
            except:
                pass
        #print(df.to_string(index = False))
        if col_check:
            np.savetxt('{name}_{typ}.dat'.format(name = names[i], typ = data_type), df)
        else:
            np.savetxt('{name}_{typ}_soc.dat'.format(name = names[i], typ = data_type), df)
        #end if
    #end for i
#end def

def check_data(sys_dict, input_dict, k_values, names, num_k, data_type, col_check = True):
    total_charge_dict = dict(zip(k_values, [None]*len(k_values)))
    estimate = 0
    for k,v in input_dict.items():
        total_charge = sum(input_dict[k])
        total_charge_dict[k] = total_charge*num_k 
        estimate += total_charge
    #print(total_charge_dict)
    print("\nTotal charge estimate for {} system: ".format(data_type), estimate)
    print("\nAverage total charge per k-point: ", estimate/nKpoints)

    atom_dicts = []
    start = 0
    for i in range(len(names)):
        est = 0
        #print(i)
        a_dict = dict(zip(k_values, [None]*len(k_values)))
        norbs = len(sys_dict[names[i]])
        #print(norbs)
        for k,v in input_dict.items():
            added = sum(input_dict[k][start:(start+norbs)])
            a_dict[k] = added
            est += added
        start += norbs
        #print(start)
        if data_type == "qe":
            print("\nQE charge on " + atoms[i] + " " + str(est))
        else:
            print("\nQMC charge on " + atoms[i] + " " + str(est))
        atom_dicts.append(a_dict)
    #print(len(atom_dicts))

    #end of charge checks

    titles = ['KX', 'KY', 'KZ', 'Total Charge']
    titles = titles + names
    #print(titles)
    df2 = pd.DataFrame(columns = [n for n in titles])
    
    tot_charges = []
    for k, v in total_charge_dict.items():
        tot_charges.append(v)
    
    atc_lst = []
    for i in range(len(atom_dicts)):
        a_tmp_list = []
        for k,v in atom_dicts[i].items():
            a_tmp_list.append(v)
        atc_lst.append(a_tmp_list)
    #print(len(atc_lst))
    #print("length of charges column ", len(charges))
   
    kxvals = []
    kyvals = []
    kzvals = []
    for k in kValues:
        for i, v in enumerate(k):
            if i == 0: kxvals.append(v)
            elif i == 1: kyvals.append(v)
            elif i == 2: kzvals.append(v)
            #end if
        #end for i,v
    #end for k

    df2['KX'] = kxvals
    df2['KY'] = kyvals
    df2['KZ'] = kzvals

    P = len(kxvals)
    p = len(tot_charges)

    if p != P:
        print("\nunequal lengths of kpoints and charges")
        print("\nLength of charges = ", p)
        print("\nLength of k-points = ", P)
    #end if
    
    df2['Total Charge'] = tot_charges
    title_copy = titles.copy()
    title_copy = title_copy[4:]
    #print(title_copy)
    for i, el in enumerate(title_copy):
        #print(i, el)
        df2[el] = atc_lst[i]
    #end for
    if printing_on:
        print("\n")
        print(df2)
    #quit()
    if col_check:
        np.savetxt('total_charges_{typ}.dat'.format(name = names[i], typ = data_type), df2)
    else:
        np.savetxt('total_charges_{typ}_soc.dat'.format(name = names[i], typ = data_type), df2)
    #end if

#end def

#End Defintions


if __name__ == '__main__':

    # Exit if atomic_proj.xml and outdir locations not given
    
    if qmc_on:
        if(len(sys.argv)<5):
            print("Usage: lowdin.py <pw_prefix> <pw_outdir> <qmc_directory> <qmc_identifier>")
            quit()
    else:
        if(len(sys.argv)<3):
            print("usage: lowdin.py <pw_prefix> <pw_outdir>")
            quit()
    #end if

    pw_prefix = sys.argv[1]
    pw_outdir = sys.argv[2]

    if qmc_on:
        qmc_directory = sys.argv[3]
        qmc_identifier = sys.argv[4]
    #end if    
    
    sp = 0 #up

    # Collect parameters from atomic_proj.xml.
    nBands,nKpoints,kWeights,nAtomicWFC,nSpin,atomicProjections,atomicOverlaps,invAtomicOverlaps, kValues = collectValuesFromAtomicProj(pw_outdir+"/"+pw_prefix+".save/atomic_proj.xml")

    # Collect parameters from <prefix>.xml
    nAtom,nElec,nOccUp,nOccDown,orbitals, atoms, tot_orbs, spin_orbs, collinear = collectValuesFromXML(pw_outdir+"/"+pw_prefix+".xml")

    print('\nNumber of up electrons: {}'.format(nOccUp))
    print('Number of down electrons: {}'.format(nOccDown))
    print('nElec',nElec)

    #For Printing
    #First, make a dictionary of the atom names+index to the orbitals per atom
    head, tail = os.path.split(pw_outdir)
    #print("head ", head)
    #print("tail ", tail)
    pa = ProjwfcAnalyzer(head + "/pwf.in", "pwf.output", analyze = True)

    ao_dict = dict(zip(atoms, [None]*len(atoms)))
    n = 0
    for key, val in ao_dict.items():
        t = pa.lowdin[n].tot
        if collinear:
            tmp_orbs = [k for k in orbitals if k in t]
            try:
                if (abs(t.f - 0.0) < 1e-5): del tmp_orbs[9:]
            except: pass
            try:
                if (abs(t.d - 0.0) < 1e-5): del tmp_orbs[4:]
            except: pass
            if (abs(t.p - 0.0) < 1e-5): del tmp_orbs[1:]
        else:
            tmp_orbs = spin_orbs.copy()
            try:
                if (abs(t.f - 0.0) < 1e-5): del tmp_orbs[18:]
            except: pass
            try:
                if (abs(t.d - 0.0) < 1e-5): del tmp_orbs[8:]
            except: pass
            if (abs(t.p - 0.0) < 1e-5): del tmp_orbs[2:]
        #end if
        ao_dict[key] = tmp_orbs
        n += 1
    #end for
    print("\nAO dictionary\n", ao_dict)
    #quit()
    
    #Next make a dictionary of the atom names+index to their total orbitals and a list of values to check our summations over orbitals below
    tot_dict = dict(zip(atoms, [None]*len(atoms)))
    check = []
    n = 0
    for key, val in ao_dict.items():
        t2 = pa.lowdin[n].tot
        tmp_orbs = [k for k in tot_orbs if k in t2]
        check.append(t2.s)
        try:
            if (abs(t2.p - 0.0) < 1e-5):
                del tmp_orbs[1:]
            else:
                check.append(t2.p)
        except:
            pass
        try:
            if (abs(t2.d - 0.0) < 1e-5):
                del tmp_orbs[2:]
            else:
                check.append(t2.d)
        except:
            pass
        try:
            if (abs(t2.f - 0.0) < 1e-5):
                del tmp_orbs[3]
            else:
                check.append(t2.f)
        except:
            pass
        tot_dict[key] = tmp_orbs
        n += 1
    print("\nFinal dictionary\n", tot_dict)
    #quit()

    #Take the keys amd values as a list
    keys   = list(tot_dict.keys()) #atom names
    shells = list(tot_dict.values())

    #Make a list of starting point indices for the big loop below
    starting_pts = [0]
    i = 0
    for key, values in ao_dict.items():
        starting_pts.append(starting_pts[i]+len(ao_dict[key]))
        i += 1
    #print(starting_pts)
    #end for

    #This is where we can print to terminal the per k-point contributions, but it is mainly where we accumulate those contributions into our main dictionary
    newKvals = [tuple(v) for v in kValues]
    #gamma = (0.0, 0.0, 0.0)
    #print(newKvals[0])
    #newKvals[450] = (1e-10, 0.0, 0.0)
    #print(1e-10)
    #print(newKvals[450])
    #for i, v in enumerate(newKvals):
    #    if v == newKvals[0]: print(i)
    #print(newKvals.count(gamma))
    #print(len(kValues))
    #print(len(newKvals))
    #print(nKpoints)
    #test = [None]*len(newKvals)
    #print(len(test)) 

    # Obtain exact number matrix corresponding to single determinant with no jastrow, projected
    # on AO basis.
    if collinear: pass
    else: nOccUp *= 2

    exct_nmqmc = np.empty((nKpoints,nSpin,nAtomicWFC,nAtomicWFC),dtype=complex)
    for k in range(nKpoints):
        exct_nmqmc[k][sp] = kWeights[k]*np.matmul(atomicProjections[k][sp][:,:nOccUp],np.conj(atomicProjections[k][sp][:,:nOccUp].T))
    #end for
    #Sum over all k-points
    exavg = np.sum(exct_nmqmc,axis=0)

    print("\nTotal Charge of system (QE): " + str(np.trace(exavg[sp].real)) +"\n")
    
    #if printing_on:
    if ao_print:
        for a in range(nAtomicWFC):
            print("          charge on AO "+str(a)+" = "+str(exavg[sp][a][a].real))
        #end for

        print("QE charges resolved per k-point on each AO" + "\n")
    #end if
    
    #We want the per k-point breakdown of orbital occupations
    main_qedict = collect_data(exct_nmqmc, nKpoints, newKvals, starting_pts) 
    #quit()


    #Checking our results
    sum_qek = [0]*(len(check))
    for k,v in main_qedict.items():
        for c in range(len(sum_qek)):
            sum_qek[c] += main_qedict[k][c]
        #end for
    #end for
    print("Our summed charges: ", sum_qek)
    print("\nDFT's charges: ", check)

    comparison = True
    c = 0
    for i in range(len(check)):
        if (abs(sum_qek[i] - check[i])< 1e-2):
            #print(abs(sum_qek[i] - check[i]))
            continue
        else: c += 1
        #end if
    #end for
    if c > 5: comparison = False
    if comparison and c < 5:
        print("\nGood agreement between QE results and DFT lowdin charges")
    else:
        print("\nFound bad agreement between QE results and DFT lowdin charges")
    #end if

    #Now write to files
    save_data(tot_dict, atoms, main_qedict, nKpoints, 'qe', col_check = collinear)  
    check_data(tot_dict, main_qedict, newKvals, atoms, nKpoints, 'qe', col_check = collinear)

    #QMC Analysis for primitive and supercell runs
    #print('bef qmc on')
    if qmc_on:
        
        supercell,stat_files = get_qmcpack_run_info(qmc_directory,qmc_identifier,qmc_series)

        #print('aft qmc on')
        #Definition to enter into each einspline file to map from supercell twists to primitive cell twists
        #print("The current working directory is: ", cwd)

        #nwd = os.getcwd() #new working directory (qmc directory)
        #print("The current working directory is: ", nwd)

        
        #Analyze QMC data
        nm = []  # number matrix

        if not supercell:
            for stat_file in stat_files:
                nmh5 = read_density_matrix_from_stat_h5(stat_file,equil)
                if sp==0:
                    nm.append([nmh5.u])
                else:
                    nm.append([nmh5.d])
                #end if
            #end for
            #print("quitting here")
            #quit()
        else:
            cwd = os.getcwd()
            #print("current working directory ", cwd)
            os.chdir(qmc_directory)
            nwd = os.getcwd()
            #print("new working directory ", nwd)
            count = 0
            #Reading all einspline files
            for f in os.listdir():
                #print(f)
                if f.endswith('bandinfo.dat'):
                    #print("checking error: ", f)
                    #print("type of f: ", type(f))
                    x = f.find("g")
                    y = f.find(".band")
                    #print(x)
                    #print(y)
                    z = slice(x+1,y)
                    #print(f[z])
                    twist = int(f[z])
                    #file_path = "{}/{}".format(cwd + "/" + qmc_directory,f)
                    file_path = "{}/{}".format(qmc_directory,f)
                    k_dict, s_dict = read_einspline(file_path)
                    tn = count
                    print("\n Supercell Twist #", twist)
                    print("Unique twist indices: ", list(k_dict.keys()))
                    print("K-point per twist: \n",k_dict)
                    print("Set of primitive cell states per twist: \n", s_dict)

                    # read density matrix data from stat.h5 file
                    #stat_h5_filename = '{}.g{:03d}.s000.stat.h5'.format(qmc_identifier,tn)
                    stat_h5_filename = '{}.s000.stat.h5'.format(qmc_identifier, tn)
                    nmh5 = read_density_matrix_from_stat_h5(stat_h5_filename,equil)

                    # get the density matrix (called the number matrix here)
                    nm_tmp = []
                    if sp==0:
                        nm_tmp.append(nmh5.u)
                    else:
                        nm_tmp.append(nmh5.d)
                    #end if
                    nmt = nm_tmp
                    nmt = np.array(nmt)
                    print(nmt.shape)

                    for key,value in k_dict.items():
                        #assign a list the set of states for each key in the dictionary
                        s = s_dict[key]
                        s = np.array(s)
                        #print(s)
                        #print(nmt[:,:,s,s])
                        nmk = nmt[:,:,s][:,:,:,s]
                        #print(nmk.shape)
                        nm.append(nmk)
                    #end for key, value
                    count += 1
                #end if
            #end for f
            #print(nm)
            os.chdir(cwd)
        #end if
        nm = np.array(nm)
        print(nm.shape)
        #print("Final number matrix dimensions: ", nm.shape)
        nblocks,nstates,nstates = nm[0][0].shape         
        #quit()
        
        # Perform "unitary" transform on each block's number matrix individually
        # and store in nmqmcu (i.e., up component of number matrix prime)
        # After the transformation, number matrix has been transformed from
        # the MO basis to the AO basis

        ###Annette checks 11/4/2024
        #creating diagonal only matrix
        for k in range(nKpoints):
            for b in range(nblocks):
                new_matrix = nm[k][sp][b][:,:]
                diag_matrix = np.zeros_like(new_matrix)
                np.fill_diagonal(diag_matrix, np.diagonal(new_matrix))
                nm[k][sp][b] = diag_matrix
        #print(nm)
        #print(nm.shape)
        
        #creating off diagonal only matrix
        #for k in range(nKpoints):
        #    for b in range(nblocks):
        #        np.fill_diagonal(nm[k][sp][b], 0)
        #print(nm)
        #print(nm.shape)
        
        #np.fill_diagonal(nm, 0) ##off diagonals only, just set diagonals to 0
        ##diag_matrix = np.zeros_like(nm) #creating matrix with only diagonal elements
        ##np.fill_diagonal(diag_matrix, np.diagonal(nm)) #fill the zero matrix's diagonal
        ##print(diag_matrix.shape)
        #exit()
        ###end checks
       

        assert nstates<=nBands
        ns = nstates

        nmqmc = np.empty((nKpoints,nSpin,nblocks,nAtomicWFC,nAtomicWFC),dtype=complex)

        for k in range(nKpoints):#Problem here is that nKpoints its 144 from nscf, but 36 twists
            for b in range(nblocks):
                nmqmc[k][sp][b] = kWeights[k]*np.matmul(atomicProjections[k][sp][:,:ns],np.matmul(nm[k][sp][b][:,:],np.conj(atomicProjections[k][sp][:,:ns].T)))
            #end for
        #end for

        # jtk: magic normalization factor for Bi2Te3 w/ SOC !!
        #nmqmc *=14

        
        m_ao,v_ao,e_ao,k_ao = simstats(nmqmc,dim=2)

        #K-point resolved m_ao without uncertainties (for plotting purposes)
        copy_means = m_ao
        #print("Copy_means ", copy_means)
        #print("e_ao ",e_ao)
        #print("The dimensions of matrix are " + str(copy_means.shape))

        m_ao_avg = np.sum(unumpy.uarray(m_ao.real,e_ao.real),axis=0)
        #print(m_ao_avg)
        #print("The dimensions of matrix are " + str(m_ao_avg.shape))
    
        #K-point resolved m_ao with uncertainty
        m_ao = unumpy.uarray(m_ao.real,e_ao.real)
        #print("The dimensions of array are " + str(m_ao.shape))
        #print("Array m_ao ", m_ao)

        print("\nTotal Charge of system (QMCPACK): " + str(np.trace(m_ao_avg[sp])) +"\n")
        #if printing_on:
        if ao_print:
            for a in range(nAtomicWFC):
                print("          charge on AO "+str(a)+" = "+str(m_ao_avg[sp][a][a]))
            #end for
            
            print("QMC charges resolved per k-point on each AO" + "\n")
        #end if         

        main_qmcdict = collect_data(copy_means, nKpoints, newKvals, starting_pts)

        #Checking our qmc results
        sum_qmck = [0]*(len(check))
        for k,v in main_qmcdict.items():
            for q in range(len(sum_qmck)):
                sum_qmck[q] += main_qmcdict[k][q]
            #end for
        #end for
        #print("Our summed QMC charges: ", sum_qmck)
        #print("\n DFT's charges: ", check)

        comparison = True
        q = 0
        for i in range(len(check)):
            if (abs(sum_qmck[i] - check[i])< 1e-2):
                continue
            else: q += 1
            #end if
        #end for

        if q > 5: comparison = False
        if comparison and q < 5:
            print("\nGood agreement between QMC results and DFT lowdin charges")
        else:
            print("\nFound bad agreement between QMC results and DFT lowdin charges")
        #end if

        #Now write to files
        save_data(tot_dict, atoms, main_qmcdict, nKpoints,'qmc', col_check = collinear)
        check_data(tot_dict, main_qmcdict, newKvals, atoms, nKpoints, 'qmc', col_check = collinear)
        main_qmcerror_dict = collect_data(e_ao, nKpoints, newKvals, starting_pts) 
        save_data(tot_dict, atoms, main_qmcerror_dict, nKpoints, 'qmcerror', col_check = collinear)

        diff_qmc = []
        for i in range(len(check)):
            m = abs(sum_qmck[i] - check[i])
            diff_qmc.append(m)
    
    #end if qmc_on 

    #Print out QE and QMC differences from Quantum Espresso PWF Output
    diff_qe = []
    for i in range(len(check)):
        d = abs(sum_qek[i] - check[i])
        diff_qe.append(d)

    check_list = [sum_qek, check, diff_qe]
    column_head = ['QE', 'PWF Output', 'QE Difference']

    if qmc_on:
        check_list.insert(1, sum_qmck)
        check_list.insert(4, diff_qmc)
        column_head.insert(1, 'QMC')
        column_head.insert(4, 'QMC Difference')

    data_frame = pd.DataFrame(columns = column_head)
    for x in range(len(check_list)):
        data_frame[column_head[x]] = check_list[x]
    if printing_on:
        print('\n')
        print(data_frame)
    np.savetxt("check.dat", data_frame)
    #end printing

    #dm  = data_frame['QMC'].to_numpy()
    #dme = data_frame['QMC Difference'].to_numpy()
    #print(dme/dm)

#end if main




















