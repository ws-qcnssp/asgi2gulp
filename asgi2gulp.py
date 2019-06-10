# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 23:35:30 2018

@author: ws

This module allows to generate models of amorphous materials in a form
of a cluster by bottom-up approach. It is an interface to General Utility
Lattice Program (GULP), which is used in MM calculations.

It can also take the GULP output file with IR frequencies and intensities
and generate IR spectrum. 
"""

max_bond_length = {
	"Ag-Te" : 4.5,
	"Sb-Te" : 4.5,
	"Si-C" : 2.3,#2.1
	"Si-O" : 1.9,#1.7
	"O-H" : 1.2,#1.1
	"C-H" : 1.3,#1.1
	"C-C" : 1.65,#1.55
	"H-H" : 0.9,
	"O-O" : 1.6,
	"Si-Si" : 1.8,
	"Si-H" : 1.5,
	"default" : 0.8,
}

cut_bonds = {
	"O-H" : 1.0,
}

MIN_BOND_LENGTH = 0.01
EULER = 2.718281828459
PI = 3.141592653589
phiMesh = [2 * PI / 36 * x for x in range(36)]
thetaMesh = [2 * PI / 36 * x for x in range(18)]
BOND_DEFAULT=1.6

import numpy as np
import math
import copy
import random
import string
import subprocess
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
#from os.path import basename

# checking for $GULP_SRC in a system
try:
	GULP_SRC = os.environ["GULP_SRC"]
except KeyError as e:
	print("Evironment variable GULP_SRC is not defined!")
	print("You must specify this variable before continuing.")
	print("See GULP user's guide for details.")
	raise



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))

def distance(At1, At2):
	dx = At1.x - At2.x 
	dy = At1.y - At2.y 
	dz = At1.z - At2.z
	result = ( dx ** 2 + dy ** 2 + dz **2 ) ** 0.5
	return result

def vec_length(x, y, z):
	res = ( x ** 2 + y ** 2 + z **2 ) ** 0.5
	return res
	
def total_distance(mol1, mol2):
	sum = 0.0
	mol1reduced = set()
	for atom in mol1.atoms:
		if vec_length(atom.x, atom.y, atom.z) < 7.5:
			mol1reduced.add(atom)
	for at1 in mol1reduced:
		for at2 in mol2.atoms:
			sum += distance(at1, at2)
	return sum
	
def total_distance_atom_sets(set1, set2):
	sum = 0.0
	for atom1 in set1:
		for atom2 in set2:
			sum += distance(atom1, atom2)
	return sum

def total_distance_mat(mol1m, mol2m):
	sum = 0.0
	lenM1 = len(mol1m)
	lenM2 = len(mol2m)
	for i in range(lenM1):
		for j in range(lenM2):
			sum += np.linalg.norm(mol1m[i]-mol2m[j])
	return sum

	
def rad(angle):
	res = ( angle / 180.0 ) * PI
	return res
	
def bond_type(At1, At2):
	typeAB = At1.type + '-' + At2.type
	typeBA = At2.type + '-' + At1.type
	if typeAB in max_bond_length:
		return typeAB
	elif typeBA in max_bond_length:
		return typeBA
	else:
		return "default"
		
def bonded(At1, At2):
	type = bond_type(At1, At2)
	max = max_bond_length[type]
	dist = distance(At1, At2)
	if dist > MIN_BOND_LENGTH and dist < max:
		return dist
	else:
		return 0

def mol_copy(mol):
	molc = Molecule(str(mol.name))
	for atom in mol.atoms:
		x = float( atom.x )
		y = float( atom.y )
		z = float( atom.z )
		type = str( atom.type )
		name = str( atom.name )
		idNum = int( atom.id )
		Atom(idNum, name, type, x, y, z, molc)
	molc.set_bonds()
	return molc
	
def generate_sphere_points(At):
	points = Molecule("sphere")
	for phi1deg in range(36):
		for thetadeg in range(18):
			for phi2deg in range(36):
				phi1 = rad(10 * float(phi1deg))
				theta = rad(10 * float(thetadeg))
				phi2 = rad(10 * float(phi2deg))
				axisZ = [0, 0, 1]
				axisXp = [1, 0, 0]
				axisZb = [0, 0, 1]
				rotM1 = rotation_matrix(axisZ, phi1)
				axisXp = np.dot(rotM1, axisXp)
				axisZb = np.dot(rotM1, axisZb)
				rotM2 = rotation_matrix(axisXp, theta)
				axisZb = np.dot(rotM2, axisZb)
				rotM3 = rotation_matrix(axisZb, phi2)
				rotMtot = np.dot(rotM2, rotM1)
				rotMtot = np.dot(rotM3, rotMtot)
				[a, b, c] = np.dot(rotMtot, [1, 0, 0])
				newId = id_generator()
				atom = Atom(newId, "ROT", "ROT", a, b, c, points)
	points.translation(At.x, At.y, At.z)
	return points
	
def average_point_from_atoms(set1):
	avX = 0.0
	avY = 0.0
	avZ = 0.0
	for atom in set1:
		avX += atom.x
		avY += atom.y
		avZ += atom.z
	avX = avX/len(set1)
	avY = avY/len(set1)
	avZ = avZ/len(set1)
	return (avX, avY, avZ)
	
def get_vec(At1, At2):
	a = At2.x - At1.x
	b = At2.y - At1.y
	c = At2.z - At1.z
	return (a, b, c)
	
def find_connector(At1, At2):
	neigh1 = At1.get_neighbours()
	neigh2 = At2.get_neighbours()
	x, y, z = average_point_from_atoms(neigh1)
	helpAt1 = Atom(id_generator(), "HELP", "HELP", x, y, z, At1.mol)
	a, b, c = get_vec(At1, helpAt1)
	d = distance(At1, helpAt1)
	s = -BOND_DEFAULT / d
	helpAt1.translation(s*a, s*b, s*c)
	x, y, z = average_point_from_atoms(neigh2)
	helpAt2 = Atom(id_generator(), "HELP", "HELP", x, y, z, At2.mol)
	a, b, c = get_vec(At2, helpAt2)
	d = distance(At2, helpAt2)
	s = -BOND_DEFAULT / d
	helpAt2.translation(s*a, s*b, s*c)
	helpers = set()
	helpers.add(helpAt1)
	helpers.add(helpAt2)
	x, y, z = average_point_from_atoms(helpers)
	At1.mol.del_atom(helpAt1)
	At2.mol.del_atom(helpAt2)
	return (x, y, z)
	
		
def rotation_matrix(axis, theta):
	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.
	"""
	axis = np.asarray(axis)
	axis = axis/math.sqrt(np.dot(axis, axis))
	a = math.cos(theta/2.0)
	b, c, d = -axis*math.sin(theta/2.0)
	aa, bb, cc, dd = a*a, b*b, c*c, d*d
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
					 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
					 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def generate_rot():
	rotSphere = [[[0 for i in range(36)] for i in range(18)] for i in range(36)]
	for phi1deg in range(36):
		for thetadeg in range(18):
			for phi2deg in range(36):
				phi1 = rad(10 * float(phi1deg))
				theta = rad(10 * float(thetadeg))
				phi2 = rad(10 * float(phi2deg))
				axisZ = [0, 0, 1]
				axisXp = [1, 0, 0]
				axisZb = [0, 0, 1]
				rotM1 = rotation_matrix(axisZ, phi1)
				axisXp = np.dot(rotM1, axisXp)
				axisZb = np.dot(rotM1, axisZb)
				rotM2 = rotation_matrix(axisXp, theta)
				axisZb = np.dot(rotM2, axisZb)
				rotM3 = rotation_matrix(axisZb, phi2)
				rotMtot = np.dot(rotM2, rotM1)
				rotMtot = np.dot(rotM3, rotMtot)
				rotSphere[phi1deg][thetadeg][phi2deg] = copy.deepcopy(rotMtot)
	return rotSphere

rotSphere = generate_rot()
					 
					
class Gulp(object):
	def __init__(self):
		self.maxCycles = 2000
		self.library = "reaxff_general"
		self.header = "opti conp"
		self.output = "output xyz"
		
gulp = Gulp()

class Config(object):
	def __init__(self):
		self.nProc = 1
		self.connectionType = "O-H"
		self.saturationType = "O-H"
		self.saturationMethod = "connector" # default / connector
		
config = Config()

class VibConfig(object):
	def __init__(self):
		self.lorentzChar = 0.5
		self.halfwidth = 50.0
		self.maxHDisp = 1000.0
		self.plotMin = 0
		self.plotMax = 2000
		self.plotMesh = 1
		self.plotScaleY = 1000

vibConfig = VibConfig()

class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def yadd(self, num):
		self.y = self.y + num
		
			
class VibUtils(object):
	def __init__(self):
		pass
		
	def gauss(self, x, mu, sigma):
		res = 1 / ( sigma * (2 * PI) ** ( 0.5 ) ) * EULER ** \
			  ( -0.5 * ( ( x - mu ) / sigma ) ** 2 )
		return res
		
	def lorentz(self, x, mu, gamma):
		res = 1 / ( PI * gamma * ( 1 + ( ( x - mu ) / gamma ) ** 2 ) )
		return res
		
	def check_disp_h(self, Atom, x, y, z, vibConf = vibConfig):
		if Atom.type == 'H':
			length = x ** 2 + y ** 2 + z ** 2
			length = length ** 0.5
			if length > vibConf.maxHDisp:
				return 1
			else:
				return 0
		else:
			return 0
	
	def read_vibs(self, mol, inputFile, method = 1, readGoutStart = 0):
		if readGoutStart == 1:
			loadedMol = mol
		else:
			print("\nChecking input molecule:")
			loadedMol = read_gout(inputFile)
		if loadedMol.name == mol.name and len(loadedMol.atoms) == len(mol.atoms):
			if readGoutStart != 1:
				print("Molecule OK\n")
				if len(mol.vibs) > 0:
					mol.vibs.clear()
			with open(inputFile, 'r') as f:
				content = f.readlines()
			freqLines = []
			atomCount = len(mol.atoms)
			for index, line in enumerate(content):
				ln = line.split()
				if 0 < len(ln):
					if (ln[0] == "Frequency"):
						freqLines.append(index)
			#pList = set()
			#pStep = int(len(freqLines) / 10)
			counter = 0
			#for i in range(1, 11):
			#	pList.add(i*pStep)
			#sys.stdout.write("Fetching vibrations data: (method: {}) \n".format(method))
			#sys.stdout.flush()
			for nm in freqLines:
				counter += 1
				#if counter in pList:
				#	sys.stdout.write('.')
				#	sys.stdout.flush()
				sys.stdout.write('Fetching vibrations data: (method: %s) %s / %s \r' % (method, counter, len(freqLines)))
				sys.stdout.flush()
				if nm != max(freqLines):
					for i in range(13, 63, 10):
						if i + 10 < len(content[nm]):
							vibFreq = float(content[nm][i:i + 10])
							vibIntensIR = float(content[nm + 1][i:i + 10])
							vibIntensRaman = float(content[nm + 5][i:i + 10])
							vib = Vibration(vibFreq, vibIntensIR, vibIntensRaman, mol)
							if method == 2:
								for atNum in range(0, atomCount):
									atomId = int(content[nm + 7 + atNum * 3][0:6])
									dispX = float(content[nm + 7 + atNum * 3][i:i + 10])
									dispY = float(content[nm + 8 + atNum * 3][i:i + 10])
									dispZ = float(content[nm + 9 + atNum * 3][i:i + 10])
									for atom in mol.atoms:
										if atom.id == atomId:
											vib.add_disp(atom, dispX, dispY, dispZ)
			#print(" OK")
			print('\n')
		else:
			print("Loaded molecule differs! Aborting...")
			
	def plot_ir(self, mol, filename, method = 1, vibConf = vibConfig):
		sortVibs = sorted(mol.vibs, key=lambda x: x.frequency) 
		#for vib in sortVibs:
		#	print(vib.frequency)
		# setting plot points - START 
		plot = set()
		for x in range(vibConf.plotMin, vibConf.plotMax, vibConf.plotMesh):
			pt = Point(float(x), 0.0)    
			plot.add(pt)
		# setting plot points - END   
		# adding bands to the plot - START
		lorentzChar = vibConf.lorentzChar
		pList = set()
		pStep = int(len(sortVibs) / 10)
		counter = 0
		for i in range(1, 11):
			pList.add(i*pStep)
		sys.stdout.write("Calculating points for plot: ")
		sys.stdout.flush()
		for vib in sortVibs:
			counter += 1
			if counter in pList:
				sys.stdout.write('.')
				sys.stdout.flush()
			sumDisp = 0
			if method == 2:
				for disp in vib.disp:
					sumDisp += self.check_disp_h(*disp[:])
			if method == 3:
				nSi = 0
				nO = 0
				nC = 0
				nH = 0
				sumSi = 0
				sumO = 0
				sumC = 0
				sumH = 0
				for atom in mol.atoms:
					atType = atom.type
					atDX, atDY, atDZ = vib.read_disp(atom)
					disp = vec_length(atDX, atDY, atDZ)
					if atType == "Si":
						nSi += 1
						sumSi += disp
					elif atType == "O":
						nO += 1
						sumO += disp
					elif atType == "C":
						nC += 1
						sumC += disp
					elif atType == "H":
						nH += 1
						sumH += disp
				if nSi > 0:
					sumSi = sumSi / nSi
				if nO > 0:
					sumO = sumO / nO
				if nC > 0:
					sumC = sumC / nC
				if nH > 0:
					sumH = sumH / nH
				if sumH > sumSi and sumH > sumO and sumH > sumC:
					sumDisp += 1
			if sumDisp == 0:
				for pt in plot:
					val = (1.0 - lorentzChar) * self.gauss(pt.x, vib.frequency, vib.halfwidth / 2.35482) + \
						lorentzChar * self.lorentz(pt.x, vib.frequency, vib.halfwidth / 2.0)
					res = vibConf.plotScaleY * vib.intensityIR * val
					pt.yadd(res)
		# adding bands to the plot - END
		print(" OK")
		out = []
		for pt in plot:
			out.append((pt.x, pt.y))
		sout = sorted(out, key=lambda x: x[0])
		with open(filename + ".dat", 'w') as f:
			for i in range(0,len(sout)):
				f.write('{0:8.1f} {1:8.4f} \n'.format(sout[i][0], sout[i][1]))
		xout = []
		yout = []
		for i in range(0,len(sout)):
			xout.append(sout[i][0])
			yout.append(sout[i][1])
		plt.gcf().clear()
		plt.plot(xout, yout)
		plt.savefig(filename + '.png')
		#plt.show()
		
	def write_xyzvib(self, mol, filename):
		fileOut = filename + ".xyz"
		content = ''
		atomCount = len(mol.atoms)
		header = str(atomCount) + '\n' + mol.name + '  '
		svibs = sorted(mol.vibs, key=lambda x: x.frequency)
		for vib in svibs:
			content += header + str(vib.frequency) + '\n'
			for atom in mol.atoms:
				atType = atom.type
				atX = atom.x
				atY = atom.y
				atZ = atom.z
				atDX, atDY, atDZ = vib.read_disp(atom)
				content += '{} {: 2.6f} {: 2.6f} {: 2.6f} {: 2.6f} {: 2.6f} {: 2.6f}\n'. \
					format(atType, atX, atY, atZ, atDX, atDY, atDZ)
			content += '\n\n'
		with open(fileOut, 'w') as f:
			f.write(content)
			
	def write_av_disp(self, mol, filename):
		fileOut = filename + '.vib_disp'
		content = ''
		vibCount = len(mol.vibs)
		header = '# Average atomic displacements \n# Frequency Si_disp O_disp C_disp H_disp'
		content += header + '\n'
		svibs = sorted(mol.vibs, key=lambda x: x.frequency)
		currentVib = 0
		for vib in svibs:
			currentVib += 1
			nSi = 0
			nO = 0
			nC = 0
			nH = 0
			sumSi = 0
			sumO = 0
			sumC = 0
			sumH = 0
			for atom in mol.atoms:
				atType = atom.type
				atDX, atDY, atDZ = vib.read_disp(atom)
				disp = vec_length(atDX, atDY, atDZ)
				if atType == "Si":
					nSi += 1
					sumSi += disp
				elif atType == "O":
					nO += 1
					sumO += disp
				elif atType == "C":
					nC += 1
					sumC += disp
				elif atType == "H":
					nH += 1
					sumH += disp
			if nSi > 0:
				sumSi = sumSi / nSi
			if nO > 0:
				sumO = sumO / nO
			if nC > 0:
				sumC = sumC / nC
			if nH > 0:
				sumH = sumH / nH
			content += '{: 6.2f} {: 2.6f} {: 2.6f} {: 2.6f} {: 2.6f}\n'. \
				format(vib.frequency, sumSi, sumO, sumC, sumH)
			sys.stdout.write('\r Vibration %s / %s ' % (currentVib, vibCount))
		sys.stdout.write('\n')
		with open(fileOut, 'w') as f:
			f.write(content)
			
	def write_intens(self, mol, filename):
		fileOut = filename + '.vib_intens'
		content = ''
		vibCount = len(mol.vibs)
		header = '# Vibrations intensities \n# Frequency IR_intens Raman_intens'
		content += header + '\n'
		svibs = sorted(mol.vibs, key=lambda x: x.frequency)
		currentVib = 0
		for vib in svibs:
			currentVib += 1
			content += '{: 6.2f} {: 2.6f} {: 2.6f}\n'. \
				format(vib.frequency, vib.intensityIR, vib.intensityRaman)
			sys.stdout.write('\r Vibration %s / %s ' % (currentVib, vibCount))
		sys.stdout.write('\n')
		with open(fileOut, 'w') as f:
			f.write(content)
		
vibUtils = VibUtils()
					 
class Molecule(object):
	def __init__(self, Name):
		self.name = Name
		self.atoms = set()
		self.bonds = set()
		self.vibs = set()
		self.id = id_generator()
	
	def add_atom(self, Atom):
		self.atoms.add(Atom)
	
	def add_bond(self, Bond):
		self.bonds.add(Bond)
		
	def add_vib(self, Vibration):
		self.vibs.add(Vibration)
		
	def del_vib(self, Vibration):
		if Vibration in self.vibs:
			self.vibs.remove(Vibration)
		
	def del_bond(self, Bond):
		if Bond in self.bonds:
			at1 = Bond.at1
			at2 = Bond.at2
			at1.bonds.remove(Bond)
			at2.bonds.remove(Bond)
			self.bonds.remove(Bond)
		
	def del_atom(self, Atom):
		if Atom in self.atoms:
			bonds = copy.copy(Atom.bonds)
			for bond in bonds:
				self.del_bond(bond)
			self.atoms.remove(Atom)
	
	def get_mat(self):
		posList = []
		for atom in self.atoms:
			posAtom = (atom.x, atom.y, atom.z)
			posList.append(posAtom)
		posMatrix = np.array(posList)
		return posMatrix
		
	def print_atoms(self):
		for atom in self.atoms:
			print("{} {} {} {} {} {}".format(atom.id, atom.name, atom.type, atom.x, atom.y, atom.z))
		
	def print_bonds(self):
		for bond in self.bonds:
			print("{} {}".format(bond.type, bond.bondLength))
	
	def count_bonds(self, bond_type):
		counter = 0
		for bond in self.bonds:
			if bond.type == bond_type:
				counter += 1
		print("No. of " + bond_type + " bonds: " + str(counter))
	
	def print_all(self):
		self.print_atoms()
		self.print_bonds()
		
	def get_atom(self, name):
		for atom in self.atoms:
			if atom.name == name:
				return atom
		
	def adjust(self, mol2):
		result = []
		for phi1deg in range(0, 360, 30):
			for thetadeg in range(0, 180, 30):
				for phi2deg in range(0, 360, 30):
					# molc = copy.deepcopy(mol2)
					molc = mol_copy(mol2)
					phi1 = rad(phi1deg)
					theta = rad(thetadeg)
					phi2 = rad(phi2deg)
					molc.rotation(phi1, theta, phi2)
					dist_sum = total_distance(self, molc)
					# print(( dist_sum, phi1, theta, phi2 ))
					result.append( ( dist_sum, phi1, theta, phi2, phi1deg, thetadeg, phi2deg ) )
		best = max(result, key=lambda x: x[0])
		print(best)
		for phi1deg in range(best[4]-15, best[4]+15, 5):
			for thetadeg in range(best[5]-15, best[5]+15, 5):
				for phi2deg in range(best[6]-15, best[6]+15, 5):
					# molc = copy.deepcopy(mol2)
					molc = mol_copy(mol2)
					phi1 = rad(phi1deg)
					theta = rad(thetadeg)
					phi2 = rad(phi2deg)
					molc.rotation(phi1, theta, phi2)
					dist_sum = total_distance(self, molc)
					# print(( dist_sum, phi1, theta, phi2 ))
					result.append( ( dist_sum, phi1, theta, phi2, phi1deg, thetadeg, phi2deg ) )
		best = max(result, key=lambda x: x[0])
		print(best)
		mol2.rotation(best[1], best[2], best[3])
		# mol2.rotation(rad(10 * float(best[1])), rad(10 * float(best[2])), rad(10 * float(best[3])))
		molOut = Molecule(self.name + '-' + mol2.name)
		counter = 0
		for atom in self.atoms:
			x = atom.x
			y = atom.y
			z = atom.z
			counter += 1
			Atom(counter, atom.name, atom.type, x, y, z, molOut)
		for atom in mol2.atoms:
			x = atom.x
			y = atom.y
			z = atom.z
			if vec_length(x, y, z) < 0.01:
				pass
			else:
				counter += 1
				Atom(counter, atom.name, atom.type, x, y, z, molOut)
		return molOut
	
	def set_bonds(self):
		for atom1 in self.atoms:
			for atom2 in self.atoms:
				bondedAtoms = bonded(atom1, atom2)
				if bondedAtoms:
					new = 1
					for bond in self.bonds:
						if (atom1 is bond.at1 and atom2 is bond.at2) or \
							(atom1 is bond.at2 and atom2 is bond.at1):
							new = 0
					if new:
						Bond(atom1, atom2, self)
						
	def translation(self, x, y, z):
		for atom in self.atoms:
			atom.translation(x, y, z)
			
	def rotation(self, phi1, theta, phi2):
		axisZ = [0, 0, 1]
		axisXp = [1, 0, 0]
		axisZb = [0, 0, 1]
		rotM1 = rotation_matrix(axisZ, phi1)
		axisXp = np.dot(rotM1, axisXp)
		axisZb = np.dot(rotM1, axisZb)
		rotM2 = rotation_matrix(axisXp, theta)
		axisZb = np.dot(rotM2, axisZb)
		rotM3 = rotation_matrix(axisZb, phi2)
		for atom in self.atoms:
			atomVec = [atom.x, atom.y, atom.z]
			atomVec = np.dot(rotM1, atomVec)
			atomVec = np.dot(rotM2, atomVec)
			atomVec = np.dot(rotM3, atomVec)
			atom.change_pos(atomVec[0], atomVec[1], atomVec[2])

	def rot_step(self, phi1step, thetastep, phi2step):
		for atom in self.atoms:
			atomVec = [atom.x, atom.y, atom.z]
			atomVec = np.dot(rotSphere[phi1step][thetastep][phi2step], atomVec)
			atom.change_pos(atomVec[0], atomVec[1], atomVec[2])

			
	def set_zero_atom(self, Atom):
		if Atom in self.atoms:
			x = 0 - Atom.x
			y = 0 - Atom.y
			z = 0 - Atom.z
			self.translation(x, y, z)
			return (x, y, z)
			
	def activate(self, type):
		ohBonds = set()
		for b in self.bonds:
			if b.type == type:
				ohBonds.add(b)
		print(str(len(ohBonds)) + ' ' + type + ' bonds found in ' + self.name)
		transVec = 0
		while 1:
			bond = random.sample(ohBonds, 1)
			bond = bond[0]
			if bond.type == type:
				at1 = bond.at1
				at2 = bond.at2
				if at1.type == "H":
					self.del_atom(at1)
					transVec = self.set_zero_atom(at2)
					break
				elif at2.type == "H":
					self.del_atom(at2)
					transVec = self.set_zero_atom(at1)
					break
		return transVec

class Atom(object):
	def __init__(self, idNum, Name, Type, X, Y, Z, Molecule):
		self.id = idNum
		self.name = Name
		self.type = Type
		self.x = X
		self.y = Y
		self.z = Z
		self.mol = Molecule
		self.bonds = set()
		Molecule.add_atom(self)
		self.charge = "na"
		
	def add_bond(self, Bond):
		self.bonds.add(Bond)
		
	def translation(self, x, y, z):
		self.x += x
		self.y += y
		self.z += z
	
	def change_pos(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
		
	def get_neighbours(self):
		neighbours = set()
		for bond in self.bonds:
			neighbours.add(bond.second(self))
		return neighbours
		
	def get_local(self):
		mol = self.mol
		for atom in mol.atoms:
			if atom is not self:
				dist = distance(atom, self)
				if dist < 4:
					print(atom.name + ' ' + str(dist) + ' ' + str(atom.x) + ' ' + str(atom.y) + ' ' + str(atom.z))
		
	def move_toward(self, At, frac):
		a, b, c = get_vec(self, At)
		self.translation(frac*a, frac*b, frac*c) 
		

class Bond(object):
	def __init__(self, At1, At2, Molecule):
		self.at1 = At1
		self.at2 = At2
		self.mol = Molecule
		At1.add_bond(self)
		At2.add_bond(self)
		Molecule.add_bond(self)
		self.bondLength = self.length()
		self.type = bond_type(self.at1, self.at2)
#		self.springK = 0.0
#		self.springLength = self.bondLength
		
	def length(self):
		dist = distance( self.at1, self.at2 )
		return dist
		
	def second(self, At):
		if At is self.at1:
			return self.at2
		elif At is self.at2:
			return self.at1
		
#	def set_spring(self, newK, newLength):
#		self.springK = newK
#		self.springLength = newLength
#		
#	def unset_spring(self):
#		self.k = 0.0
#		self.springLength = self.bondLength
		
class Vibration(object):
	def __init__(self, frequency, intensityIR, intensityRaman, Molecule, vibConf = vibConfig):
		self.frequency = frequency
		self.intensityIR = intensityIR
		self.intensityRaman = intensityRaman
		self.halfwidth = vibConf.halfwidth
		self.disp = set()
		Molecule.add_vib(self)
		
	def add_disp(self, Atom, x, y, z):
		self.disp.add((Atom, x, y, z))
		
	def read_disp(self, Atom):
		for disp in self.disp:
			if disp[0] is Atom:
				return (disp[1], disp[2], disp[3])
		
	
def read_xyz(inputFile):
	with open(inputFile, 'r') as f:
		content = f.readlines()
	atomCount = content[0].split()
	atomCount = int(atomCount[0])
	molName = content[1].split()
	molName = molName[0]
	mol = Molecule(molName)
	for i in range(2, atomCount+2):
		ln = content[i].split()
		atomId = i-1
		atomName = ln[0] + str(atomId)
		atomType = ln[0]
		atomX = float(ln[1])
		atomY = float(ln[2])
		atomZ = float(ln[3])
		Atom(atomId, atomName, atomType, atomX, atomY, atomZ, mol)
		
	return mol
	
def read_gout(inputFile, vibs = 0, vibUt = vibUtils):
	with open(inputFile, 'r') as f:
		content = f.readlines()
	for line in content:
		ln = line.split()
		if 0 < len(ln): 
			if ln[0] == 'Formula':
				molName = ln[2]
			if 2 < len(ln):
				if ln[2] == 'irreducible':
					atomCount = int(ln[5])
	mol = Molecule(molName)
	geometry = "initial"
	for index, line in enumerate(content):
		if line[0:20] == "  Final cartesian co":
			geometry = "final"
			firstAtom = index + 6
	if geometry == "initial":
		for index, line in enumerate(content):
			if line[0:20] == "  Cartesian coordina":
				firstAtom = index + 6
	# for loading visibility - start
	sys.stdout.write("Fetching atoms data: ")
	pList = set()
	pStep = int(atomCount / 10)
	counter = 0
	for i in range(1, 11):
		pList.add(i*pStep)
	# for loading visibility - end
	for i in range(firstAtom, firstAtom + atomCount):
		counter += 1;
		ln = content[i].split()
		atomId = int(ln[0])
		atomName = ln[1] + str(ln[0])
		atomType = ln[1]
		atomX = float(ln[3])
		atomY = float(ln[4])
		atomZ = float(ln[5])
		Atom(atomId, atomName, atomType, atomX, atomY, atomZ, mol)
		if counter in pList:
			sys.stdout.write('.')
			sys.stdout.flush()
	print(" OK")
	for index, line in enumerate(content):
		if line[0:27] == "  Final charges from ReaxFF":
			sys.stdout.write("Fetching atomic charges data: ")
			firstAtom = index + 5
			counter = 0
			for i in range(firstAtom, firstAtom + atomCount):
				counter += 1
				ln = content[i].split()
				atomId = int(ln[0])
				atomCharge = float(ln[2])
				for atom in mol.atoms:
					if atom.id == atomId:
						atom.charge = atomCharge
				if counter in pList:
					sys.stdout.write('.')
					sys.stdout.flush()
			print(" OK")
	if vibs > 0:
		vibUtils.read_vibs(mol, inputFile, vibs, 1)
		
	return mol
		
def write_gin(mol, outputFile, glp = gulp):
	case = outputFile.split('.')
	case = case[0]
	content = ''
	content += '{} \ntitle \n'.format(glp.header)
	content += mol.name + '\n'
	content += 'end \ncart \n'
	for atom in mol.atoms:
		if atom.charge == "na":
			content += '{0:4s} {1: 3.6f} {2: 3.6f} {3: 3.6f}\n'.format(atom.type, atom.x, atom.y, atom.z)
		else:
			content += '{0:4s} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f}\n'.format(atom.type, atom.x, atom.y, atom.z, atom.charge)
	content += 'maxcyc {} \nlibrary {} \n'.format(glp.maxCycles, glp.library)
	content += glp.output + ' ' + case
	with open(outputFile, 'w') as f:
		f.write(content)

def write_xyz(mol, outputFile):
	case = outputFile.split('.')
	case = case[0]
	content = ''
	content += str(len(mol.atoms)) + '\n'
	content += mol.name + '\n'
	for atom in mol.atoms:
		content += '{0:4s} {1: 3.6f} {2: 3.6f} {3: 3.6f}\n'.format(atom.type, atom.x, atom.y, atom.z)
	content += '\n\n'
	with open(outputFile, 'w') as f:
		f.write(content)
		
def im(inputFile, vibs = 0):
	chunks = inputFile.split('.')
	extension = chunks[-1]
	if extension == 'gout':
		mol = read_gout(inputFile, vibs)
	elif extension == 'gin':
		mol = read_gin(inputFile)
	elif extension == 'xyz':
		mol = read_xyz(inputFile)
	return mol

def join(mol1b, mol2b, bondType):
	print('Connecting new mer to the system...')
	# mol1 = copy.deepcopy(mol1b)
	# mol2 = copy.deepcopy(mol2b)
	mol1 = mol_copy(mol1b)
	mol2 = mol_copy(mol2b)
	start = time.time()
	if len(mol1.bonds) == 0:
		mol1.set_bonds()
	if len(mol2.bonds) == 0:
		mol2.set_bonds()
	transVec = mol1.activate(bondType)
	mol2.activate(bondType)
	molOut = mol1.adjust(mol2)
	molOut.set_bonds()
	molOut.translation( -transVec[0], -transVec[1], -transVec[2])
	end = time.time()
	print('Connecting has finished in ' + str(end - start) + ' s. ')
	return molOut
	
def condense(molIn, method = "default"):
	print('Checking for possible condensation in ' + molIn.name)
	# mol = copy.deepcopy(molIn)
	mol = mol_copy(molIn)
	found = 0
	for bond in mol.bonds:
		if found == 1:
			break
		if bond.type == 'O-H':
			if bond.at1.type == "H":
				atHPri = bond.at1
				atOPri = bond.at2
			elif bond.at2.type == "H":
				atHPri = bond.at2
				atOPri = bond.at1
			neighbours = set()
			for atom in mol.atoms:
				if atom is not atHPri and distance(atom, atHPri) < 1.87:
					neighbours.add(atom)
			if len(neighbours) == 2:
				# print("New candidate: " + atHPri.name)
				for atom in neighbours:
					# print(atom.name)
					if atom is not atOPri and atom.type == 'O':
						atOSec = atom
						for priBond in atOPri.bonds:
							if priBond.type == 'Si-O':
								if priBond.at1 is not atOPri:
									atSiPri = priBond.at1
								elif priBond.at2 is not atOPri:
									atSiPri = priBond.at2
						for secBond in atOSec.bonds:
							if secBond.type == 'O-H':
								found = 1
								if secBond.at1 is not atOSec:
									atHSec = secBond.at1
								elif secBond.at2 is not atOSec:
									atHSec = secBond.at2
							if secBond.type == 'Si-O':
								if secBond.at1 is not atOSec:
									atSiSec = secBond.at1
								elif secBond.at2 is not atOSec:
									atSiSec = secBond.at2
						# check if found connection is between neigbouring Si - ...Si-O-Si...
						try:
							for priBond in atSiPri.bonds:
								for secBond in atSiSec.bonds:
									if priBond.at1 is atSiPri:
										nSiPri = priBond.at2
									else:
										nSiPri = priBond.at1
									if secBond.at1 is atSiSec:
										nSiSec = secBond.at2
									else:
										nSiSec = secBond.at1
									if nSiPri is nSiSec:
										found = 0
						except UnboundLocalError:
							print("Problem with finding atSiPri...")
	if found == 1:
		print('Found new connection!')
		mol.del_atom(atHPri)
		mol.del_atom(atOPri)
		mol.del_atom(atHSec)
		mol.del_atom(atOSec)
		if method == "connector":
			x, y, z = find_connector(atSiPri, atSiSec)
		else:
			x = ( atSiPri.x + atSiSec.x ) / 2
			y = ( atSiPri.y + atSiSec.y ) / 2
			z = ( atSiPri.z + atSiSec.z ) / 2
		newO = Atom(len(mol.atoms) + 1, "O", "O", x, y, z, mol)
		if method == "connector":
			atSiPri.move_toward(newO, 0.2)
			atSiSec.move_toward(newO, 0.2)
		mol.set_bonds()
		return mol
	else:
		print('Nothing was found.')
		return 0
	
def rubbish_check(mol):
	print('Checking for rubbish...')
	rubbish = 0
	for atom in mol.atoms:
		if atom.type == 'O' and len(atom.bonds) > 2:
			rubbish = "Too many bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'H' and len(atom.bonds) > 1:
			rubbish = "Too many bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'Si' and len(atom.bonds) > 4:
			rubbish = "Too many bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'C' and len(atom.bonds) > 4:
			rubbish = "Too many bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
	for bond in mol.bonds:
		if bond.type == "H-H" or bond.type == "O-O" or bond.type == "Si-Si"  or bond.type == "C-C":
			rubbish = "Strange bond occurence: " + bond.type + ", length: " + str(bond.bondLength)
			return rubbish
			
def free_atom_check(mol):
	print('Checking for loose ends...')
	rubbish = 0
	for atom in mol.atoms:
		if atom.type == 'O' and len(atom.bonds) < 2:
			rubbish = "Too little bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'H' and len(atom.bonds) < 1:
			rubbish = "Too little bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'Si' and len(atom.bonds) < 4:
			rubbish = "Too little bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
		elif atom.type == 'C' and len(atom.bonds) < 4:
			rubbish = "Too little bonds in atom " + atom.name + ': '
			for bond in atom.bonds:
				rubbish += bond.type + '(' + str(bond.bondLength) + ')' + ' '
			return rubbish
	for bond in mol.bonds:
		if bond.type == "H-H" or bond.type == "O-O" or bond.type == "Si-Si"  or bond.type == "C-C":
			rubbish = "Strange bond occurence: " + bond.type + ", length: " + str(bond.bondLength)
			return rubbish
	
def run_gulp(ginFile, goutFile):
	print('Starting GULP run: ' + ginFile.name)
	start = time.time()
	p = subprocess.Popen(['gulp'], stdin=ginFile, stdout=goutFile)
	out = p.wait()
	end = time.time()
	print('GULP calculation has finished in ' + str(end - start) + ' s. ')
	return out

def run_gulp_para(ginFileName, nProc):
	caseName = ginFileName.split('.')
	print('Starting parallel (' + str(nProc) + ' threads) GULP run: ' + caseName[0])
	start = time.time()
	p = subprocess.Popen(['mpirun', '-np', str(nProc), GULP_SRC + '/gulp', caseName[0]])
	out = p.wait()
	end = time.time()
	print('GULP calculation has finished in ' + str(end - start) + ' s. ')
	return out
	
def amorph_gen(inputLine, bondType, method="connector"):
	inputText = inputLine
	print(inputText)
	inputText = inputText.split()
	print(inputText)
	if len(inputText) < 2 or len(inputText) % 2 == 1:
		print("Input is incorrect, try again:")
		amorph_gen(bondType)
	else:
		structures = set()
		structureList = []
		for i in range(0, len(inputText), 2):
			m = im(inputText[i])
			m.set_bonds()
			structures.add(m)
			for j in range(0,int(inputText[i+1])): 
				structureList.append(m.id)
				print("Adding to list: " + m.name)
		currentMol = structureList.pop(0)
		random.shuffle(structureList)
		for structure in structures:
			if currentMol == structure.id:
				# currentMol = copy.deepcopy(structure)
				currentMol = mol_copy(structure)
				print("Starting from " + currentMol.name)
				break
		count = len(structureList)
		for i in range(0, count):
			print('\n\nStep: ' + str(i+2))
			molAdd = structureList.pop()
			for structure in structures:
				if molAdd == structure.id:
					# molAdd = copy.deepcopy(structure)
					molAdd = mol_copy(structure)
					print("In this step is added: " + molAdd.name)
					break
			for trial in range(0, 10):
				newMol = join(currentMol, molAdd, bondType)
				rubbish = rubbish_check(newMol)
				if not rubbish:
					currentMol = newMol
					print("New structure found in trial no. " + str(trial+1))
					break
				else:
					print("Trial no. " + str(trial+1) + " has failed due to: " + rubbish)
					if trial == 9:
						print("Limit of trials has been reached.")
						return currentMol
			condensed = condense(currentMol, method)
			if condensed:
				currentMol = condensed
				ginFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gin"
				goutFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gout"
			else:
				ginFileName = "amorph_" + "{:03d}".format(i+2) + ".gin"
				goutFileName = "amorph_" + "{:03d}".format(i+2) + ".gout"
			xyzInFileName = "amorph_" + "{:03d}".format(i+2) + "_0.xyz"
			write_xyz(currentMol, xyzInFileName)
			write_gin(currentMol, ginFileName)
			ginFile = open(ginFileName, 'r')
			goutFile = open(goutFileName, 'w')
			out = run_gulp(ginFile, goutFile)
			ginFile.close()
			goutFile.close()
			if out != 0:
				print("error in GULP")
				break
			else:
				currentMol = read_gout(goutFileName)
				currentMol.set_bonds()
				rubbish = rubbish_check(currentMol)
				if rubbish:
					print(rubbish)
					return rubbish
				# condensed = condense(currentMol)
				# if condensed:
					# newMol = condensed
					# ginFileName = "amorph_" + str(i+2) + "_c.gin"
					# goutFileName = "amorph_" + str(i+2) + "_c.gout"
					# write_gin(newMol, ginFileName)
					# ginFile = open(ginFileName, 'r')
					# goutFile = open(goutFileName, 'w')
					# out = run_gulp(ginFile, goutFile)
					# ginFile.close()
					# goutFile.close()
					# newMol = read_gout(goutFileName)
					# newMol.set_bonds()
					# rubbish = rubbish_check(newMol)
					# if not rubbish:
						# currentMol = newMol
						# print("New condensed structure is used.")
					# else:
						# print('Error after condensation: ' + rubbish + '\nPrimary structure is used.')
		return currentMol

def amorph_gen_para(inputLine, bondType, nProc, method="connector"):
	inputText = inputLine
	print(inputText)
	inputText = inputText.split()
	print(inputText)
	if len(inputText) < 2 or len(inputText) % 2 == 1:
		print("Input is incorrect, try again:")
		amorph_gen(bondType)
	else:
		structures = set()
		structureList = []
		for i in range(0, len(inputText), 2):
			m = im(inputText[i])
			m.set_bonds()
			structures.add(m)
			for j in range(0,int(inputText[i+1])): 
				structureList.append(m.id)
				print("Adding to list: " + m.name)
		currentMol = structureList.pop(0)
		random.shuffle(structureList)
		for structure in structures:
			if currentMol == structure.id:
				# currentMol = copy.deepcopy(structure)
				currentMol = mol_copy(structure)
				print("Starting from " + currentMol.name)
				break
		count = len(structureList)
		for i in range(0, count):
			print('\n\nStep: ' + str(i+2))
			molAdd = structureList.pop()
			for structure in structures:
				if molAdd == structure.id:
					# molAdd = copy.deepcopy(structure)
					molAdd = mol_copy(structure)
					print("In this step is added: " + molAdd.name)
					break
			for trial in range(0, 10):
				newMol = join(currentMol, molAdd, bondType)
				rubbish = rubbish_check(newMol)
				if not rubbish:
					currentMol = newMol
					print("New structure found in trial no. " + str(trial+1))
					break
				else:
					print("Trial no. " + str(trial+1) + " has failed due to: " + rubbish)
					if trial == 9:
						print("Limit of trials has been reached.")
						return currentMol
			condensed = condense(currentMol, method)
			if condensed:
				currentMol = condensed
				ginFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gin"
				goutFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gout"
			else:
				ginFileName = "amorph_" + "{:03d}".format(i+2) + ".gin"
				goutFileName = "amorph_" + "{:03d}".format(i+2) + ".gout"
			xyzInFileName = "amorph_" + "{:03d}".format(i+2) + "_0.xyz"
			write_xyz(currentMol, xyzInFileName)
			write_gin(currentMol, ginFileName)
			out = run_gulp_para(ginFileName, nProc)
			if out != 0:
				print("error in GULP")
				break
			else:
				currentMol = read_gout(goutFileName)
				currentMol.set_bonds()
				rubbish = rubbish_check(currentMol)
				if rubbish:
					print(rubbish)
					return rubbish
				# condensed = condense(currentMol)
				# if condensed:
					# newMol = condensed
					# ginFileName = "amorph_" + str(i+2) + "_c.gin"
					# goutFileName = "amorph_" + str(i+2) + "_c.gout"
					# write_gin(newMol, ginFileName)
					# ginFile = open(ginFileName, 'r')
					# goutFile = open(goutFileName, 'w')
					# out = run_gulp(ginFile, goutFile)
					# ginFile.close()
					# goutFile.close()
					# newMol = read_gout(goutFileName)
					# newMol.set_bonds()
					# rubbish = rubbish_check(newMol)
					# if not rubbish:
						# currentMol = newMol
						# print("New condensed structure is used.")
					# else:
						# print('Error after condensation: ' + rubbish + '\nPrimary structure is used.')
		return currentMol

def amorph_gen_no_gulp(inputLine, bondType):
	# inputText = input("Please, give names of input structures files followed by their quantities, separated with spaces:")
	inputText = inputLine
	print(inputText)
	inputText = inputText.split()
	print(inputText)
	if len(inputText) < 2 or len(inputText) % 2 == 1:
		print("Input is incorrect, try again:")
		amorph_gen(bondType)
	else:
		structures = set()
		structureList = []
		for i in range(0, len(inputText), 2):
			m = im(inputText[i])
			m.set_bonds()
			structures.add(m)
			for j in range(0,int(inputText[i+1])): 
				structureList.append(m.id)
				print("Adding to list: " + m.name)
		currentMol = structureList.pop(0)
		random.shuffle(structureList)
		for structure in structures:
			if currentMol == structure.id:
				# currentMol = copy.deepcopy(structure)
				currentMol = mol_copy(structure)
				print("Starting from " + currentMol.name)
				break
		count = len(structureList)
		for i in range(0, count):
			print('\n\nStep: ' + str(i+2))
			molAdd = structureList.pop()
			for structure in structures:
				if molAdd == structure.id:
					# molAdd = copy.deepcopy(structure)
					molAdd = mol_copy(structure)
					print("In this step is added: " + molAdd.name)
					break
			for trial in range(0, 10):
				newMol = join(currentMol, molAdd, bondType)
				rubbish = rubbish_check(newMol)
				if not rubbish:
					currentMol = newMol
					print("New structure found in trial no. " + str(trial+1))
					break
				else:
					print("Trial no. " + str(trial+1) + " has failed due to: " + rubbish)
					if trial == 9:
						print("Limit of trials has been reached.")
						return currentMol
			condensed = condense(currentMol)
			if condensed:
				currentMol = condensed
				# ginFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gin"
				# goutFileName = "amorph_" + "{:03d}".format(i+2) + "_c.gout"
			# else:
				# ginFileName = "amorph_" + "{:03d}".format(i+2) + ".gin"
				# goutFileName = "amorph_" + "{:03d}".format(i+2) + ".gout"
			xyzInFileName = "amorph_" + "{:03d}".format(i+2) + "_0.xyz"
			write_xyz(currentMol, xyzInFileName)
			# write_gin(currentMol, ginFileName)
			# ginFile = open(ginFileName, 'r')
			# goutFile = open(goutFileName, 'w')
			# out = run_gulp(ginFile, goutFile)
			# ginFile.close()
			# goutFile.close()
			# if out != 0:
				# print("error in GULP")
				# break
			# else:
				# currentMol = read_gout(goutFileName)
			currentMol.set_bonds()
			rubbish = rubbish_check(currentMol)
			if rubbish:
				print(rubbish)
				return rubbish
				# condensed = condense(currentMol)
				# if condensed:
					# newMol = condensed
					# ginFileName = "amorph_" + str(i+2) + "_c.gin"
					# goutFileName = "amorph_" + str(i+2) + "_c.gout"
					# write_gin(newMol, ginFileName)
					# ginFile = open(ginFileName, 'r')
					# goutFile = open(goutFileName, 'w')
					# out = run_gulp(ginFile, goutFile)
					# ginFile.close()
					# goutFile.close()
					# newMol = read_gout(goutFileName)
					# newMol.set_bonds()
					# rubbish = rubbish_check(newMol)
					# if not rubbish:
						# currentMol = newMol
						# print("New condensed structure is used.")
					# else:
						# print('Error after condensation: ' + rubbish + '\nPrimary structure is used.')
		return currentMol
		
def ag(inputLine):
	start = time.time()
	amorph_gen(inputLine, 'O-H')
	end = time.time()
	print('\n\nStructure built in ' + str(end - start) + ' s. ')

def agng(inputLine):
	start = time.time()
	amorph_gen_no_gulp(inputLine, 'O-H')
	end = time.time()
	print('\n\nStructure built in ' + str(end - start) + ' s. ')
	
def agp(inputLine, cfg = config):
	start = time.time()
	mol = amorph_gen_para(inputLine, cfg.connectionType, cfg.nProc)
	end = time.time()
	print('\n\nStructure built in ' + str(end - start) + ' s. ')
	return mol
	
def saturate(mol, method="connector"):
	iter = 0
	currentMol = mol_copy(mol)
	while 1:
		iter += 1
		condensed = condense(currentMol, method)
		if condensed:
			newMol = condensed
			ginFileName = "sat_" + "{:03d}".format(iter) + ".gin"
			xyzFileName = "sat_" + "{:03d}".format(iter) + "_0.xyz"
			goutFileName = "sat_" + "{:03d}".format(iter) + ".gout"
			write_gin(newMol, ginFileName)
			write_xyz(newMol, xyzFileName)
			ginFile = open(ginFileName, 'r')
			goutFile = open(goutFileName, 'w')
			out = run_gulp(ginFile, goutFile)
			ginFile.close()
			goutFile.close()
			newMol = read_gout(goutFileName)
			newMol.set_bonds()
			rubbish = rubbish_check(newMol)
			if not rubbish:
				currentMol = newMol
				print("New condensed structure is used.")
			else:
				print('Error after condensation: ' + rubbish + '\nPrimary structure is used.')
		else:
			return currentMol
			
def saturate_para(mol, cfg = config):
	iter = 0
	currentMol = mol_copy(mol)
	while 1:
		iter += 1
		condensed = condense(currentMol, cfg.saturationMethod)
		if condensed:
			newMol = condensed
			ginFileName = "sat_" + "{:03d}".format(iter) + ".gin"
			xyzFileName = "sat_" + "{:03d}".format(iter) + "_0.xyz"
			goutFileName = "sat_" + "{:03d}".format(iter) + ".gout"
			write_gin(newMol, ginFileName)
			write_xyz(newMol, xyzFileName)
			out = run_gulp_para(ginFileName, cfg.nProc)
			newMol = read_gout(goutFileName)
			newMol.set_bonds()
			rubbish = rubbish_check(newMol)
			if not rubbish:
				currentMol = newMol
				print("New condensed structure is used.")
			else:
				print('Error after condensation: ' + rubbish + '\nPrimary structure is used.')
		else:
			return currentMol
	
def sp(mol, cfg = config):
	currentMol = saturate_para(mol, cfg)
	return currentMol
	

