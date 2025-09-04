# This is a Jython script calculated the persistent homology of a clique dictionary.
# It has been developed to work with a weighted clique filtration, but it
# can adapted to be used to any series of persistent homology dictionaries
#
# NOTE: THIS CODE RUNS IN JYTHON, NOT PYTHON!!!

import pickle
import sys
import os
from collections import defaultdict, OrderedDict
import Holes
from subprocess import Popen, PIPE
import subprocess
import base64
import codecs
import sys
import io
import gc
import json


def list2simplexes(list, dim):
    num = dim + 1
    simplexes = []
    for i in range(0, len(list), num):
        simplexes.append(list[i:i + num]);
    return simplexes

# print('This code needs as input:');
# print('1) full filtration data',clique_dic_file)
# print('2) the maximum homology dimension to calculate',dimension)
# print('3) the directory name for output',Dir)
# print('4) the tag name for the output files',stringie)
# print('6) the full path to your javaplex directory',javaplex_path)


print(sys.argv)
data = raw_input()  #  This is jython command that is the analogous of "input" in python
data = data.replace("\'", "\"")


Clique_dictionary = json.loads(data, object_pairs_hook=OrderedDict)
del data
gc.collect()

dimension = int(sys.argv[1])
Dir = str(sys.argv[2])
stringie = str(sys.argv[3])
javaplex_path = str(sys.argv[4])
save_generators = bool(sys.argv[5])


edge_weights = list([float(weights[1])
                     for weights in list(Clique_dictionary.values())])


# # In our case we want it sorted in ascending order
edge_weights = list(sorted(edge_weights))
# NOTE: you need to put here the path to the javaPlex distribution on your system
libs = [
    os.path.join(javaplex_path, 'javaplex.jar')
]
# print(libs)

for s in libs:
    sys.path.append(s)
# print(sys.path)

import edu.stanford.math.plex4
import edu.stanford.math.plex4.api
import edu.stanford.math.plex.Persistence


Complex = edu.stanford.math.plex4.api.Plex4.createExplicitSimplexStream()
max_index = 0
print("Clique dictionary: parsing started.");
for key in Clique_dictionary:
    original_key = key
    # print(original_key)
    key = str(key)
    key = key.strip('[]')
    key = key.split(', ')
    key_buona = []
    for n in range(len(key)):
        # print(key[n])
        key_buona.append(int(float(eval(key[n]))))
    if len(key_buona) == 1:
        Complex.addVertex(key_buona[0], 0)
    else:
        Complex.addElement(key_buona, int(Clique_dictionary[original_key][0]))
        if int(Clique_dictionary[original_key][0]) > max_index:
            max_index = int(Clique_dictionary[original_key][0])

print("Parsing over. Closing now.")
Complex.finalizeStream()
print("Complex is valid? ", Complex.validateVerbose())
print("Size of complex filtration:", Complex.getSize());
max_filtration_value = max_index
pH = edu.stanford.math.plex4.api.Plex4.getModularSimplicialAlgorithm(
    dimension + 1, 2)
print("Starting pH calculation...")
complex_computation = pH.computeIntervals(Complex)
print("Done!")
print("Results incoming:")
# infinite_barcodes = complex_computation.getInfiniteIntervals()
annotated_intervals = pH.computeAnnotatedIntervals(Complex)
# betti_numbers_string = infinite_barcodes.getBettiNumbers()
# print('The betti numbers are:', betti_numbers_string);
# print('while the annotated intervals are: \n', annotated_intervals);


#####################
# NEW: Betti curves and final vector from curves 
# Filtration length is the last arrival index + 1
L = int(max_index) + 1

# Build Betti curves by counting active intervals per filtration index
betti_curves = {}
for h in range(0, dimension + 1):
    arr = [0] * L
    # intervals are paired 1–1 with generators; we only need the intervals
    list_intervals = list(annotated_intervals.getIntervalsAtDimension(h))
    for iv in list_intervals:
        # Robust parse to indices: convert to string " [b, d) " or " [b, infinity) "
        parts = str(iv).split(',')
        b_str = parts[0].strip(' [')
        d_str = parts[1].strip(' )')
        # Convert to integer indices in 0..L; replace infinity with L
        b = int(float(b_str))
        d = L if (d_str == 'infinity') else int(float(d_str))
        if b < 0: b = 0
        if d > L: d = L
        # Add +1 on [b, d) (inclusive of b, exclusive of d)
        for k in range(b, d):
            arr[k] += 1
    betti_curves[h] = arr

# Persist curves and the final vector (A from B)
import json as _json
betti_curves_dir = os.path.join(Dir, 'betti_curves')
if not os.path.exists(betti_curves_dir):
    os.makedirs(betti_curves_dir)

with open(os.path.join(betti_curves_dir, 'betti_curves_' + str(stringie) + '.json'), 'w') as bf1:
    _json.dump(betti_curves, bf1)

final_betti_from_curves = {h: (betti_curves[h][-1] if len(betti_curves[h]) > 0 else 0)
                           for h in betti_curves}
with open(os.path.join(betti_curves_dir, 'betti_final_from_curves_' + str(stringie) + '.json'), 'w') as bf2:
    _json.dump(final_betti_from_curves, bf2)

#  Betti numbers from Javaplex infinite intervals 
infinite_barcodes = complex_computation.getInfiniteIntervals()
# Jython returns a Java int[]; convert to a Python list of ints
betti_numbers_vec = [int(x) for x in list(infinite_barcodes.getBettiNumbers())]

betti_dir = os.path.join(Dir, 'betti')
if not os.path.exists(betti_dir):
    os.makedirs(betti_dir)

with open(os.path.join(betti_dir, 'betti_numbers_' + str(stringie) + '.json'), 'w') as bf3:
    _json.dump({"betti": betti_numbers_vec}, bf3)

# sanity print 
print('Betti numbers (Javaplex infinite intervals):', betti_numbers_vec)
print('Betti final from curves (should match):',
      [final_betti_from_curves.get(h, 0) for h in range(0, dimension + 1)])
#  END new Betti outputs
############



# # Here we save the full generator dictionary and save the interval files in order to be
# # able to reopen them later for other purposes, for example comparison of random and null
# # models..


gendir = os.path.join(Dir, 'gen')
if not os.path.exists(gendir):
    os.makedirs(gendir)

Generator_dictionary = {}
import re
import string
for h in range(1, dimension + 1):
    Generator_dictionary[h] = []
    list_gen = list(annotated_intervals.getGeneratorsAtDimension(h))
    # if save_generators==True:
    # 	gen_details_file=open(gendir+'details_generators_'+str(h)+'_'+str(stringie)+'.pck','w');
    # 	pickle.dump(list_gen,gen_details_file);

    list_intervals = list(annotated_intervals.getIntervalsAtDimension(h))
    for n, key in enumerate(list_gen):
        test = str(list_intervals[n]).split(',')
        test[0] = test[0].strip(' [')
        test[1] = test[1].strip(' )')
        if test[1] == 'infinity':
            test[1] = str(max_filtration_value)
        line = str(key)
        line = line.translate(string.maketrans('', ''), '-[]')
        line = re.sub('[,+ ]', ' ', line)
        line = line.split()
        # tempcycle=Holes.Cycle(h,list2simplexes(line,h),test[0],test[1]);
        # print(test[0],test[1])
        # print()
        tempcycle = Holes.Cycle(h, list2simplexes(line, h), edge_weights[int(
            float(test[0]))], edge_weights[int(float(test[1]))])
        Generator_dictionary[h].append(tempcycle)
        del tempcycle
    for cycle in Generator_dictionary[h]:
        cycle.summary()

filename = os.path.join(gendir, 'generators_' + str(stringie) + '.pck')
with open(filename, 'wb') as generator_dict_file:
    pickle.dump(Generator_dictionary, generator_dict_file)
print('Generator dictionary dumped to ' + filename)

