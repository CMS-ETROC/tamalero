import csv
import re
import yaml
import numpy as np

class hexint(int): pass
def hexint_presenter(dumper, data):
    return dumper.represent_int(hex(data))
yaml.add_representer(hexint, hexint_presenter)

def rangetomask(start, end):
  start = int(start)
  end = int(end)
  mask = 1
  for i in range(end-start):
    mask = (mask<<1) + 1
  return "0x{:02X}".format(mask<<start)

def pix2rc(pix):
  return pix%16, int(np.floor(pix/16))

def parse(row):
    name = re.split('\<|\[', row[0])[0]
    default = hex(int(re.split('b',row[2])[1],2))
    adr = list(filter(None, re.split('\], |\],| \[|\[|\]|:',row[1])))
    if len(adr) == 2 :
      regadr = adr[0]
      shift = adr[1]
      mask = rangetomask(adr[1], adr[1])
    elif len(adr) == 3:
      regadr = adr[0]
      shift = adr[2]
      mask = rangetomask(adr[2], adr[1])
    elif len(adr) == 6:
      regadr = [adr[0], adr[3]]
      shift = [adr[2], adr[5]]
      mask = [rangetomask(adr[2], adr[1]),rangetomask(adr[5], adr[4])]
    elif len(adr) == 5:
      if len(adr[2]) > 2: # 2-3 case
        regadr = [adr[0], adr[2]]
        shift = [adr[1], adr[4]]
        mask = [rangetomask(adr[1], adr[1]),rangetomask(adr[4], adr[3])]
      else: # 3-2 case
        regadr = [adr[0], adr[3]]
        shift = [adr[2], adr[4]]
        mask = [rangetomask(adr[2], adr[1]),rangetomask(adr[4], adr[4])]
    elif len(adr) == 12:
        regadr = [adr[0], adr[3], adr[6], adr[9]]
        shift = [adr[2], adr[5], adr[8], adr[11]]
        mask = [rangetomask(adr[2], adr[1]),rangetomask(adr[5], adr[4]),rangetomask(adr[8], adr[7]),rangetomask(adr[11], adr[10])]
    else:
      print('Something is  wrong!', name)

    return (name, regadr, shift, mask, default)

with open('ETROC2_regs.csv', newline='', encoding='utf-8-sig') as csvfile:
  f = csv.reader(csvfile, delimiter=',')
  dumpdata = {}
  for row in f:
    name, regadr, shift, mask, default = parse(row)

    # now dump the data to a yaml file

    # if register is across many addresses
    if isinstance(regadr, list):
      dumpdata[name] = {
        'shift': [int(s) for s in shift],
        'mask': [hexint(int(m, 16)) for m in mask],
        'default': hexint(int(default, 16)),
        'regadr': {}
      }
      if '<Rn>' in regadr[0]: # if it's an in-pixel register
        for pix in range(256):
          r, c = pix2rc(pix)
          dumpdata[name]['regadr'][pix] = [s.replace('<Rn>', str(r)).replace('<Cn>', str(c)) for s in regadr]
      else: # if it's a peripheral register, convert regadr to int
        dumpdata[name]['regadr'] = [int(i) for i in regadr]

    # if register is just one address, it's easier
    else:
      dumpdata[name] = {
        'shift': int(shift),
        'mask': hexint(int(mask, 16)),
        'default': hexint(int(default, 16)),
        'regadr': {}
      }
      if '<Rn>' in regadr:
        for pix in range(256):
          r, c = pix2rc(pix)
          dumpdata[name]['regadr'][pix] = regadr.replace('<Rn>', str(r)).replace('<Cn>', str(c))
      else:
        dumpdata[name]['regadr'] = int(regadr)
 
# dump register data by name
with open(r'ETROC2.yaml', 'w') as file:
    documents = yaml.dump(dumpdata, file)

# dump default values per register address
dumpregs = {}
with open('ETROC2_def.csv', newline='', encoding='utf-8-sig') as csvfile:
  f = csv.reader(csvfile, delimiter=',')
  for row in f:
    if '<Rn>' in row[0]:
      for pix in range(256):
        r, c = pix2rc(pix)
        regname = re.split('/',row[0])[0].replace('<Rn>', str(r)).replace('<Cn>', str(c))
        dumpregs[regname] = hexint(int(row[1], 16))
    else:
      regname = int(re.split('/',row[0])[0])
      dumpregs[regname] = hexint(int(row[1], 16))
with open(r'ETROC2_def.yaml', 'w') as file:
    documents = yaml.dump(dumpregs, file)
