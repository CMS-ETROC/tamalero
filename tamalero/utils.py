import math
import numpy as np
from itertools import combinations
from time import sleep
from yaml import load, dump
import os

from tamalero.KCU import KCU

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

here = os.path.dirname(os.path.abspath(__file__))

def get_temp(v_out, v_ref, r_ref, t_1, r_1, b, celcius=True, thermistor=None):
    """
    Calculate the temperature of a thermistor, given the voltage measured on it.

    Arguments:
    v_out (float) -- voltage measured on the thermistor
    v_ref (float) -- reference voltage
    r_ref (float) -- volatge divider resistor
    t_1 (float) -- reference temperature of thermistor
    r_1 (float) -- resistance of NTC at reference temperature
    b (float) -- B coefficient, with B = (ln(r_1)-ln(r_t)) / (1/t_1 - 1/t_out)

    Keyword arguments:
    celcius (bool) -- give and return the temperature in degree celcius. Kelvin scale used otherwise.
    """

    delta_t = 273.15 if celcius else 0
    try:
        r_t = r_ref / (v_ref/v_out - 1)
        # print(r_1, r_t, v_ref, v_out)
        if thermistor is not None:
            find_temp = temp_res_fit(thermistor=thermistor)
            t_2 = find_temp(np.log10(r_t))
        else:
            t_2 = b/((b/(t_1+delta_t)) - math.log(r_1) + math.log(r_t))
    except ZeroDivisionError:
        print ("Temperature calculation failed!")
        return -999
    except ValueError:
        print("Negative resistance values not allowed in temperature calculation")
        r_t = r_ref / (v_ref/v_out - 1)
        print(f"""
              \tr_1   (float) -- resistance of NTC at reference temperature: {r_1}\n
              \tr_ref (float) -- volatge divider resistor:                   {r_ref}\n
              \tv_out (float) -- voltage measured on the thermistor:         {v_out}\n
              \tv_ref (float) -- reference voltage:                          {v_ref}\n
              \tr_t = r_ref / (v_ref/v_out - 1)\n
              \t    = {r_ref} / ({v_ref}/{v_out}-1)\n
              \t    = {r_ref/(v_ref/v_out-1)}\n
              """)
        return -999
    return t_2-delta_t

def get_temp_direct(v_out, curr_dac, thermistor="NTCG063JF103FTB", celcius=True, verbose=False):
    """
    Calculate the temperature of a thermistor, given the voltage measured on it.

    Arguments:
    v_out (float) -- voltage measured on the thermistor
    curr_dac (float) -- current source value set on the DAC (in uA)
    thermistor (str="NTCG063JF103FTB") -- thermistor used for temperature calculation

    Keyword arguments:
    celcius (bool) -- give and return the temperature in degree celcius. Kelvin scale used otherwise.
    """

    find_temp = temp_res_fit(thermistor=thermistor)

    r_t = v_out / (curr_dac / 10**6)
    t = find_temp(np.log10(r_t))

    delta_t = 0 if celcius else 273.15

    if verbose:
        print(f"Thermistor: {thermistor}")
        print(f"Voltage: {v_out} \t Current: {curr_dac} uA")
        print(f"Resistance: {r_t}")
        print(f"Temperature: {t}")

    return t+delta_t

def temp_res_fit(thermistor="NTCG063JF103FTB", power=2):

    T_ref = 25
    if thermistor=="NTCG063JF103FTB":
        B_list = [3194, 3270, 3382, 3422]
        T_list = [-25, 0, 50, 75]
        R_ref = 10e3
    elif thermistor=="NTCG063UH103HTBX":
        B_list = [3770, 3822, 3900, 3926]
        T_list = [-25, 0, 50, 75]
        R_ref = 10e3
    elif thermistor=="NCP03XM102E05RL":
        B_list = [3500, 3539, 3545, 3560]
        T_list = [50, 80, 85, 100]
        R_ref = 1e3
    elif thermistor=="NTCS0603E3272FMT":
        B_list = [3600, 3600]
        T_list = [25, 85]
        R_ref = 2.7e3
    else:
        raise ValueError(f"Only thermistors NTCG063JF103FTB, NTCG063UH103HTBX, NTCS0603E3272FMT or NCP03XM102E05RL are currently allowed, but {thermistor} was passed.")

    R_list = []

    for B, T in zip(B_list, T_list):
        R = R_ref * math.exp(-B * ((1/298.15) - (1/(T+273.15))))
        R_list.append(R)

    if thermistor=="NTCG063JF103FTB" or thermistor=="NTCG063UH103HTBX" or thermistor=='NTCS0603E3272FMT':
        T_list.insert(2, T_ref)     # Reference temperature of thermistor
        R_list.insert(2, R_ref)     # Reference resistance of NTC at reference temperature
    elif thermistor=="NCP03XM102E05RL":
        T_list = [T_ref] + T_list
        R_list = [R_ref] + R_list
    
    poly_coeffs = np.polyfit(np.log10(R_list), T_list, power)
    fit = np.poly1d(poly_coeffs)

    return fit

def read_mapping(f_in, selection='adc', flavor='small'):
    flavors = {'small':0, 'medium':1, 'large': 2}
    i_flavor = flavors[flavor]
    with open(f_in) as f:
        mapping = load(f, Loader=Loader)[selection]
    return {v:mapping[v] for v in mapping.keys() if flavors[mapping[v]['flavor']] <= flavors[flavor]}

def dump_alignment_to_file(rb, f_out):
    res = rb.dump_uplink_alignment()
    with open(f_out, 'w') as f:
        dump(res, f, Dumper=Dumper)

def load_alignment_from_file(f_in):
    with open(f_in, 'r') as f:
        res = load(f, Loader=Loader)
    return res

def load_yaml(f_in):
    with open(f_in, 'r') as f:
        res = load(f, Loader=Loader)
    return res

def ffs(x):
    '''
    Returns the index, counting from 0, of the
    least significant set bit in `x`.
    from https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
    There really is no better way!
    '''
    return (x&-x).bit_length()-1

def bit_count(x):
    '''
    get number of bits from a mask. this is ugly, python 3.10 has `int .bit_count()`
    '''
    return bin(x).count('1')

def prbs_phase_scan(lpgbt, f_out='phase_scan.txt'):
    with open(f_out, "w") as f:
        for phase in range(0x0, 0x1ff, 1):
            phase_ns = (50.0*(phase&0xf) + 800.0*(phase>>4))/1000
            lpgbt.set_ps0_phase(phase)
            lpgbt.reset_pattern_checkers()
            sleep(0.5)
            #read_pattern_checkers()
            prbs_errs = lpgbt.read_pattern_checkers(quiet=True)[0]
            s = ("{} "*(len(prbs_errs)+1)).format(*([phase_ns]+prbs_errs))
            f.write("%s\n" % s)
            print (s)


def plot_phase_scan(f_in, channel):
    import matplotlib.pyplot as plt
    data = np.loadtxt(f_in)
    plt.yscale("log")
    plt.plot(data[:,0], data[:,channel])
    plt.show()


def header(configured=False):
    from tamalero.colors import magenta, green
    col = green if configured else magenta
    try:
        print(col("\n\n\
        ████████╗ █████╗ ███╗   ███╗ █████╗ ██╗     ███████╗███████╗\n\
        ╚══██╔══╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔════╝██╔════╝\n\
           ██║   ███████║██╔████╔██║███████║██║     █████╗  ███████╗\n\
           ██║   ██╔══██║██║╚██╔╝██║██╔══██║██║     ██╔══╝  ╚════██║\n\
           ██║   ██║  ██║██║ ╚═╝ ██║██║  ██║███████╗███████╗███████║\n\
           ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝\n\n\
        "))
    except UnicodeEncodeError:
        print (col("\n\n\
        #########################\n\
        #######  TAMALES  #######\n\
        #########################\n\n\
        "))


def make_version_header(res):
    from tamalero.colors import blue
    print ("\n ### Testing ETL Readout Board: ###")
    print (blue("- Version: %s.%s"%(res["rb_ver_major"], res["rb_ver_minor"])))
    print (blue("- Flavor: %s"%res["rb_flavor"]))
    print (blue("- Serial number: %s"%res["serial_number"]))
    print (blue("- lpGBT version: %s"%res["lpgbt_ver"]))
    print (blue("- lpGBT serial number: %s"%res['lpgbt_serial']))
    print (blue("- Trigger lpGBT mounted: %s"%res['trigger']))
    print ("")


def chunk(in_list, n):
    return [in_list[i * n:(i + 1) * n] for i in range((len(in_list) + n - 1) // n )] 

def get_last_commit_sha(version):
    import requests
    import json


    r2 = requests.get(f"https://gitlab.cern.ch/api/v4/projects/107856/repository/commits?ref=devel")
    log = json.loads(r2.content)
    last_commit_sha = log[0]['id'][:7]
    return last_commit_sha

def download_address_table(version, quiet=False):
    import os
    import requests
    import json
    import urllib.parse

    if not os.path.isdir(os.path.expandvars(f"$TAMALERO_BASE/address_table/{version}")):
        last_commit_sha = get_last_commit_sha(version)
        r = requests.get(f"https://gitlab.cern.ch/api/v4/projects/107856/repository/tree?ref={version}&&path=address_tables&&recursive=True")
        tree = json.loads(r.content)
        if isinstance(tree, list):
            if not quiet:
                print ("Successfully got list of address table files from gitlab.")
        else:
            version = last_commit_sha
            if os.path.isdir(os.path.expandvars(f'$TAMALERO_BASE/address_table/{version}/')):
                # already downloaded.
                return version
            r = requests.get(f"https://gitlab.cern.ch/api/v4/projects/107856/repository/tree?ref=devel&&path=address_tables&&recursive=True")
            tree = json.loads(r.content)
            print (f"Local firmware version detected. Will download address table corresponding to commit {version}.")

        if not quiet:
            print (f"Downloading latest firmware version address table to address_table/{version}")
            print(f"Making directory: {os.path.expandvars('$TAMALERO_BASE')}/address_table/{version}")
        os.makedirs(os.path.expandvars(f"$TAMALERO_BASE/address_table/{version}"))
        for f in tree:
            if f['type'] == 'tree':
                os.makedirs(os.path.expandvars(f"$TAMALERO_BASE/address_table/{version}/{f['name']}"))
            elif f['type'] == 'blob':
                # needs URL encode: https://www.w3schools.com/tags/ref_urlencode.ASP
                path = urllib.parse.quote_plus(f['path']).replace('.', '%2E')  # python thinks . is fine, so we replace it manually
                res = requests.get(f"https://gitlab.cern.ch/api/v4/projects/107856/repository/files/{path}/raw?ref={version}")
                local_path = f['path'].replace('address_tables/', '')
                open(os.path.expandvars(f"$TAMALERO_BASE/address_table/{version}/{local_path}"), 'wb').write(res.content)

    return version

def check_repo_status(kcu_version=None):
    import requests
    import json
    import os
    from git import Repo
    from tamalero.colors import red, green
    from emoji import emojize

    # get remote repo log
    r = requests.get(f"https://gitlab.cern.ch/api/v4/projects/110883/repository/commits")
    log = json.loads(r.content)
    last_commit_sha = log[0]['id']

    # get local log
    working_tree_dir = '/' + os.path.join(*(here.split('/')[:-1]))
    repo = Repo(working_tree_dir)
    hashes = [ c.hexsha for c in repo.iter_commits(max_count=50) ]
    tags = [ t.name.strip('v') for t in repo.tags ]

    #
    commit_based = (last_commit_sha in hashes)
    tag_based = kcu_version in tags if kcu_version is not None else True

    if commit_based and tag_based:
        print (green("Your tamalero repository is up-to-date with master"))
    else:
        print ( emojize("\n:warning: :warning: ") + red(" WARNING: You are potentially working on an outdated or out-of-sync version of tamalero ") + emojize(" :warning: :warning:"))
        if not tag_based:
            print (red(f"You are using KCU firmware version {kcu_version}, but the corresponding tag has not been found in your local tamalero repo."))
            print (red(f"You can ignore this warning for firmware versions BEFORE 1.3.5\n"))
        else:
            print (red("Please pull a more recent version from gitlab.\n"))

def get_kcu(kcu_address, control_hub=True, host='localhost', verbose=False, quiet=False):
    # Get the current firmware version number
    if verbose:
        if control_hub:
            print(f"Using control hub on host={host}, kcu_address={kcu_address}")
        else:
            print(f"NOT using control hub on host={host}, kcu_address={kcu_address}")

    import uhal
    import time
    if control_hub:
        ipb_path = f"chtcp-2.0://{host}:10203?target={kcu_address}:50001"
    else:
        ipb_path = f"ipbusudp-2.0://{kcu_address}:50001"
    if not quiet:
        print (f"IPBus address: {ipb_path}")

    trycnt = 0
    while (True):
        try:
            kcu_tmp = KCU(name="tmp_kcu",
                        ipb_path=ipb_path,
                        adr_table=os.path.expandvars(f"$TAMALERO_BASE/address_table/generic/etl_test_fw.xml"))
            break
        except uhal.exception or uhal._core.exception:
            if control_hub:
                # we could be checking if control hub is running earlier, but since the control hub path is hardcoded
                # I only want to do it if really necessary. if controlhub is running happily from another directory, that's fine too
                try:
                    control_hub_running = os.popen("/opt/cactus/bin/controlhub_status").readlines()[0].count("ControlHub is up")
                except IndexError:
                    control_hub_running = False
                    print(f"Trying to run using controlhub, but it seems to not be installed in /opt/cactus/bin/")

                if control_hub_running:
                    # if control hub is running, try again to establish a connection
                    pass
                else:
                    raise RuntimeError("Controlhub is not running. Start it with: /opt/cactus/bin/controlhub_start")

            trycnt += 1
            time.sleep(1)
            if (trycnt > 10):
                if control_hub: print("controlhub status:", "running" if control_hub_running else "not running")
                raise RuntimeError(f"Could not establish connection with KCU board {ipb_path}")

        #raise
    xml_sha     = kcu_tmp.get_xml_sha()
    if verbose:
        print (f"Address table hash: {xml_sha}")

    #last_commit = get_last_commit_sha(xml_sha)
    #if not os.path.isdir(f"address_table/{last_commit}"):
    xml_sha = download_address_table(xml_sha, quiet=quiet)
    #else:
    #    xml_sha = last_commit

    kcu = KCU(name="my_device",
              ipb_path=ipb_path,
              adr_table=os.path.expandvars(f"$TAMALERO_BASE/address_table/{xml_sha}/etl_test_fw.xml"))

    data = 0xabcd1234
    kcu.write_node("LOOPBACK.LOOPBACK", data)
    if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
        raise RuntimeError(f"No communication with KCU board {ipb_path} established.")

    kcu.get_firmware_version(string=False)

    if not quiet:
        print(f"KCU firmware version: {kcu.firmware_version['major']}.{kcu.firmware_version['minor']}.{kcu.firmware_version['patch']}")

    return kcu

def get_config(config, version='v2', verbose=False):
    default_cfg = load_yaml(os.path.join(here, f'../configs/rb_default_{version}.yaml'))
    if config != 'default':
        updated_cfg = load_yaml(os.path.join(here, f'../configs/{config}_{version}.yaml'))
        for chip in ['SCA', 'LPGBT', 'MUX64']:
            if chip not in updated_cfg:
                continue
            for interface in ['adc', 'gpio', 'channel']:
                if interface not in updated_cfg[chip]:
                    continue
                if updated_cfg[chip][interface] is not None:
                    for k in updated_cfg[chip][interface]:
                        if verbose:
                            print(f"\n - Updating configuration for {chip}, {interface}, {k} to:")
                            print(updated_cfg[chip][interface][k])
                        if k in default_cfg[chip][interface] and k not in updated_cfg[chip][interface]:
                            del default_cfg[chip][interface][k]
                        default_cfg[chip][interface][k] = updated_cfg[chip][interface][k]
        if 'inversions' in updated_cfg:
            for links in ['trigger', 'clocks', 'downlink', 'uplink']:
                default_cfg['inversions'][links] = updated_cfg['inversions'][links]
        if 'modules' in updated_cfg:
            default_cfg['modules'] = updated_cfg['modules']
        if 'mux64' in updated_cfg:
            default_cfg['mux64'] = updated_cfg['mux64']
    return default_cfg

def majority_vote(values, majority=None):
    from functools import reduce
    if majority is None:
        majority = len(values)-1
    combs = combinations(range(len(values)), majority)
    votes = []
    for comb in combs:
        #print(comb)
        tmp_list = [values[i] for i in comb]
        votes.append(reduce(lambda x, y: x & y, tmp_list))

    #print(votes)
    return reduce(lambda x, y: x | y, votes)

if __name__ == '__main__':
    print ("Temperature example:")
    print (get_temp(0.8159, 1.5, 10000, 25, 10000, 3900))
