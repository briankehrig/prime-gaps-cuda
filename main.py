import datetime as dt
import json
import math
import os
import subprocess as subp
import sys

# TODO: Detect shared memory size
# TODO: Detect duplicated work units with different device IDs
# TODO: Make a way to save progress for long work units in case it crashes

WORKTODO_FILE = "worktodo.txt"
SETTINGS_FILE = "settings.json"
MAIN_CUDA_FILE = "prime_gaps.cu"
DEVICE_PROPERTIES_FILE = "device_properties.cu"
def getLastParamsFile(deviceId):
    return f"_LAST_PARAMS_{deviceId}"
def getCompiledCudaFile(deviceId):
    return f"prime_gaps_{deviceId}.out"
def getFilenameSuffix(start, end, minGap):
    return f"{start}e12_{end}e12_{minGap}.txt"
def getReportFileName(start, end, minGap):
    return f"reports/GapReport_" + getFilenameSuffix(start, end, minGap)
def getLogFileName(start, end, minGap):
    return f"logs/log_" + getFilenameSuffix(start, end, minGap)

class Style:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    NOCOLOR = '\033[39m'

    BRBLACK = '\033[90m'
    BRRED = '\033[91m'
    BRGREEN = '\033[92m'
    BRYELLOW = '\033[93m'
    BRBLUE = '\033[94m'
    BRMAGENTA = '\033[95m'
    BRCYAN = '\033[96m'
    BRWHITE = '\033[97m'

    WIPE_LINE = '\33[2K\r'

    @staticmethod
    def CUSTOMCOL(val):
        return f'\033[38;5;{val}m'
    @staticmethod
    def RGB(r,g,b):
        return f'\033[38;2;{r};{g};{b}m'

def printWarn(thing, *args, **kwargs):
    print(Style.YELLOW+thing+Style.RESET, *args, **kwargs)

def printError(thing, *args, **kwargs):
    print(Style.BRRED+thing+Style.RESET, *args, **kwargs)

def getStdoutWhileRunning(cmd):
    with subp.Popen(cmd, stdout=subp.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            yield line
    if p.returncode != 0:
        raise subp.CalledProcessError(p.returncode, p.args)

def stringRepresents(string):
    try:
        int(string)
        return "int"
    except ValueError:
        pass

    try:
        float(string)
        return "float"
    except ValueError:
        pass
    
    return "string"

def getDeviceInfo():
    deviceInfo = []
    if not os.path.exists("device_properties.out") or \
      os.path.getmtime(DEVICE_PROPERTIES_FILE) > os.path.getmtime("device_properties.out"):
        print(f"Compiling {DEVICE_PROPERTIES_FILE} with command: "
              f"{Style.BRYELLOW}'nvcc {DEVICE_PROPERTIES_FILE} -o device_properties.out'{Style.RESET}")
        subp.run(["nvcc", DEVICE_PROPERTIES_FILE, "-o", "device_properties.out"])
    
    for line in getStdoutWhileRunning(["./device_properties.out"]):
        line = line.split()
        if line[0] == "NewDevice":
            deviceInfo.append({})
        else:
            represents = stringRepresents(line[1])
            if represents == "int":
                deviceInfo[-1][line[0]] = int(line[1])
            elif represents == "float":
                deviceInfo[-1][line[0]] = float(line[1])
            else:
                deviceInfo[-1][line[0]] = " ".join(line[1:]).upper()
    return deviceInfo

def removeDoubleSlashComments(string):
    lines = string.split('\n')
    output = []
    for line in lines:
        output.append(line.split('//')[0])
    return '\n'.join(output)

def getRecommendedParameters(deviceInfo, targetMemoryUsage, settings):
    '''
    small prime wheels = 4, medium primes = 4096
    word length = 240
    gpu blocks: take cuda cores, divide out all powers of 2, that is the starting amount of blocks
        then, multiply by 2 until blocks > 128 AND blocks*threads >= total cuda cores
    shared size: ??? experimentally, 8192 is good for GPUs like mine,
        but maybe 10240 or 11264 or 12288 is better for bigger ones
    Block size = roughly biggest that can fit into memory while being a "nice" multiple of shared size * blocks

    the following parameters need to be experimentally determined:
    PROPORTION_OF_BLOCKS_FOR_SIEVING is either 0.5 or 0.75 (0.5 for Titan V)
    SHARED_SIZE_WORDS is somewhere between 8192 and 12288, (12288 for Titan V)
    NUM_MEDIUM_PRIMES_BASE is usually from 3072 to 5120, but it's 8192 for Titan V (it must be a multiple of 512)
    SMALL_PRIME_WHEELS is usually 4, but it's sometimes 3 (it's 4 for Titan V)

    '''
    # We have to handle WORD_LENGTH separately and first
    # because the ACTUAL value of WORD_LENGTH (not just the recommended value)
    # affects the recommendations of other parameters
    recommended = {"WORD_LENGTH": settings["KernelOptions"]["WORD_LENGTH"]}
    recommended["WORD_LENGTH"] = recommended["WORD_LENGTH"] if recommended["WORD_LENGTH"] != -1 else 240

    if "TITAN V" in deviceInfo["Name"]:
        recommended["SHARED_SIZE_WORDS"] = 12288
    elif "GTX 1660" in deviceInfo["Name"]:
        recommended["SHARED_SIZE_WORDS"] = 8192
    elif "RTX 4090" in deviceInfo["Name"]:
        recommended["SHARED_SIZE_WORDS"] = 10240
    else:
        recommended["SHARED_SIZE_WORDS"] = 12288

    if any(x in deviceInfo["Name"] for x in ("RTX 4070", "RTX 4090", "RTX 5080")):
        recommended["SMALL_PRIME_WHEELS"] = 3
    else:
        recommended["SMALL_PRIME_WHEELS"] = 4
        
    if "GTX 1660" in deviceInfo["Name"]:
        recommended["NUM_MEDIUM_PRIMES_BASE"] = 4096
    elif any(x in deviceInfo["Name"] for x in ("RTX 3080", "RTX 4070")):
        recommended["NUM_MEDIUM_PRIMES_BASE"] = 10240
    elif "RTX 5080" in deviceInfo["Name"]:
        recommended["NUM_MEDIUM_PRIMES_BASE"] = 12288
    else:
        recommended["NUM_MEDIUM_PRIMES_BASE"] = 8192

    recommended["PROPORTION_OF_BLOCKS_FOR_SIEVING"] = 0.5

    gpuBlocks = deviceInfo["CUDACores"]
    while (gpuBlocks % 2 == 0): gpuBlocks //= 2
    while (gpuBlocks <= 128 or gpuBlocks*512 < deviceInfo["CUDACores"]): gpuBlocks *= 2
    recommended["GPU_BLOCKS"] = gpuBlocks

    # we divide targetMemoryUsage by 2 since we will have 2 lists in memory at the same time
    # we first subtract 1.5 GB since that's about how much the small prime wheels take up
    blockSize = int((deviceInfo["GlobalMemGB"]-1.5)*2**30 * targetMemoryUsage/2) * recommended["WORD_LENGTH"]//4
    blockSize -= blockSize % (recommended["SHARED_SIZE_WORDS"] * recommended["WORD_LENGTH"] * recommended["GPU_BLOCKS"])
    recommended["BLOCK_SIZE"] = blockSize

    # TODO: The above calculation needs to depend on the ACTUAL value of SHARED_SIZE_WORDS, not the recommended value!!
    '''
    4070: 3 small wheels is better, 10240 medium, speed 640B/s
    '''


    return recommended

def sanityCheckParameters(parameters, minGap):
    if minGap % (parameters["WORD_LENGTH"] // 4) != 0:
        printError(f"ERROR: minGap must be a multiple of {parameters['WORD_LENGTH'] // 4} = WORD_LENGTH/4")
        sys.exit(1)

    if minGap < 960 and parameters["WORD_LENGTH"] != 120:
        printWarn(f"WARNING: If WORD_LENGTH={parameters['WORD_LENGTH']} and minGap<960, the search is likely to be very slow. "
                  f"\nNote: Try setting WORD_LENGTH=120.")

    elif minGap < 1200 and parameters["WORD_LENGTH"] != 120:
        printWarn(f"WARNING: If WORD_LENGTH={parameters['WORD_LENGTH']} and minGap<1200, the search could be slow (if you have a fast GPU). "
                  f"Note: Try setting WORD_LENGTH=120.")
    
    if minGap >= 1200 and parameters["WORD_LENGTH"] == 120:
        printWarn(f"WARNING: With a large minGap (>=1200) you can (and should) optimize speed by setting WORD_LENGTH=240.")

    if parameters["SMALL_PRIME_WHEELS"] not in (3,4):
        printWarn(f"WARNING: SMALL_PRIME_WHEELS should always be either 3 or 4")

    if parameters["SHARED_SIZE_WORDS"] not in (8192,9216,10240,11264,12288):
        printWarn(f"WARNING: SHARED_SIZE_WORDS should be 8-12 times a multiple of 1024 to optimize speed")

    if parameters["NUM_MEDIUM_PRIMES_BASE"] % 512:
        printError(f"ERROR: NUM_MEDIUM_PRIMES_BASE must be a multiple of 512")
        sys.exit(1)

    if parameters["PROPORTION_OF_BLOCKS_FOR_SIEVING"] not in (0.5,0.75):
        printWarn(f"WARNING: PROPORTION_OF_BLOCKS_FOR_SIEVING should be 0.5 (sometimes 0.75) to optimize speed")

    if parameters["PROGRESS_EVERY"] != 1:
        printError(f"ERROR: PROGRESS_EVERY must be 1")
        sys.exit(1)

    if parameters["RUN_FROM_PYTHON"] != 1:
        printError(f"ERROR: RUN_FROM_PYTHON must be 1")
        sys.exit(1)

def progressBar(length, progress):
    filled = min(length, int(progress*(length+1)))
    return f"{Style.BRGREEN}{'0'*filled}{Style.WHITE}{Style.DIM}{'.'*(length-filled)}{Style.RESET}"

def formatETA(seconds):
    x = int(seconds)
    s = x % 60 # seconds
    x //= 60
    m = x % 60 # minutes
    x //= 60
    h = x % 24 # hours
    x //= 24
    d = x # days
    if d>9999: return f">9999d"
    if d>99: return f"{d}d{h:02d}h"
    if d: return f"{d}d{h:02d}h{m:02d}m"
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def printProgress(proportionDone, currentlyAt, top5, speed, eta):
    width, _ = os.get_terminal_size()
    print(f"{Style.WIPE_LINE}{proportionDone*100:5.2f}%{' ' if proportionDone<1 else ''}"
          f"[{progressBar(width-92, proportionDone)}] "
          f"{Style.BRBLUE}At:{Style.RESET} {currentlyAt:.7e} "
          f"{Style.BRMAGENTA}Best:{Style.RESET} {' '.join(top5)} "
          f"{Style.BRYELLOW}Speed:{Style.RESET}{speed:7.2f} B/s "
          f"{Style.BRRED}ETA:{Style.RESET} {formatETA(eta)}",
          end=''
    )

def needToRecompile(parameters, deviceIdx):
    fname = getLastParamsFile(deviceIdx)
    if not os.path.exists(fname): return True
    if os.path.getmtime(MAIN_CUDA_FILE) > os.path.getmtime(fname): return True
    with open(fname) as f:
        return json.loads(f.read()) != parameters

def recompile(parameters, deviceIdx):
    if needToRecompile(parameters, deviceIdx):
        command = f'nvcc {MAIN_CUDA_FILE} -o {getCompiledCudaFile(deviceIdx)}'
        for param, value in parameters.items():
            command += f" -D{param}={value}"
        
        print(f"Compiling {MAIN_CUDA_FILE} with command: '{Style.BRYELLOW}{command}{Style.RESET}'")
        p = subp.run(command.split(), stdout=subp.PIPE, bufsize=1, universal_newlines=True)
        if p.returncode != 0:
            raise subp.CalledProcessError(p.returncode, p.args)
        with open(getLastParamsFile(deviceIdx), "w") as f:
            f.write(json.dumps(parameters))

def runOne(parameters, start, end, minGap, deviceIdx, logFilename, startTime, skippedBlocks=0, startTop5=None):
    if start >= end: return []
    start -= start % parameters["WORD_LENGTH"]
    blocksToTest = (end - start - 1) // parameters["BLOCK_SIZE"] + 1 + skippedBlocks
    totalSpeed = 0
    top5 = ["----"] * 5 if startTop5 is None else startTop5
    allResults = []
    sanityCheckParameters(parameters, minGap)

    def printLog(data):
        with open(logFilename, "a") as f:
            f.write(data)

    if not os.path.exists(logFilename):
        printLog(f"{deviceIdx}\n{json.dumps(parameters)}\n{startTime}\n")
    dataToLog = ""

    for line in getStdoutWhileRunning(
        ['./'+getCompiledCudaFile(deviceIdx), str(minGap), str(start),
         str(blocksToTest - skippedBlocks), str(deviceIdx)]
    ):
        #print(line, end='')
        line = line.split()
        if line[0] == "Progress":
            blocksDone = int(line[1]) + skippedBlocks
            currentlyAt = int(line[2])
            speed = float(line[3]) if len(line)>3 else 0
            if blocksDone >= blocksToTest-1:
                # the last 2 blocks are a lot faster due to how the code works, so we ignore their speed
                speed = totalSpeed
            if blocksDone - skippedBlocks:
                speedFactor = 1/(blocksDone - skippedBlocks)**0.5
                totalSpeed = speedFactor*speed + (1-speedFactor)*totalSpeed
            eta = 0 if speed==0 else parameters["BLOCK_SIZE"] * (blocksToTest - blocksDone) / (speed * 1e9)
            printProgress(blocksDone/blocksToTest, currentlyAt, top5, totalSpeed, eta)
            if blocksDone % 10 == 0:
                dataToLog += f"Progress {currentlyAt} {blocksDone} {blocksToTest}\n"
                printLog(dataToLog)
                dataToLog = ""
        elif line[0] == ":":
            p = int(line[1])
            gap = int(line[2])
            merit = float(line[3])
            top5 = sorted(top5+[f"{gap:4d}"], reverse=True, key=lambda x: int(x) if x != "----" else 0)[:5]
            if p < end:
                newResult = (gap, merit, p)
                if newResult[2] == 0:
                    # on my GPU, this happens if minGapSize is 360 or lower with WORD_LENGTH==120
                    raise "ERROR: Something weird happened, maybe you set minGapSize too low?"
                allResults.append((gap, merit, p))
                dataToLog += f"Gap {gap} {merit:.4f} {p}\n"
        elif line[0] == "ERROR:":
            print(' '.join(line))
    return allResults

def formatDatetime(dt):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return f"{dt.date().year:4d}-{months[dt.date().month-1]}-{dt.date().day:02d} " \
           f"{dt.time().hour:02d}:{dt.time().minute:02d}:{dt.time().second:02d}.{dt.time().microsecond//1000:03d} UTC"

def writeOutputFile(parameters, start, end, minGap, results, reportOptions, startTime, endTime, gpuName):
    fname = getReportFileName(start, end, minGap)

    kernelParams = "\n".join(f"    {key}={value}" for key, value in parameters.items())

    hundredStats = ""
    x = ((minGap-1)//100+1)*100 # smallest multiple of 100 that's >=minGap
    while True:
        count = sum(1 for r in results if r[0] >= x)
        if count: hundredStats += f"Gaps >={x}: {count}\n"
        else: break
        x += 100
    if hundredStats: hundredStats = "\n" + hundredStats
    
    if results:
        largest = [r for r in results if r[0] == max(x[0] for x in results)]

        largest.sort(key=lambda x: x[2])
        if reportOptions["SORT_OUTPUT_BY_GAPSIZE"]:
            results.sort(key=lambda x: x[0], reverse=True)
        else:
            results.sort(key=lambda x: x[2])
        # we recalculate the merit here, because we don't want to round it twice and get incorrect results
        largestStr = "\n".join(f"{r[0]:{len(str(largest[0][0]))}d} {r[0]/math.log(r[2]):7.4f} {r[2]}" for r in largest)
        resultsStr = "\n".join(f"{r[0]:{len(str(largest[0][0]))}d} {r[0]/math.log(r[2]):7.4f} {r[2]}" for r in results)
    else:
        largestStr = "(No gaps)"
        resultsStr = "(No gaps)"
        
    contents = f"""======== PRIME GAP REPORT ========

Name: {reportOptions['NAME']}
Start time: {formatDatetime(startTime)}
End time: {formatDatetime(endTime)}
GPU: {gpuName}
Kernel parameters:
{kernelParams}

Target gap size: {minGap}
Range searched: {start}e12 - {end}e12
{hundredStats}
Largest gap:
{largestStr}

Full list of gaps >= {minGap}:
{resultsStr}
"""
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    with open(fname, "w") as f:
        f.write(contents)

def main():
    if len(sys.argv) < 2:
        print("Device index not specified, defaulting to 0. Try running 'python3 prime_gaps.py <deviceIdx>'")
        deviceIdx = 0
        runMode = 'run'
    elif len(sys.argv) > 3:
        print("Usage: 'python3 prime_gaps.py [deviceIdx] [mode]'")
        print("'mode' can be one of ('run', 'forcerun', 'continue'). Defaults to 'run'.")
        return
    else:
        deviceIdx = int(sys.argv[1])
        runMode = 'run' if len(sys.argv) <= 2 else sys.argv[2]
        if runMode not in ("run", "forcerun", "continue"):
            print(f"Unknown mode {runMode} is not one of ('run', 'forcerun', 'continue')")
            return
    
    if runMode == "forcerun":
        print(f"{Style.YELLOW}Forcerun will overwrite the log file"
              f" and gap report file of any previous overlapping runs.")
        result = input(f"Are you sure you wish to continue? (input 'y') {Style.RESET}")
        if result.lower() != "y": return
    
    deviceInfo = getDeviceInfo()
    if deviceIdx >= len(deviceInfo):
        print(f"Device index {deviceIdx} is out of range. Detected {len(deviceInfo)} devices total.")
        return
    
    with open(SETTINGS_FILE) as f:
        settings = json.loads(removeDoubleSlashComments(f.read()))
    parameters = getRecommendedParameters(deviceInfo[deviceIdx], 0.9, settings)

    for setting in settings["KernelOptions"]:
        if settings["KernelOptions"][setting] != -1:
            parameters[setting] = settings["KernelOptions"][setting]

    parameters["PROGRESS_EVERY"] = 1 # this should never be changed
    parameters["RUN_FROM_PYTHON"] = 1 # this should never be changed

    
    if runMode == "continue":
        for logfile in os.listdir("logs"):
            logfile = logfile[4:] # remove 'log_' at the start
            if os.path.exists(f"reports/GapReport_{logfile}"): continue

            with open(f"logs/log_{logfile}") as f:
                logdata = f.read().split('\n')
            newDeviceIdx = int(logdata[0])
            if newDeviceIdx != deviceIdx:
                continue # not for us to do
            parameters = json.loads(logdata[1])

            recompile(parameters, newDeviceIdx)
            start, end, minGap = logfile.split("_")
            start, end, minGap = int(start.split("e")[0]), int(end.split("e")[0]), int(minGap.split(".")[0])

            results = [] # repopulate results
            for line in logdata:
                line = line.split()
                if not line: continue
                if line[0] == "Gap":
                    results.append((int(line[1]), float(line[2]), int(line[3])))
            
            if logdata[-2].startswith("Progress"):
                skippedBlocks = int(logdata[-2].split()[2])
                newStart = int(logdata[-2].split()[1])
            else:
                # the log file doesn't have any Progress lines at all
                skippedBlocks = 0
                newStart = start*10**12

            top5 = sorted([f"{r[0]:4d}" for r in results], reverse=True, key=lambda x: int(x) if x != "----" else 0)[:5]
            top5 += ["----"] * (5-len(top5))
            print(f"Finishing work unit from log file: '{logfile}'")
            startTime = dt.datetime.now(dt.timezone.utc)
            results += runOne(parameters, newStart, end*10**12, minGap,
                              newDeviceIdx, f"logs/log_{logfile}", startTime, skippedBlocks, top5)
            endTime = dt.datetime.now(dt.timezone.utc)
            print("\nSaving to file... ", end='')
            writeOutputFile(parameters, start, end, minGap, results, settings["ReportOptions"],
                            startTime, endTime, deviceInfo[newDeviceIdx]["Name"])
            print("Done")
    
    elif runMode in ("run", "forcerun"):
        with open(WORKTODO_FILE) as f:
            work = f.read().split('\n')
    
        high64 = None
        for line in work:
            line = line.split("#")[0].strip()
            if not line: continue
            try:
                data = list(map(int, line.split(",")))
            except ValueError:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (not a number)")
                continue

            if len(data) > 4:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (too many arguments)")
                continue
            if len(data) < 3:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (too few arguments)")
                continue

            if len(data) == 3: data.append(0) # we default to device index 0 if it's not specified
            if data[3] != deviceIdx:
                continue # this work unit is not for us to do

            start, end, minGap = data[:3]
            if start >= end:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (start must be < end)")
                continue
            if start < 1:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (start must be >=1e12)")
                continue
            if end*10**12 > 2**78:
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' (end must be >2^78)")
                continue
            if minGap % (parameters["WORD_LENGTH"]//4):
                printWarn(f"WARNING: Skipping invalid work unit: '{Style.RED}{line}{Style.YELLOW}' "
                        f"(minGap must be a multiple of {parameters['WORD_LENGTH']//4})")
                continue

            msg = ""
            logFilename = getLogFileName(start, end, minGap)
            if os.path.exists(getReportFileName(start, end, minGap)):
                if runMode == "forcerun":
                    msg = f"{Style.BRCYAN}This work unit was fully completed, rerunning it{Style.RESET}"
                else:
                    print(f"{Style.BRCYAN}Skipping fully completed work unit: '{Style.YELLOW}{line}{Style.BRCYAN}' "
                        f"(redo it using 'main.py <deviceIdx> forcerun){Style.RESET}")
                    continue
            
            if not os.path.exists(os.path.dirname(logFilename)):
                os.makedirs(os.path.dirname(logFilename))
            if os.path.exists(logFilename) and runMode == "forcerun":
                os.remove(logFilename)
            
            if os.path.exists(logFilename):
                if runMode == "forcerun" and not msg:
                    msg = f"{Style.BRCYAN}This work unit was partially completed, restarting it{Style.RESET}"
                else:
                    print(f"{Style.BRCYAN}Skipping partially completed work unit: '{Style.YELLOW}{line}{Style.BRCYAN}' "
                        f"(finish using 'main.py <deviceIdx> continue'){Style.RESET}")
                    continue

            high64_start = (start*10**12) >> 64
            high64_end = (end*10**12) >> 64
            if high64_start == high64_end:
                parameters["HIGH_64"] = high64_start
            else:
                if "HIGH_64" in parameters: parameters.pop("HIGH_64")
                printWarn(f"This work unit sits on a 64-bit border, it might be slower than usual!")

            recompile(parameters, deviceIdx)
            print(f"Running work unit: '{line}' {msg}")
            startTime = dt.datetime.now(dt.timezone.utc)
            results = runOne(parameters, start*10**12, end*10**12, minGap, deviceIdx, logFilename, startTime)
            endTime = dt.datetime.now(dt.timezone.utc)
            print("\nSaving to file... ", end='')
            writeOutputFile(parameters, start, end, minGap, results, settings["ReportOptions"],
                            startTime, endTime, deviceInfo[deviceIdx]["Name"])
            print("Done")

    print()

if __name__ == '__main__':
    main()