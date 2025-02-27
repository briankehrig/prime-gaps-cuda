import json
import math
import os
import subprocess as subp
import sys

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

def getStdoutWhileRunning(cmd):
    with subp.Popen(cmd, stdout=subp.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            yield line
    if p.returncode != 0:
        raise subp.CalledProcessError(p.returncode, p.args)

def getDeviceInfo():
    deviceInfo = []
    for line in getStdoutWhileRunning(["./device_properties.out"]):
        line = line.split()
        currentDeviceNum = None
        if line[0] == "NewDevice":
            deviceInfo.append({})
        else:
            try:
                deviceInfo[-1][line[0]] = int(line[1])
            except ValueError:
                deviceInfo[-1][line[0]] = float(line[1])
    return deviceInfo

def removeDoubleSlashComments(string):
    lines = string.split('\n')
    output = []
    for line in lines:
        output.append(line.split('//')[0])
    return '\n'.join(output)

def getRecommendedParameters(deviceInfo, targetMemoryUsage):
    '''
    small prime wheels = 4, medium primes = 4096
    word length = 240
    gpu blocks: take cuda cores, divide out all powers of 2, that is the starting amount of blocks
        then, multiply by 2 until blocks > 128 AND blocks*threads >= total cuda cores
    shared size: ??? experimentally, 8192 is good for GPUs like mine, but maybe 10240 or 11264 or 12288 is better for bigger ones
    Block size = roughly biggest that can fit into memory while being a "nice" multiple of shared size * blocks

    the following parameters need to be experimentally determined:
    PROPORTION_OF_BLOCKS_FOR_SIEVING is either 0.5 or 0.75 (0.5 for Titan V)
    SHARED_SIZE_WORDS is somewhere between 8192 and 12288, (12288 for Titan V)
    NUM_MEDIUM_PRIMES_BASE is usually from 3072 to 5120, but it's 8192 for Titan V (it must be a multiple of 512)
    SMALL_PRIME_WHEELS is usually 4, but it's sometimes 3 (it's 4 for Titan V)

    '''
    recommended = {"WORD_LENGTH": 240}
    recommended["SHARED_SIZE_WORDS"] = 12288
    recommended["SMALL_PRIME_WHEELS"] = 4
    recommended["NUM_MEDIUM_PRIMES_BASE"] = 8192
    recommended["PROPORTION_OF_BLOCKS_FOR_SIEVING"] = 0.5

    # TODO: THE RECOMMENDED VALUES SHOULD TAKE INTO ACCOUNT BEING OVERWRITTEN BY SETTINGS.JSON!!!!!


    gpuBlocks = deviceInfo["CUDACores"]
    while (gpuBlocks % 2 == 0): gpuBlocks //= 2
    while (gpuBlocks <= 128 or gpuBlocks*512 < deviceInfo["CUDACores"]): gpuBlocks *= 2
    recommended["GPU_BLOCKS"] = gpuBlocks

    # we divide targetMemoryUsage by 2 since we will have 2 lists in memory at the same time
    blockSize = int(deviceInfo["GlobalMemGB"]*2**30 * targetMemoryUsage/2) * recommended["WORD_LENGTH"]//4
    blockSize -= blockSize % (recommended["SHARED_SIZE_WORDS"] * recommended["GPU_BLOCKS"])
    recommended["BLOCK_SIZE"] = blockSize//2

    return recommended

def progressBar(length, progress):
    filled = min(length, int(progress*(length+1)))
    return Style.BRGREEN + '0'*filled + Style.WHITE+Style.DIM + '.'*(length-filled) + Style.RESET

def formatETA(seconds):
    s = int(seconds)
    d = s // 86400
    s -= d * 86400
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    if d: return f"{d}d{h:02d}h{m:02d}m{s:02d}s"
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def printProgress(proportionDone, currentlyAt, top5, speed, eta):
    print(f"{Style.WIPE_LINE}{proportionDone*100:5.2f}%{' ' if proportionDone<1 else ''}"
          f"[{progressBar(25, proportionDone)}] "
          f"{Style.BRBLUE}At:{Style.RESET} {currentlyAt:.7e} "
          f"{Style.BRMAGENTA}Best:{Style.RESET} {' '.join(top5)} "
          f"{Style.BRYELLOW}Speed:{Style.RESET}{speed:7.2f} B/s "
          f"{Style.BRRED}ETA:{Style.RESET} {formatETA(eta)}",
          end=''
    )

def needToRecompile(parameters):
    if not os.path.exists("_LASTPARAMS"): return True
    if os.path.getmtime("prime_gaps.cu") > os.path.getmtime("_LASTPARAMS"): return True
    with open("_LASTPARAMS") as f:
        return json.loads(f.read()) != parameters

def runOne(parameters, start, end, minGap, deviceIdx):
    start -= start % parameters["WORD_LENGTH"]
    blocksToTest = (end - start - 1) // parameters["BLOCK_SIZE"] + 1
    totalSpeed = 0
    top5 = ["----"] * 5
    allResults = []
    for line in getStdoutWhileRunning(
        ["./prime_gaps_py.out", str(minGap), str(start), str(blocksToTest), str(deviceIdx)]
    ):
        #print(line, end='')
        line = line.split()
        if line[0] == "Progress":
            blocksDone = int(line[1])
            currentlyAt = int(line[2])
            speed = float(line[3]) if len(line)>3 else 0
            if blocksDone >= blocksToTest-1:
                # the last 2 blocks are a lot faster due to how the code works, so we ignore their speed
                speed = totalSpeed
            if blocksDone:
                speedFactor = 1/blocksDone**0.5
                totalSpeed = speedFactor*speed + (1-speedFactor)*totalSpeed
            eta = 0 if speed==0 else parameters["BLOCK_SIZE"] * (blocksToTest - blocksDone) / (speed * 1e9)
            printProgress(blocksDone/blocksToTest, currentlyAt, top5, totalSpeed, eta)
        elif line[0] == ":":
            p = int(line[1])
            gap = int(line[2])
            top5 = sorted(top5+[f"{gap:4d}"], reverse=True, key=lambda x: int(x) if x != "----" else 0)[:5]
            if p < end:
                newResult = (gap, float(line[3]), p)
                if newResult[2] == 0:
                    # on my GPU, this happens if minGapSize <= 360
                    raise "ERROR: Something weird happened, maybe you set minGapSize too low?"
                allResults.append((gap, float(line[3]), p))
        elif line[0] == "ERROR:":
            print(' '.join(line))
    return allResults

def writeOutputFile(parameters, start, end, minGap, results, reportOptions):
    '''
    ===== PRIME GAP REPORT =====
    Target gap size: <mingap>
    Range Searched: 
    Gaps >=1200: <x> (or whatever hundred is at least as large as mingap)
    Gaps >=1250: <x>
    Gaps >=1300: <x>
    ... keep going until there are none left
    Largest gap: <size> <merit> <prime>

    Full list of gaps >= <mingap>: # format: <gapsize> <startprime> <merit>
    1572 35.4308 18571673432051830099
    1552 34.9844 18470057946260698231 # (these would be in the opposite order if SORT_OUTPUPT_BY_GAPSIZE=0)
    '''
    fname = f"reports/GapReport_{start}e12_{end}e12_{minGap}.txt"

    kernelParams = "\n".join(f"    {key}={value}" for key, value in parameters.items())

    hundredStats = ""
    x = ((minGap-1)//100+1)*100 # smallest multiple of 100 that's >=minGap
    while True:
        count = sum(1 for r in results if r[0] >= x)
        if count: hundredStats += f"Gaps >={x}: {count}\n"
        else: break
        x += 100
    
    largest = max(results, key=lambda x: x[0])
    largestStr = [r for r in results if r[0] == largest[0]]
    largestStr.sort(key=lambda x: x[2])
    largestStr = "\n".join(" ".join(map(str, r)) for r in largestStr)

    if reportOptions["SORT_OUTPUT_BY_GAPSIZE"]:
        results.sort(key=lambda x: x[0], reverse=True)
    else:
        results.sort(key=lambda x: x[2])
    # we recalculate the merit here, because we don't want to round it twice and get incorrect results

    resultsStr = "\n".join(f"{r[0]:{len(str(largest[0]))}d} {r[0]/math.log(r[2]):7.4f} {r[2]}" for r in results)
        
    contents = f"""======== PRIME GAP REPORT ========

Name: {reportOptions['NAME']}

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
    elif len(sys.argv) > 2:
        print("Usage: 'python3 prime_gaps.py [deviceIdx]'")
        print("See settings.json for more info on configuration.")
        return
    else:
        deviceIdx = int(sys.argv[1])
    parameters = getRecommendedParameters(getDeviceInfo()[0], 0.5)
    with open("settings.json") as f:
        data = removeDoubleSlashComments(f.read())
        settings = json.loads(data)
    for setting in settings["KernelOptions"]:
        if settings["KernelOptions"][setting] != -1:
            parameters[setting] = settings["KernelOptions"][setting]

    parameters["PROGRESS_EVERY"] = 1 # this should never be changed
    parameters["RUN_FROM_PYTHON"] = 1 # this should never be changed

    if needToRecompile(parameters):
        command = 'nvcc prime_gaps.cu -o prime_gaps_py.out'
        for param, value in parameters.items():
            print(f"Using parameter:{Style.BRGREEN} {param}{Style.RESET}={value}")
            command += f" -D{param}={value}"
        
        print(f"Compiling with command: '{command}'")
        p = subp.run(command.split(), stdout=subp.PIPE, bufsize=1, universal_newlines=True)
        if p.returncode != 0:
            raise subp.CalledProcessError(p.returncode, p.args)
        with open("_LASTPARAMS", "w") as f:
            f.write(json.dumps(parameters))
    else:
        print("Skipping recompilation")

    with open("worktodo.txt") as f:
        lines = f.read().split('\n')
    
    for line in lines:
        line = line.split("#")[0].strip()
        if not line: continue
        start, end, minGap = map(int, line.split(","))
        print(f"Running work unit: '{line}'")
        results = runOne(parameters, start*10**12, end*10**12, minGap, deviceIdx)
        print("\nSaving to file... ", end='')
        writeOutputFile(parameters, start, end, minGap, results, settings["ReportOptions"])
        print("done")

    print()

if __name__ == '__main__':
    main()