# run
-------------

This markdown documents some of the run commands that are frequently used within the cross correlation
pipeline. Note that some things must be commented out or in of `corrutils.py`; not everything is fully automated due to using DR1/DR2
in different settings...

## Fiducial n(z)
----------------

### HSC:
--------
```bash
OUTDIR="-o outputs/correction/autos_HSC" && DEFAULT_FLAGS="-ns 50 -r1 15 -r2 30 -s 0 -c 200 -z -j" && setcc && python run_corr.py -t1 HSC
```

### DESI:
--------
```bash
# NGC:
OUTDIR="-o outputs/dr1PIP/autos_NGC" && DEFAULT_FLAGS="-s 0 -ns 100 -re 256 -j -k -a 1 -w PIP" && setcc && python run_corr.py $OUTDIR -t1 ELG_LOPnotqso $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 LRG $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 BGS_ANY $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 QSO $DEFAULT_FLAGS
# SGC:
OUTDIR="-o outputs/dr1PIP/autos_SGC" && DEFAULT_FLAGS="-s 0 -ns 100 -re 256 -j -k -a 3 -w PIP" && setcc && python run_corr.py $OUTDIR -t1 ELG_LOPnotqso $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 LRG $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 BGS_ANY $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 QSO $DEFAULT_FLAGS
```

### DESIxHSC:
--------
```bash
### DR1: ###
# A :
OUTDIR="-o outputs/dr1PIP/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -z -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 LRG
# B :
OUTDIR="-o outputs/dr1PIP/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -z -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 BGS_ANY && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 ELG_LOPnotqso
### DR2: ###
# A :
OUTDIR="-o outputs/dr2PIP/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -z -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 LRG
# B :
OUTDIR="-o outputs/dr2PIP/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -z -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 BGS_ANY && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 ELGnotqso
```

## Correcting photoz galaxy bias
--------------------------------

### HSC:
--------
```bash
OUTDIR="-o outputs/correction/autos_HSC" && DEFAULT_FLAGS="-ns 50 -r1 15 -r2 30 -s 0 -c 200 -z -j" && setcc && python run_corr.py -t1 HSC
```

### DESI:
--------
```bash
#NGC:
OUTDIR="-o outputs/calibrationPIP/dr1/autos_NGC" && DEFAULT_FLAGS="-s 0 -ns 100 -re 256 -j -k -a 1 -w PIP" && setcc && python run_corr.py $OUTDIR -t1 ELG_LOPnotqso $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 LRG $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 BGS_ANY $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 QSO $DEFAULT_FLAGS
#SGC:
OUTDIR="-o outputs/calibrationPIP/dr1/autos_SGC" && DEFAULT_FLAGS="-s 0 -ns 100 -re 256 -j -k -a 3 -w PIP" && setcc && python run_corr.py $OUTDIR -t1 ELG_LOPnotqso $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 LRG $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 BGS_ANY $DEFAULT_FLAGS && python run_corr.py $OUTDIR -t1 QSO $DEFAULT_FLAGS
```


### DESIxHSC:
--------
```bash
### DR1: ###
# A :
OUTDIR="-o outputs/calibrationPIP/dr1/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 LRG
# B :
OUTDIR="-o outputs/calibrationPIP/dr1/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 BGS_ANY && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 ELG_LOPnotqso
### DR2: ###
# A :
OUTDIR="-o outputs/calibrationPIP/dr2/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 LRG
# B :
OUTDIR="-o outputs/calibrationPIP/dr2/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -j -w PIP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 BGS_ANY && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 ELGnotqso
```

### Bin 4 QSO (appendix B):
--------
```bash
OUTDIR="-o outputs/calibration_bin4_qso/dr1/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -z -j -w nonKP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO
```

### Bin 5 (appendix B):
--------
```bash
OUTDIR="-o outputs/calibration_bin5/dr2/cross" && DEFAULT_FLAGS="-t2 HSC -ns 100 -s 0 -j -w nonKP" && setcc && python run_corr.py $OUTDIR $DEFAULT_FLAGS -t1 QSO
```


## Simulations :
-----------------
Code can also run on the simulated buzzard mocks. See below for an example.

```bash
# Autos :
OUTDIR="outputs/sims/auto_j64_ns256" && setcc && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 ELG_LOPnotqso -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 LRG -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 BGS_ANY -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 HSC -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 

# Cross : 
OUTDIR="outputs/sims2/cross_j64_ns256" && setcc && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r1 3 -r2 3 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r1 3 -r2 3 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 ELG_LOPnotqso -t2 HSC -ns 64 -re 256 -r1 3 -r2 3
```